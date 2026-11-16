"""End-to-end planner: Search proposes -> NN checks marks -> CSP checks schedule."""
from typing import Callable, Mapping, Sequence

from src.search.algorithms import uniform_cost_search, greedy_best_first_search, astar_search
from src.search.heuristics import remaining_mark_gap
from src.csp.domains import build_domains, normalize_slots
from src.csp.backtracking import solve_bt
from src.csp.forward_checking import solve_fc
from src.csp.ac3 import solve_ac3
from .effort_model import (adjust_features_for_effort, describe_effort_assumption, CLICKS_PER_SESSION,
                           ACTIVE_DAYS_PER_SESSION, RESOURCES_PER_SESSION)

SEARCHES = {"ucs": uniform_cost_search, "greedy": greedy_best_first_search,
            "gbfs": greedy_best_first_search, "greedy_best_first": greedy_best_first_search,
            "astar": astar_search, "a*": astar_search}
FAILURE_TARGET = "target_unreachable"      # NN never reaches the target, even with all slots
FAILURE_SCHEDULE = "schedule_infeasible"   # NN says "enough" but the CSP cannot schedule it
MAX_FALLBACK_CSP_CHECKS = 300


def monotone_envelope(raw_values):
    """Planning marks from raw NN marks: mark(n) = max(raw(0..n)), so more effort never lowers it."""
    out, best = [], None
    for v in raw_values:
        best = v if best is None else max(best, v)
        out.append(best)
    return out


def _default_predictor():
    from src.nn.predictor import predict_mark   # lazy: torch/numpy loaded only when used
    return predict_mark


class StudyPlanner:
    """Coordinate candidate allocations and timetable feasibility checking."""

    def __init__(self, predictor: Callable = None, clicks_per_session: int = CLICKS_PER_SESSION,
                 csp_algorithm: str = "ac3", enforce_monotone: bool = True,
                 active_days_per_session: int = ACTIVE_DAYS_PER_SESSION,
                 resources_per_session: int = RESOURCES_PER_SESSION):
        self.predictor = predictor or _default_predictor()
        self.clicks_per_session = clicks_per_session
        self.active_days_per_session = active_days_per_session
        self.resources_per_session = resources_per_session
        self.enforce_monotone = enforce_monotone
        self.csp_algorithm = csp_algorithm.lower()
        if self.csp_algorithm not in {"bt", "fc", "ac3"}:
            raise ValueError("csp_algorithm must be bt, fc, or ac3")

    def plan(self, assignments: Sequence[Mapping], available_slots: Sequence, search_algorithm: str = "ucs"):
        """Return a dict describing the plan, or a clear failure.

        Each assignment needs ``name``, ``target_mark``, ``deadline_week`` and ``features``.
        ``deadline_week`` is the last allowed week (inclusive).
        """
        if not assignments:
            raise ValueError("At least one assignment is required")
        algorithm = search_algorithm.lower()
        if algorithm not in SEARCHES:
            raise ValueError("search_algorithm must be 'ucs', 'greedy' or 'astar'")
        slots = normalize_slots(available_slots)
        targets = [float(a["target_mark"]) for a in assignments]
        deadlines = [int(a["deadline_week"]) for a in assignments]
        feature_sets = [dict(a["features"]) for a in assignments]
        max_sessions = [sum(s.week <= d for s in slots) for d in deadlines]

        # ---- Two separate series per assignment:
        #   RAW      = what the NN actually predicted for n sessions (never edited, reported as-is)
        #   PLANNING = raw passed through the monotone guard, max(raw(0..n)); ONLY search uses this
        raw_cache, mono_cache = {}, {}
        info = {"raw_violations": 0, "corrections": 0}

        def raw_mark(i, n):
            if (i, n) not in raw_cache:
                adjusted = adjust_features_for_effort(feature_sets[i], n, self.clicks_per_session,
                                                      self.active_days_per_session, self.resources_per_session)
                raw_cache[(i, n)] = float(self.predictor(adjusted)["predicted_mark"])
            return raw_cache[(i, n)]

        def mark(i, n):
            if (i, n) not in mono_cache:
                raw = raw_mark(i, n)
                if n == 0:
                    mono_cache[(i, n)] = raw
                else:
                    prev = mono_cache[(i, n - 1)] if (i, n - 1) in mono_cache else mark(i, n - 1)
                    if raw < prev:
                        info["raw_violations"] += 1
                    mono_cache[(i, n)] = max(raw, prev) if self.enforce_monotone else raw
                    if self.enforce_monotone and raw < prev:
                        info["corrections"] += 1
            return mono_cache[(i, n)]

        def raw_state(state):
            return tuple(raw_mark(i, n) for i, n in enumerate(state))

        def adjusted_flags(state):
            return tuple(mark(i, n) > raw_mark(i, n) for i, n in enumerate(state))

        state_cache = {}

        def evaluate(state):
            if state not in state_cache:
                state_cache[state] = tuple(mark(i, n) for i, n in enumerate(state))
            return state_cache[state]

        # ---- CSP check
        solver = {"bt": solve_bt, "fc": solve_fc, "ac3": solve_ac3}[self.csp_algorithm]
        csp_attempts, last = [], {"schedule": None, "stats": None}
        totals = {"assignments_attempted": 0, "backtracks": 0, "domain_values_removed": 0, "constraint_checks": 0}

        def capacity_ok(state):
            """Pigeonhole check (deadline domains are nested, so it is exact for slot counts):
            for every deadline d, sessions due by d must not exceed the slots up to week d."""
            return all(sum(n for n, dl in zip(state, deadlines) if dl <= d) <= sum(sl.week <= d for sl in slots)
                       for d in set(deadlines))

        def run_csp(state):
            if not capacity_ok(state):   # fast, avoids exponential backtracking on hopeless plans
                res = {"success": False, "assignment": None,
                       "stats": {"assignments_attempted": 0, "backtracks": 0, "domain_values_removed": 0,
                                 "constraint_checks": 0, "execution_time_seconds": 0.0,
                                 "solution_found": False, "rejected_by": "capacity_precheck"}}
            else:
                res = solver(build_domains(state, slots, deadlines))
            csp_attempts.append({"allocation": state, "success": res["success"], "stats": res["stats"]})
            for k in totals:
                totals[k] += res["stats"].get(k, 0)
            return res

        def accept_goal(state, marks):
            res = run_csp(state)
            last["stats"] = res["stats"]
            if res["success"]:
                last["schedule"] = res["assignment"]
            return res["success"]

        result = SEARCHES[algorithm](max_sessions, targets, evaluate, accept_goal)

        out = {
            "success": result.found, "assignments": [a.get("name", f"Assignment {i+1}") for i, a in enumerate(assignments)],
            "targets": targets, "allocation": result.allocation, "predicted_marks": result.predicted_marks,
            "marks_with_no_extra_study": evaluate(tuple(0 for _ in assignments)),
            "raw_marks_with_no_extra_study": raw_state(tuple(0 for _ in assignments)),
            "effort_assumption": describe_effort_assumption(self.clicks_per_session, self.active_days_per_session,
                                                            self.resources_per_session),
            "raw_predicted_marks": None, "planning_predicted_marks": None, "monotone_adjusted": None,
            "raw_best_schedulable_marks": None,
            "search_algorithm": algorithm, "search_stats": result.stats, "search_trace": result.trace,
            "csp_algorithm": self.csp_algorithm, "csp_stats": last["stats"], "csp_total_stats": dict(totals),
            "csp_attempts": csp_attempts, "rejected_goal_count": len(result.rejected_goals),
            "rejected_goals": result.rejected_goals, "evaluated_states": len(state_cache),
            "monotone_guard": {"enabled": self.enforce_monotone, "raw_violations_seen": info["raw_violations"],
                               "corrections_applied": info["corrections"]},
            "timetable": None, "weekly_timetable": None, "failure_type": None, "failure_reason": None,
            "best_allocation": result.stats.get("best_allocation"),
            "best_predicted_marks": result.stats.get("best_predicted_marks"),
            "best_remaining_gap": result.stats.get("best_remaining_gap"),
            "best_schedulable_allocation": None, "best_schedulable_marks": None,
        }
        if result.found:
            # predicted_marks == planning_predicted_marks (kept for older callers); raw_* is the untouched NN output
            out["planning_predicted_marks"] = result.predicted_marks
            out["raw_predicted_marks"] = raw_state(result.allocation)
            out["monotone_adjusted"] = adjusted_flags(result.allocation)
            sched = last["schedule"]
            out["timetable"] = {v: {"week": s.week, "position": s.position, "slot": str(s)} for v, s in sched.items()}
            weeks = {}
            for v, s in sorted(sched.items(), key=lambda kv: kv[1].key):
                weeks.setdefault(s.week, []).append({"slot": s.position, "session": v})
            out["weekly_timetable"] = weeks
            return out

        # ---- Failure: say WHY, and show the best marks that really are possible
        if result.rejected_goals:
            out["failure_type"] = FAILURE_SCHEDULE
            out["failure_reason"] = ("The neural network says these allocations reach the target "
                                     f"({[g['allocation'] for g in result.rejected_goals]}), but the CSP cannot fit "
                                     "their sessions into the free slots before the deadlines.")
        else:
            out["failure_type"] = FAILURE_TARGET
            out["failure_reason"] = ("Target unreachable at this capacity: even using every available slot "
                                     "the predicted marks stay below the target.")
        candidates = sorted(result.explored.items(),
                            key=lambda kv: (remaining_mark_gap(kv[1], targets), sum(kv[0])))
        for state, marks in candidates[:MAX_FALLBACK_CSP_CHECKS]:
            if run_csp(state)["success"]:
                out["best_schedulable_allocation"], out["best_schedulable_marks"] = state, marks
                out["raw_best_schedulable_marks"] = raw_state(state)
                break
        out["csp_total_stats"] = dict(totals)
        return out


def plan_study(assignments, available_slots, predictor=None, search_algorithm="ucs", csp_algorithm="ac3",
               clicks_per_session=CLICKS_PER_SESSION, enforce_monotone=True):
    """Convenience wrapper around :class:`StudyPlanner`."""
    return StudyPlanner(predictor, clicks_per_session, csp_algorithm, enforce_monotone).plan(
        assignments, available_slots, search_algorithm)
