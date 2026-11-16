"""Requirement tests: 5 NORMAL, 3 EDGE, 2 IMPOSSIBLE, plus checks for fixed bugs.

Run:  python -m unittest discover -s tests -t .
"""
import itertools
import unittest

import pandas as pd

from src.csp import solve_bt, solve_fc, solve_ac3
from src.csp.domains import TimeSlot, build_domains
from src.integration.effort_model import FEATURES, adjust_features_for_effort
from src.integration.planner import StudyPlanner, FAILURE_TARGET, FAILURE_SCHEDULE
from src.search.algorithms import uniform_cost_search, greedy_best_first_search, astar_search

PROFILE = dict(zip(FEATURES, [47, 12.5, 1, 78, 78, 1200, 15, 60]))


class Linear:
    """Predictor: +1 mark per session (sessions read back from the added active days)."""
    def __init__(self, per_session=1.0, base=60.0):
        self.per, self.base = per_session, base
    def __call__(self, f):
        sessions = (f["previous_active_days"] - PROFILE["previous_active_days"]) / 2
        return {"predicted_mark": min(100.0, self.base + self.per * sessions)}


def slots(weeks, per_week):
    return [TimeSlot(w, p) for w in range(1, weeks + 1) for p in range(1, per_week + 1)]


def assignment(name, target, deadline):
    return {"name": name, "target_mark": target, "deadline_week": deadline, "features": dict(PROFILE)}


class NormalCases(unittest.TestCase):
    def test_1_single_assignment_ucs(self):
        r = StudyPlanner(Linear()).plan([assignment("A", 64, 3)], slots(3, 2), "ucs")
        self.assertTrue(r["success"]); self.assertEqual(r["allocation"], (4,)); self.assertEqual(len(r["timetable"]), 4)

    def test_2_three_assignments_minimum_sessions(self):
        a = [assignment("A1", 64, 2), assignment("A2", 65, 4), assignment("A3", 63, 6)]
        r = StudyPlanner(Linear()).plan(a, slots(6, 3), "ucs")
        self.assertTrue(r["success"]); self.assertEqual(r["allocation"], (4, 5, 3))   # the task's example

    def test_3_greedy_and_astar_reach_goal(self):
        a = [assignment("A1", 62, 3), assignment("A2", 64, 4)]
        for algo in ("greedy", "astar"):
            r = StudyPlanner(Linear()).plan(a, slots(4, 3), algo)
            self.assertTrue(r["success"], algo)
            self.assertTrue(all(m >= t for m, t in zip(r["predicted_marks"], r["targets"])))

    def test_4_timetable_respects_deadlines_and_slots(self):
        a = [assignment("A1", 63, 2), assignment("A2", 63, 3)]
        r = StudyPlanner(Linear()).plan(a, slots(3, 2), "ucs")
        used = set()
        for var, it in r["timetable"].items():
            self.assertNotIn(it["slot"], used); used.add(it["slot"])
            self.assertLessEqual(it["week"], 2 if var.startswith("A1") else 3)

    def test_5_all_three_csp_solvers_agree(self):
        dom = build_domains([2, 2, 1], slots(3, 2), [2, 3, 3])
        for solver in (solve_bt, solve_fc, solve_ac3):
            self.assertTrue(solver(dom)["success"])


class EdgeCases(unittest.TestCase):
    def test_6_target_already_met_needs_zero_sessions(self):
        r = StudyPlanner(Linear()).plan([assignment("A", 50, 2)], slots(2, 2), "ucs")
        self.assertTrue(r["success"]); self.assertEqual(r["allocation"], (0,)); self.assertEqual(r["timetable"], {})

    def test_7_exactly_enough_slots(self):
        r = StudyPlanner(Linear()).plan([assignment("A", 62, 1)], slots(1, 2), "ucs")
        self.assertTrue(r["success"]); self.assertEqual(r["allocation"], (2,))

    def test_8_bad_input_is_rejected(self):
        with self.assertRaises(ValueError): StudyPlanner(Linear()).plan([], slots(1, 1))
        with self.assertRaises(ValueError): StudyPlanner(Linear()).plan([assignment("A", 60, 1)], slots(1, 1), "dfs")
        with self.assertRaises(ValueError): adjust_features_for_effort({"date": 1}, 1)


class ImpossibleCases(unittest.TestCase):
    def test_9_target_unreachable_gives_clear_message(self):
        r = StudyPlanner(Linear()).plan([assignment("A", 99, 2)], slots(2, 2), "ucs")
        self.assertFalse(r["success"]); self.assertEqual(r["failure_type"], FAILURE_TARGET)
        self.assertIn("unreachable", r["failure_reason"].lower())
        self.assertEqual(r["best_predicted_marks"], (64.0,))          # 4 slots -> 4 sessions -> 64

    def test_10_nn_says_enough_but_csp_rejects(self):
        a = [assignment("A1", 61, 1), assignment("A2", 61, 1)]       # each needs 1 session, only 1 slot exists
        r = StudyPlanner(Linear()).plan(a, [TimeSlot(1, 1)], "ucs")
        self.assertFalse(r["success"]); self.assertEqual(r["failure_type"], FAILURE_SCHEDULE)
        self.assertEqual(r["rejected_goals"][0]["allocation"], (1, 1))
        self.assertEqual(sum(r["best_schedulable_allocation"]), 1)     # best plan that CAN be scheduled
        for solver in (solve_bt, solve_fc, solve_ac3):
            self.assertFalse(solver(build_domains([1, 1], [TimeSlot(1, 1)], [1, 1]))["success"])


class BugFixChecks(unittest.TestCase):
    def test_monotone_guard_blocks_decreasing_predictions(self):
        class Falling:
            def __call__(self, f): return {"predicted_mark": 90 - (f["previous_active_days"] - 15) / 2}
        r = StudyPlanner(Falling()).plan([assignment("A", 90, 2)], slots(2, 2), "ucs")
        self.assertTrue(r["success"]); self.assertEqual(r["allocation"], (0,))
        r2 = StudyPlanner(Falling()).plan([assignment("A", 91, 2)], slots(2, 2), "ucs")
        self.assertFalse(r2["success"])                                # more effort can never help here
        self.assertGreater(r2["monotone_guard"]["corrections_applied"], 0)

    def test_ucs_matches_brute_force(self):
        evaluate = lambda s: (50 + 7 * s[0], 40 + 5 * s[1], 55 + 3 * s[2])
        targets = [64, 60, 61]
        best = min(sum(s) for s in itertools.product(range(5), repeat=3)
                   if all(m >= t for m, t in zip(evaluate(s), targets)))
        for algo in (uniform_cost_search, astar_search):
            r = algo([4, 4, 4], targets, evaluate)
            self.assertEqual(r.stats["total_sessions"], best, algo.__name__)
        self.assertTrue(greedy_best_first_search([4, 4, 4], targets, evaluate).found)

    def test_astar_never_worse_than_ucs_and_expands_no_more(self):
        evaluate = lambda s: (50 + 4 * s[0], 50 + 6 * s[1])
        u, a = uniform_cost_search([9, 9], [70, 70], evaluate), astar_search([9, 9], [70, 70], evaluate)
        self.assertEqual(u.stats["total_sessions"], a.stats["total_sessions"])
        self.assertLessEqual(a.stats["nodes_expanded"], u.stats["nodes_expanded"])

    def test_ac3_runs_before_search(self):
        r = solve_ac3(build_domains([2], slots(2, 1), [2]))
        self.assertIn("root_ac3_removed", r["stats"]); self.assertGreater(r["stats"]["root_ac3_removed"], 0)

    def test_counters_present(self):
        r = uniform_cost_search([2], [60], lambda s: (50 + 5 * s[0],))
        for key in ("nodes_expanded", "nodes_generated"): self.assertGreater(r.stats[key], 0)
        for solver in (solve_bt, solve_fc, solve_ac3):
            st = solver(build_domains([2], slots(2, 2), [2]))["stats"]
            self.assertIn("backtracks", st); self.assertIn("assignments_attempted", st)

    def test_real_nn_reacts_to_effort_without_violations(self):
        from src.nn.predictor import predict_mark
        sample = pd.read_csv("data/processed/test.csv").sample(20, random_state=0)
        for _, row in sample.iterrows():
            f = {k: float(row[k]) for k in FEATURES}
            marks = [predict_mark(adjust_features_for_effort(f, n))["predicted_mark"] for n in range(9)]
            self.assertTrue(all(b >= a - 0.01 for a, b in zip(marks, marks[1:])), marks)

    def test_numpy_predictor_matches_saved_nn_predictions(self):
        from src.nn.predictor import predict_mark
        test = pd.read_csv("data/processed/test.csv"); saved = pd.read_csv("results/nn/test_predictions.csv")
        for i in range(0, 300, 30):
            got = predict_mark({k: test.iloc[i][k] for k in FEATURES})["raw_prediction"]
            self.assertAlmostEqual(got, saved["predicted_score"].iloc[i], delta=0.02)


if __name__ == "__main__":
    unittest.main()
