"""Uniform Cost and Greedy Best-First Search.

``accept_goal`` lets the planner reject a mark-reaching state when CSP cannot
schedule it, while keeping search alive for later candidate allocations.
"""
from dataclasses import dataclass, field
from heapq import heappush, heappop
from itertools import count
from time import perf_counter
from typing import Callable, Dict, Optional, Sequence, Tuple
from .state import State, initial_state, session_cost
from .successor import generate_successors
from .goal_test import is_goal
from .heuristics import remaining_mark_gap, unmet_assignment_count

Predictor = Callable[[State], Sequence[float]]
GoalAcceptor = Callable[[State, Sequence[float]], bool]

@dataclass
class SearchResult:
    found: bool
    allocation: Optional[State]
    predicted_marks: Optional[Tuple[float, ...]]
    stats: dict = field(default_factory=dict)
    rejected_goals: list = field(default_factory=list)
    trace: list = field(default_factory=list)      # one row per expanded node
    explored: dict = field(default_factory=dict)   # state -> predicted marks (all evaluated)


def _search(max_sessions: Sequence[int], target_marks: Sequence[float],
            evaluate: Predictor, algorithm: str, accept_goal: Optional[GoalAcceptor] = None) -> SearchResult:
    if len(max_sessions) != len(target_marks):
        raise ValueError("max_sessions and target_marks must have equal lengths")
    if any(int(x) < 0 for x in max_sessions):
        raise ValueError("max_sessions must be non-negative")
    started = perf_counter()
    serial = count()
    start = initial_state(len(max_sessions))
    start_marks = tuple(float(x) for x in evaluate(start))
    if len(start_marks) != len(target_marks):
        raise ValueError("evaluate must return one mark per assignment")
    h0 = remaining_mark_gap(start_marks, target_marks)
    best_state, best_marks, best_gap = start, start_marks, h0
    heap = []
    def prio(st, mk, gap):
        if algorithm == "ucs": return session_cost(st)
        if algorithm == "astar": return session_cost(st) + unmet_assignment_count(mk, target_marks)
        return gap
    priority = prio(start, start_marks, h0)
    heappush(heap, (priority, next(serial), start, start_marks))
    best_priority: Dict[State, float] = {start: priority}
    explored = set()
    generated = 1
    expanded = 0
    rejected = []
    trace = []
    seen_marks = {start: start_marks}
    while heap:
        _, _, state, marks = heappop(heap)
        if state in explored:
            continue
        explored.add(state)
        expanded += 1   # nodes_expanded = nodes popped from the frontier and examined
        gap_now = remaining_mark_gap(marks, target_marks)
        goal_now = is_goal(marks, target_marks)
        trace.append({"order": len(explored), "state": state, "sessions": session_cost(state),
                      "marks": tuple(marks), "gap": gap_now, "is_goal": goal_now, "csp_ok": None})
        if goal_now:
            ok = accept_goal is None or accept_goal(state, marks)
            trace[-1]["csp_ok"] = bool(ok)
            if ok:
                elapsed = perf_counter() - started
                return SearchResult(True, state, tuple(marks), {
                    "nodes_generated": generated, "nodes_expanded": expanded,
                    "explored_states": len(explored), "final_allocation": state,
                    "total_sessions": session_cost(state), "final_search_cost": session_cost(state),
                    "execution_time_seconds": elapsed, "goal_found": True,
                    "best_allocation": state, "best_predicted_marks": tuple(marks),
                    "best_remaining_gap": 0.0,
                    "algorithm": algorithm.upper()
                }, rejected, trace, seen_marks)
            rejected.append({"allocation": state, "predicted_marks": tuple(marks)})
        for child in generate_successors(state, max_sessions):
            if child in explored:
                continue
            generated += 1
            child_marks = tuple(float(x) for x in evaluate(child))
            seen_marks[child] = child_marks
            if len(child_marks) != len(target_marks):
                raise ValueError("evaluate must return one mark per assignment")
            child_gap = remaining_mark_gap(child_marks, target_marks)
            if child_gap < best_gap:
                best_state, best_marks, best_gap = child, child_marks, child_gap
            child_priority = prio(child, child_marks, child_gap)
            if child not in best_priority or child_priority < best_priority[child]:
                best_priority[child] = child_priority
                heappush(heap, (child_priority, next(serial), child, child_marks))
    return SearchResult(False, None, None, {
        "nodes_generated": generated, "nodes_expanded": expanded,
        "explored_states": len(explored), "final_allocation": None,
        "total_sessions": None, "final_search_cost": None,
        "execution_time_seconds": perf_counter() - started, "goal_found": False,
        "best_allocation": best_state, "best_predicted_marks": tuple(best_marks),
        "best_remaining_gap": best_gap,
        "algorithm": algorithm.upper()
    }, rejected, trace, seen_marks)

def uniform_cost_search(max_sessions: Sequence[int], target_marks: Sequence[float],
                        evaluate: Predictor, accept_goal: Optional[GoalAcceptor] = None) -> SearchResult:
    """Explore allocations by increasing total session count (optimal for unit costs)."""
    return _search(max_sessions, target_marks, evaluate, "ucs", accept_goal)

def greedy_best_first_search(max_sessions: Sequence[int], target_marks: Sequence[float],
                             evaluate: Predictor, accept_goal: Optional[GoalAcceptor] = None) -> SearchResult:
    """Explore lowest predicted mark-gap states first; not guaranteed optimal."""
    return _search(max_sessions, target_marks, evaluate, "greedy", accept_goal)

def astar_search(max_sessions: Sequence[int], target_marks: Sequence[float],
                 evaluate: Predictor, accept_goal: Optional[GoalAcceptor] = None) -> SearchResult:
    """A*: f = sessions so far + number of assignments still below target (admissible)."""
    return _search(max_sessions, target_marks, evaluate, "astar", accept_goal)
