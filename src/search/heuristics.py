"""Heuristics for best-first study allocation search."""
from typing import Sequence

def remaining_mark_gap(predicted_marks: Sequence[float], target_marks: Sequence[float]) -> float:
    """Sum of positive target gaps; lower values indicate a more promising state.

    This is a guidance heuristic, not a lower bound on required sessions.
    """
    if len(predicted_marks) != len(target_marks):
        raise ValueError("predicted_marks and target_marks must have equal lengths")
    return sum(max(0.0, float(t) - float(p)) for p, t in zip(predicted_marks, target_marks))


def unmet_assignment_count(predicted_marks: Sequence[float], target_marks: Sequence[float]) -> int:
    """Admissible A* heuristic: each assignment still below target needs >= 1 more session.

    It never overestimates the remaining cost (every session costs 1), and it is
    consistent (one added session lowers it by at most 1), so A* with a closed
    set returns a minimum-session plan.
    """
    if len(predicted_marks) != len(target_marks):
        raise ValueError("predicted_marks and target_marks must have equal lengths")
    return sum(1 for p, t in zip(predicted_marks, target_marks) if float(p) < float(t))
