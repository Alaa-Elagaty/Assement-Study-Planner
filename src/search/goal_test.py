"""Predicted-mark goal test."""
from typing import Sequence

def is_goal(predicted_marks: Sequence[float], target_marks: Sequence[float]) -> bool:
    """True only when every assignment meets or exceeds its target."""
    if len(predicted_marks) != len(target_marks):
        raise ValueError("predicted_marks and target_marks must have equal lengths")
    return all(float(mark) >= float(target) for mark, target in zip(predicted_marks, target_marks))
