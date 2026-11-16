"""Immutable search-state helpers."""
from typing import Tuple

State = Tuple[int, ...]

def initial_state(assignment_count: int) -> State:
    """Return the zero-session state for ``assignment_count`` assignments."""
    if assignment_count < 0:
        raise ValueError("assignment_count cannot be negative")
    return (0,) * assignment_count

def session_cost(state: State) -> int:
    """Total sessions allocated across all assignments."""
    return sum(state)
