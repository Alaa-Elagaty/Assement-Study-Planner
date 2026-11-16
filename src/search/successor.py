"""Successor generation with finite per-assignment bounds."""
from typing import Iterator, Sequence
from .state import State

def generate_successors(state: State, max_sessions: Sequence[int]) -> Iterator[State]:
    """Yield states formed by adding exactly one legal session."""
    if len(state) != len(max_sessions):
        raise ValueError("state and max_sessions must have equal lengths")
    for index, maximum in enumerate(max_sessions):
        if maximum < 0 or state[index] < 0:
            raise ValueError("session counts and limits must be non-negative")
        if state[index] < maximum:
            child = list(state)
            child[index] += 1
            yield tuple(child)
