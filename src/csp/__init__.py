"""Constraint-satisfaction scheduling algorithms."""
from .domains import TimeSlot, build_domains
from .backtracking import solve_bt
from .forward_checking import solve_fc
from .ac3 import solve_ac3
__all__ = ["TimeSlot", "build_domains", "solve_bt", "solve_fc", "solve_ac3"]
