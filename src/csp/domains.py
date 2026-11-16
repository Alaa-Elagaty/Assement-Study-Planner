"""Time-slot representation and deadline-filtered domains."""
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence
from .variables import build_variables

@dataclass(frozen=True, order=True)
class TimeSlot:
    week: int
    position: int
    label: str = ""
    def __post_init__(self):
        if self.week < 1 or self.position < 1:
            raise ValueError("week and position must be positive integers")
    @property
    def key(self):
        return (self.week, self.position)
    def __str__(self):
        return self.label or f"W{self.week}_S{self.position}"

def normalize_slots(available_slots: Iterable):
    slots=[]
    for item in available_slots:
        if isinstance(item, TimeSlot): slots.append(item)
        elif isinstance(item, Mapping): slots.append(TimeSlot(int(item["week"]), int(item["position"]), str(item.get("label", ""))))
        elif isinstance(item, (tuple, list)) and len(item) >= 2: slots.append(TimeSlot(int(item[0]), int(item[1]), str(item[2]) if len(item)>2 else ""))
        else: raise TypeError("Slots must be TimeSlot, mapping, or (week, position[, label])")
    if len({s.key for s in slots}) != len(slots): raise ValueError("available time slots must be unique")
    return sorted(slots, key=lambda s:s.key)

def build_domains(allocation: Sequence[int], available_slots: Iterable, deadlines: Sequence[int]):
    """Build variable domains; deadline is the last permitted week (inclusive)."""
    if len(allocation) != len(deadlines): raise ValueError("one deadline is required per assignment")
    slots=normalize_slots(available_slots)
    variables=build_variables(allocation)
    domains={}
    for var in variables:
        assignment_index=int(var.split("_")[0][1:])-1
        deadline=int(deadlines[assignment_index])
        if deadline < 1: raise ValueError("deadlines must be positive week numbers")
        domains[var]=[slot for slot in slots if slot.week <= deadline]
    return domains
