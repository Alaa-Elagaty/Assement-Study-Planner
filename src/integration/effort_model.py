"""Configurable simulation translating study sessions into learning activity.

GENERATED RULE (documented in docs/data_dictionary.md): one study session adds
  * 100 VLE clicks, 2 active days and 4 resources accessed.
Clicks, active days and resources are moved TOGETHER because they are
correlated in the real data; raising clicks alone gave unrealistic inputs and
the NN lowered its prediction in 54% of steps (see docs/limitations.md).

This is a planning assumption, NOT a measured causal effect. Every planner result carries it in
``effort_assumption`` and docs/limitations.md says how to read it; ``EFFORT_SCENARIOS`` lets you test
how much the plan changes when the assumption is wrong (experiment 4, effort sensitivity).
"""
from typing import Mapping

FEATURES = ("date", "weight", "previous_tma_count", "previous_tma_mean",
            "previous_tma_last_score", "previous_clicks",
            "previous_active_days", "previous_resources_accessed")
CLICKS_PER_SESSION = 100
ACTIVE_DAYS_PER_SESSION = 2
RESOURCES_PER_SESSION = 4


def adjust_features_for_effort(features: Mapping, additional_sessions: int,
                               clicks_per_session: int = CLICKS_PER_SESSION,
                               active_days_per_session: int = ACTIVE_DAYS_PER_SESSION,
                               resources_per_session: int = RESOURCES_PER_SESSION) -> dict:
    """Return a COPY of the 8 NN features with simulated study effort added."""
    missing = [n for n in FEATURES if n not in features]
    extra = [n for n in features if n not in FEATURES]
    if missing or extra:
        raise ValueError(f"Expected exactly the eight NN features; missing={missing}, extra={extra}")
    if int(additional_sessions) < 0 or min(int(clicks_per_session), int(active_days_per_session),
                                           int(resources_per_session)) < 0:
        raise ValueError("session and click counts must be non-negative")
    n = int(additional_sessions)
    adjusted = dict(features)
    adjusted["previous_clicks"] = features["previous_clicks"] + n * int(clicks_per_session)
    adjusted["previous_active_days"] = features["previous_active_days"] + n * int(active_days_per_session)
    adjusted["previous_resources_accessed"] = features["previous_resources_accessed"] + n * int(resources_per_session)
    return adjusted


# ---- the assumption is declared, labelled and testable (not hidden inside the numbers) ----
# Scenario name -> multiplier on the default per-session activity. Only "default" is the project's
# baseline guess; "low"/"high" exist to show how much the plan depends on that guess.
EFFORT_SCENARIOS = {"low": 0.5, "default": 1.0, "high": 2.0}


def effort_per_session(scale: float = 1.0) -> dict:
    """Per-session activity for a scenario multiplier (rounded to whole clicks/days/resources)."""
    return {"clicks_per_session": round(CLICKS_PER_SESSION * scale),
            "active_days_per_session": round(ACTIVE_DAYS_PER_SESSION * scale),
            "resources_per_session": round(RESOURCES_PER_SESSION * scale)}


def describe_effort_assumption(clicks_per_session: int = CLICKS_PER_SESSION,
                               active_days_per_session: int = ACTIVE_DAYS_PER_SESSION,
                               resources_per_session: int = RESOURCES_PER_SESSION) -> dict:
    """Machine-readable label attached to every plan: these numbers are assumed, not measured."""
    return {"status": "simulated_assumption", "measured_from_data": False,
            "per_session": {"clicks": int(clicks_per_session), "active_days": int(active_days_per_session),
                            "resources": int(resources_per_session)},
            "note": ("A study session is a modelled bundle of extra VLE activity. The data has no study "
                     "sessions, so how much activity a session adds, and what that does to marks, is "
                     "assumed. Read predicted marks as 'if a session adds this much activity'.")}
