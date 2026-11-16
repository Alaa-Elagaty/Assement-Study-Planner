"""Configurable simulation translating study sessions into learning activity.

GENERATED RULE (documented in docs/data_dictionary.md): one study session adds
  * 100 VLE clicks, 2 active days and 4 resources accessed.
Clicks, active days and resources are moved TOGETHER because they are
correlated in the real data; raising clicks alone gave unrealistic inputs and
the NN lowered its prediction in 54% of steps (see docs/limitations.md).

This is a planning assumption, NOT a measured causal effect.
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
