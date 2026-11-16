"""Tests for the three review fixes: raw vs planning marks, labelled effort assumption, shared baselines."""
import unittest

import pandas as pd

from src.baselines import (evaluate_all, previous_mark_predictor, make_training_mean_predictor,
                           reference_mark, BASELINE_NAMES)
from src.csp.domains import TimeSlot
from src.integration.effort_model import (FEATURES, EFFORT_SCENARIOS, effort_per_session,
                                          describe_effort_assumption)
from src.integration.planner import StudyPlanner, monotone_envelope

PROFILE = dict(zip(FEATURES, [47, 12.5, 1, 78, 78, 1200, 15, 60]))
SLOTS = [TimeSlot(w, p) for w in (1, 2, 3) for p in (1, 2)]


class Wobbly:
    """NN stand-in whose raw output DROPS at 1 session (70 -> 69) then recovers."""
    CURVE = [70, 69, 75, 74, 80, 80, 80]
    def __call__(self, f):
        return {"predicted_mark": self.CURVE[int((f["previous_active_days"] - 15) / 2)]}


def one(target):
    return [{"name": "A", "target_mark": target, "deadline_week": 3, "features": dict(PROFILE)}]


class RawVsPlanning(unittest.TestCase):
    def test_envelope(self):
        self.assertEqual(monotone_envelope([70, 69, 75, 74, 80]), [70, 70, 75, 75, 80])

    def test_raw_is_never_edited_and_planning_is_monotone(self):
        r = StudyPlanner(Wobbly()).plan(one(80), SLOTS, "ucs")
        n = r["allocation"][0]
        self.assertEqual(r["raw_predicted_marks"], (float(Wobbly.CURVE[n]),))
        self.assertEqual(r["planning_predicted_marks"], r["predicted_marks"])
        self.assertEqual(r["predicted_marks"], (80.0,))

    def test_adjustment_is_flagged_where_raw_was_overridden(self):
        r = StudyPlanner(Wobbly()).plan(one(70), SLOTS, "ucs")
        self.assertEqual(r["allocation"], (0,))
        r = StudyPlanner(Wobbly()).plan(one(75), SLOTS, "ucs")
        self.assertEqual(r["raw_predicted_marks"], (75.0,)); self.assertEqual(r["monotone_adjusted"], (False,))
        r = StudyPlanner(Wobbly()).plan(one(75.5), SLOTS, "ucs")          # first raw mark above 75.5 is 80 (4 sessions)
        self.assertEqual(r["raw_predicted_marks"], (80.0,))
        # raw dips at 1 and 3 sessions; the guard overrode them while searching and counted it
        self.assertGreater(r["monotone_guard"]["corrections_applied"], 0)

    def test_guard_off_means_planning_equals_raw(self):
        r = StudyPlanner(Wobbly(), enforce_monotone=False).plan(one(75), SLOTS, "ucs")
        self.assertEqual(r["raw_predicted_marks"], r["planning_predicted_marks"])
        self.assertEqual(r["monotone_adjusted"], (False,))

    def test_raw_marks_with_no_extra_study(self):
        r = StudyPlanner(Wobbly()).plan(one(80), SLOTS, "ucs")
        self.assertEqual(r["raw_marks_with_no_extra_study"], (70.0,))


class EffortAssumption(unittest.TestCase):
    def test_every_plan_carries_the_label(self):
        r = StudyPlanner(Wobbly()).plan(one(80), SLOTS, "ucs")
        a = r["effort_assumption"]
        self.assertEqual(a["status"], "simulated_assumption"); self.assertFalse(a["measured_from_data"])
        self.assertEqual(a["per_session"], {"clicks": 100, "active_days": 2, "resources": 4})

    def test_label_follows_custom_values(self):
        r = StudyPlanner(Wobbly(), clicks_per_session=50, active_days_per_session=1, resources_per_session=2)
        self.assertEqual(r.plan(one(70), SLOTS, "ucs")["effort_assumption"]["per_session"],
                         {"clicks": 50, "active_days": 1, "resources": 2})

    def test_scenarios(self):
        self.assertEqual(effort_per_session(EFFORT_SCENARIOS["default"]),
                         {"clicks_per_session": 100, "active_days_per_session": 2, "resources_per_session": 4})
        self.assertEqual(effort_per_session(EFFORT_SCENARIOS["low"])["clicks_per_session"], 50)
        self.assertEqual(effort_per_session(EFFORT_SCENARIOS["high"])["resources_per_session"], 8)
        self.assertTrue(describe_effort_assumption()["note"])


class SharedBaselines(unittest.TestCase):
    def test_previous_mark_baseline(self):
        self.assertEqual(previous_mark_predictor(PROFILE)["predicted_mark"], 78.0)
        none = dict(PROFILE, previous_tma_count=0, previous_tma_last_score=0)
        self.assertEqual(reference_mark(none), 55.0)

    def test_training_mean_baseline_ignores_effort(self):
        m = make_training_mean_predictor(61.5)
        self.assertEqual(m(PROFILE)["predicted_mark"], m(dict(PROFILE, previous_clicks=9999))["predicted_mark"])

    def test_same_table_for_nn_and_both_baselines(self):
        test = pd.read_csv("data/processed/test.csv"); train = pd.read_csv("data/processed/train.csv")
        saved = pd.read_csv("results/nn/test_predictions.csv")
        rows = evaluate_all(test, saved["predicted_score"].values, train["score"])
        self.assertEqual([r["Model"] for r in rows],
                         ["Neural Network", "Baseline: previous-TMA mean", "Baseline: training mean"])
        written = pd.read_csv("results/nn/evaluation_metrics.csv")
        for r, (_, w) in zip(rows, written.iterrows()):
            self.assertEqual(r["Model"], w["Model"]); self.assertAlmostEqual(r["MAE"], w["MAE"], places=6)

    def test_experiments_use_the_shared_baselines(self):
        from src.eval import experiments as E
        self.assertIs(E.reference_mark, reference_mark)
        self.assertEqual(set(BASELINE_NAMES), {"previous_mark_baseline", "training_mean_baseline"})


if __name__ == "__main__":
    unittest.main()
