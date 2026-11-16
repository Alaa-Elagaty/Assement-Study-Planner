"""Shared baselines + metrics, so Member 1 (NN evaluation) and Member 2 (planner experiments)
compare against the SAME reference predictors with the SAME metric code.

Two baselines, both effort-blind (they ignore study sessions):
  * previous_mark_baseline : the student's mean previous TMA mark (last score / 55 if none).
                             Needs the features, so it can also run inside the planner.
  * training_mean_baseline : one constant, the mean score of the training set.

Every baseline comparison in the project (src/nn/evaluate_model.py, src/eval/experiments.py)
imports from here. Do not re-implement them elsewhere.
"""
from pathlib import Path
from typing import Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREVIOUS_MARK = "previous_mark_baseline"
TRAINING_MEAN = "training_mean_baseline"
BASELINE_NAMES = (PREVIOUS_MARK, TRAINING_MEAN)
DEFAULT_MARK_IF_NO_HISTORY = 55.0


def reference_mark(f: Mapping) -> float:
    """Mark the student 'usually gets': mean of previous TMAs (last score / 55 if none)."""
    if int(f["previous_tma_count"]) > 0:
        return float(f["previous_tma_mean"])
    return float(f["previous_tma_last_score"]) or DEFAULT_MARK_IF_NO_HISTORY


def previous_mark_predictor(features: Mapping) -> dict:
    """Planner-compatible baseline: ignores study effort, so more sessions never help."""
    return {"predicted_mark": max(0.0, min(100.0, reference_mark(features)))}


def training_mean_value(train_scores) -> float:
    return float(np.mean(train_scores))


def make_training_mean_predictor(train_mean: float):
    """Planner-compatible constant baseline built from the training-set mean score."""
    value = max(0.0, min(100.0, float(train_mean)))
    return lambda features: {"predicted_mark": value}


def previous_mark_array(frame) -> np.ndarray:
    """previous_mark_baseline prediction for every row of an (unscaled) feature DataFrame."""
    return np.array([max(0.0, min(100.0, reference_mark(r))) for _, r in frame.iterrows()])


def regression_metrics(y_true, y_pred) -> dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    return {"MAE": float(mean_absolute_error(y_true, y_pred)),
            "RMSE": float(mean_squared_error(y_true, y_pred) ** 0.5),
            "R2": float(r2_score(y_true, y_pred))}


def evaluate_all(test_frame, nn_predictions, train_scores):
    """One table: NN + both baselines on the SAME held-out rows, SAME metrics.

    ``test_frame`` is the UNSCALED test.csv (needs the 8 features + score), in the same row order
    as ``nn_predictions``. Returns a list of {"Model", "MAE", "RMSE", "R2"} rows.
    """
    y = test_frame["score"].values
    if len(y) != len(nn_predictions):
        raise ValueError("test rows and NN predictions must have the same length")
    const = np.full(len(y), training_mean_value(train_scores))
    return [{"Model": "Neural Network", **regression_metrics(y, nn_predictions)},
            {"Model": "Baseline: previous-TMA mean", **regression_metrics(y, previous_mark_array(test_frame))},
            {"Model": "Baseline: training mean", **regression_metrics(y, const)}]
