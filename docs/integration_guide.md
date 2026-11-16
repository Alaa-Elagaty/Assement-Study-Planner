# Neural Network Integration Guide

## Purpose

The neural network predicts a student's expected TMA mark using previous academic performance and online learning activity.

## Required Files

- models/study_planner_nn.pth
- models/scaler.pkl
- src/nn/model.py
- src/nn/predictor.py

## Prediction Function

Import the function:

```python
from src.nn.predictor import predict_mark
```

Call it using a dictionary containing these features:

```python
student_features = {
    "date": 47,
    "weight": 12.5,
    "previous_tma_count": 1,
    "previous_tma_mean": 78,
    "previous_tma_last_score": 78,
    "previous_clicks": 1200,
    "previous_active_days": 15,
    "previous_resources_accessed": 60
}

result = predict_mark(student_features)

predicted_mark = result["predicted_mark"]
```

## Returned Values

The function returns:

- raw_prediction: Original neural network output.
- predicted_mark: Prediction bounded between 0 and 100.

## Important Requirements

1. All eight features must be provided.
2. Feature names must match exactly.
3. Do not manually standardize the input. The predictor handles scaling.
4. Do not retrain the model during normal prediction.
5. Keep the model and scaler files together in the project structure.

## Study Planner Integration

The predicted mark can be used as an estimated academic performance value by the Study Planner.

The extra_study_sessions input should be handled separately by the planner's study-effort logic. It must not be added directly to the neural network prediction as an arbitrary mark bonus.

## Update: how the planner uses the NN now
* `predict_mark` loads lazily. PyTorch is optional: without it, the same weights are read from
  `models/study_planner_nn.npz` (create with `python src/nn/export_numpy_weights.py`). Both paths agree
  (tested against `results/nn/test_predictions.csv`).
* Study effort is added by `adjust_features_for_effort` (clicks + active days + resources). The mark is never
  edited directly.
* `StudyPlanner.plan(...)` result keys to read: `success`, `allocation`, `predicted_marks`, `weekly_timetable`,
  `failure_type` (`target_unreachable` or `schedule_infeasible`), `failure_reason`, `best_schedulable_allocation`,
  `best_schedulable_marks`, `search_stats`, `csp_total_stats`, `monotone_guard`.
* Before calling the CSP the planner runs a fast capacity check (sessions due by each deadline must not
  exceed the slots up to that week). It is exact for these nested deadlines and avoids exponential backtracking.
