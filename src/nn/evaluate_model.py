import sys
from pathlib import Path

import pandas as pd
import torch

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from model import StudyPlannerNN

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.baselines import evaluate_all   # shared baselines + metrics (same as the planner experiments)

DATA_PATH = "data/processed/"

# Load the test dataset
test = pd.read_csv(DATA_PATH + "test_scaled.csv")

# Load training data (training-mean baseline) and the UNSCALED test rows (previous-mark baseline)
train = pd.read_csv(DATA_PATH + "train.csv")
test_raw = pd.read_csv(DATA_PATH + "test.csv")

feature_columns = [
    "date",
    "weight",
    "previous_tma_count",
    "previous_tma_mean",
    "previous_tma_last_score",
    "previous_clicks",
    "previous_active_days",
    "previous_resources_accessed"
]

# Prepare test tensors
X_test = torch.tensor(
    test[feature_columns].values,
    dtype=torch.float32
)

y_test = test["score"].values

# Load the trained model
model = StudyPlannerNN()

model.load_state_dict(
    torch.load(
        "models/study_planner_nn.pth",
        map_location="cpu"
    )
)

model.eval()

# Generate predictions
with torch.no_grad():
    predictions = model(X_test).numpy().flatten()

# Neural network + BOTH baselines, same rows, same metric code (src/baselines.py)
assert (test_raw["score"].values == y_test).all(), "test.csv and test_scaled.csv rows are not aligned"
rows = evaluate_all(test_raw, predictions, train["score"])

# Display results
print("\n" + "=" * 60)
print("MODEL EVALUATION ON UNSEEN TEST DATA")
print("=" * 60)
for r in rows:
    print(f"\n{r['Model']}:")
    print(f"MAE:  {r['MAE']:.4f}")
    print(f"RMSE: {r['RMSE']:.4f}")
    print(f"R2:   {r['R2']:.4f}")

# Save individual predictions
results = pd.DataFrame({
    "actual_score": y_test,
    "predicted_score": predictions
})

results.to_csv(
    "results/nn/test_predictions.csv",
    index=False
)

# Save metrics
metrics = pd.DataFrame(rows)

metrics.to_csv(
    "results/nn/evaluation_metrics.csv",
    index=False
)

print("\nEvaluation files saved successfully!")
