import pandas as pd
import torch

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from model import StudyPlannerNN

DATA_PATH = "data/processed/"

# Load the test dataset
test = pd.read_csv(DATA_PATH + "test_scaled.csv")

# Load training data to calculate the baseline
train = pd.read_csv(DATA_PATH + "train.csv")

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

# Neural network evaluation
nn_mae = mean_absolute_error(y_test, predictions)

nn_rmse = mean_squared_error(
    y_test, predictions
) ** 0.5

nn_r2 = r2_score(y_test, predictions)

# Baseline: always predict the training mean
baseline_prediction = train["score"].mean()

baseline_predictions = [
    baseline_prediction
] * len(y_test)

baseline_mae = mean_absolute_error(
    y_test, baseline_predictions
)

baseline_rmse = mean_squared_error(
    y_test, baseline_predictions
) ** 0.5

baseline_r2 = r2_score(
    y_test, baseline_predictions
)

# Display results
print("\n" + "=" * 60)
print("MODEL EVALUATION ON UNSEEN TEST DATA")
print("=" * 60)

print("\nNeural Network:")
print(f"MAE:  {nn_mae:.4f}")
print(f"RMSE: {nn_rmse:.4f}")
print(f"R2:   {nn_r2:.4f}")

print("\nBaseline (Training Mean):")
print(f"MAE:  {baseline_mae:.4f}")
print(f"RMSE: {baseline_rmse:.4f}")
print(f"R2:   {baseline_r2:.4f}")

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
metrics = pd.DataFrame({
    "Model": ["Neural Network", "Baseline"],
    "MAE": [nn_mae, baseline_mae],
    "RMSE": [nn_rmse, baseline_rmse],
    "R2": [nn_r2, baseline_r2]
})

metrics.to_csv(
    "results/nn/evaluation_metrics.csv",
    index=False
)

print("\nEvaluation files saved successfully!")
