
import pandas as pd
import torch

DATA_PATH = "data/processed/"

# Load scaled datasets
train = pd.read_csv(DATA_PATH + "train_scaled.csv")
validation = pd.read_csv(DATA_PATH + "validation_scaled.csv")
test = pd.read_csv(DATA_PATH + "test_scaled.csv")

# Separate input features and target
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

X_train = train[feature_columns].values
y_train = train["score"].values

X_validation = validation[feature_columns].values
y_validation = validation["score"].values

X_test = test[feature_columns].values
y_test = test["score"].values

# Convert NumPy arrays into PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)

X_validation = torch.tensor(
    X_validation, dtype=torch.float32
)
y_validation = torch.tensor(
    y_validation, dtype=torch.float32
).reshape(-1, 1)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

# Display tensor information
print("\n" + "=" * 60)
print("PYTORCH TENSOR PREPARATION")
print("=" * 60)

print("\nTraining:")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

print("\nValidation:")
print("X_validation shape:", X_validation.shape)
print("y_validation shape:", y_validation.shape)

print("\nTesting:")
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

print("\nTensor data types:")
print("X_train:", X_train.dtype)
print("y_train:", y_train.dtype)

print("\nExample input:")
print(X_train[0])

print("\nExample target:")
print(y_train[0])

print("\nTensor preparation completed successfully!")
