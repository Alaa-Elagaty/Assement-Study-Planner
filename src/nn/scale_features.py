import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler

DATA_PATH = "data/processed/"
FEATURE_COLUMNS = [
    "date",
    "weight",
    "previous_tma_count",
    "previous_tma_mean",
    "previous_tma_last_score",
    "previous_clicks",
    "previous_active_days",
    "previous_resources_accessed"
]

# Load the three datasets
train = pd.read_csv(DATA_PATH + "train.csv")
validation = pd.read_csv(DATA_PATH + "validation.csv")
test = pd.read_csv(DATA_PATH + "test.csv")

# Create the scaler
scaler = StandardScaler()

# Fit ONLY on training features
X_train = scaler.fit_transform(
    train[FEATURE_COLUMNS]
)

# Transform validation and test using training statistics
X_validation = scaler.transform(
    validation[FEATURE_COLUMNS]
)

X_test = scaler.transform(
    test[FEATURE_COLUMNS]
)

# Convert scaled arrays back to DataFrames
X_train = pd.DataFrame(
    X_train,
    columns=FEATURE_COLUMNS
)

X_validation = pd.DataFrame(
    X_validation,
    columns=FEATURE_COLUMNS
)

X_test = pd.DataFrame(
    X_test,
    columns=FEATURE_COLUMNS
)

# Save scaled features with their targets
train_scaled = X_train.copy()
train_scaled["score"] = train["score"].values

validation_scaled = X_validation.copy()
validation_scaled["score"] = validation["score"].values

test_scaled = X_test.copy()
test_scaled["score"] = test["score"].values

train_scaled.to_csv(
    DATA_PATH + "train_scaled.csv",
    index=False
)

validation_scaled.to_csv(
    DATA_PATH + "validation_scaled.csv",
    index=False
)

test_scaled.to_csv(
    DATA_PATH + "test_scaled.csv",
    index=False
)

# Save the fitted scaler for future predictions
joblib.dump(
    scaler,
    "models/scaler.pkl"
)

# Display results
print("\n" + "=" * 60)
print("FEATURE SCALING RESULTS")
print("=" * 60)

print("\nTraining scaled feature statistics:")
print(X_train.describe().round(3))

print("\nDataset shapes:")
print("Training:", train_scaled.shape)
print("Validation:", validation_scaled.shape)
print("Testing:", test_scaled.shape)

print("\nScaler saved to models/scaler.pkl")
print("\nScaled datasets saved successfully!")
