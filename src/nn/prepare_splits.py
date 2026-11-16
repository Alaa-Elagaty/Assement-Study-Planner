import pandas as pd

INPUT_PATH = "data/processed/fff_tma_features.csv"
OUTPUT_PATH = "data/processed/"

# Load the complete feature dataset
df = pd.read_csv(INPUT_PATH)

# Select the eight input features
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

# Keep presentation as metadata for checking
# Keep score as the prediction target
selected_columns = (
    ["code_presentation"]
    + feature_columns
    + ["score"]
)

df = df[selected_columns].copy()

# Chronological split
train = df[
    df["code_presentation"].isin(["2013B", "2013J"])
].copy()

validation = df[
    df["code_presentation"] == "2014B"
].copy()

test = df[
    df["code_presentation"] == "2014J"
].copy()

# Display split sizes
print("\n" + "=" * 60)
print("TRAIN / VALIDATION / TEST SPLIT")
print("=" * 60)

print("\nTraining dataset:")
print("Shape:", train.shape)
print(train["code_presentation"].value_counts())

print("\nValidation dataset:")
print("Shape:", validation.shape)
print(validation["code_presentation"].value_counts())

print("\nTesting dataset:")
print("Shape:", test.shape)
print(test["code_presentation"].value_counts())

# Check feature and target columns
print("\nInput features:")
print(feature_columns)

print("\nTarget:")
print("score")

# Save datasets
train.to_csv(
    OUTPUT_PATH + "train.csv",
    index=False
)

validation.to_csv(
    OUTPUT_PATH + "validation.csv",
    index=False
)

test.to_csv(
    OUTPUT_PATH + "test.csv",
    index=False
)

print("\nAll three datasets saved successfully!")
