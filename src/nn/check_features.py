import pandas as pd

FILE_PATH = "data/processed/fff_tma_features.csv"

# Load dataset
df = pd.read_csv(FILE_PATH)

print("\n" + "=" * 60)
print("FINAL FEATURE DATASET INSPECTION")
print("=" * 60)

# Dataset dimensions
print("\nDataset shape:")
print(df.shape)

# Column names and data types
print("\nColumn names and data types:")
print(df.dtypes)

# Missing values
print("\nMissing values:")
print(df.isnull().sum())

# Duplicate rows
print("\nDuplicate rows:")
print(df.duplicated().sum())

# Check duplicate student-assessment records
print("\nDuplicate student-assessment combinations:")
print(
    df.duplicated(
        subset=["id_student", "id_assessment"]
    ).sum()
)

# Numeric feature statistics
numeric_columns = [
    "score",
    "previous_tma_count",
    "previous_tma_mean",
    "previous_tma_last_score",
    "previous_clicks",
    "previous_active_days",
    "previous_resources_accessed"
]

print("\nNumeric feature statistics:")
print(df[numeric_columns].describe())

# Check negative values
print("\nNegative values:")
print((df[numeric_columns] < 0).sum())

print("\nInspection completed!")
