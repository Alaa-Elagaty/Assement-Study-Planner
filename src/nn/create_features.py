import pandas as pd

DATA_PATH = "data/processed/"

# Load the target dataset
df = pd.read_csv(
    DATA_PATH + "fff_tma_targets.csv"
)

# Sort records chronologically for each student
df = df.sort_values(
    by=["code_presentation", "id_student", "date"]
).copy()

# Group records by student and presentation
student_groups = df.groupby(
    ["code_presentation", "id_student"]
)

# Feature 1: Number of previous TMAs with valid scores
df["previous_tma_count"] = student_groups.cumcount()

# Feature 2: Average score in previous TMAs
df["previous_tma_mean"] = (
    student_groups["score"]
    .transform(
        lambda scores: scores.shift(1).expanding().mean()
    )
)

# Feature 3: Most recent previous TMA score
df["previous_tma_last_score"] = (
    student_groups["score"].shift(1)
)

# Students with no previous TMA history
# receive 0 for the previous-score features
df["previous_tma_mean"] = (
    df["previous_tma_mean"].fillna(0)
)

df["previous_tma_last_score"] = (
    df["previous_tma_last_score"].fillna(0)
)

# Make the count an integer column
df["previous_tma_count"] = (
    df["previous_tma_count"].astype(int)
)

# Display results
print("\n" + "=" * 60)
print("HISTORICAL PERFORMANCE FEATURES")
print("=" * 60)

print("\nDataset shape:")
print(df.shape)

print("\nNew feature columns:")
print(
    df[
        [
            "previous_tma_count",
            "previous_tma_mean",
            "previous_tma_last_score"
        ]
    ].describe()
)

print("\nExample student records:")
print(
    df[
        [
            "id_student",
            "code_presentation",
            "date",
            "score",
            "previous_tma_count",
            "previous_tma_mean",
            "previous_tma_last_score"
        ]
    ].head(10).to_string(index=False)
)

# Save the feature dataset
df.to_csv(
    DATA_PATH + "fff_tma_features.csv",
    index=False
)

print("\nFeature dataset saved successfully!")