import pandas as pd

DATA_PATH = "data/raw/oulad/"

# Load datasets
assessments = pd.read_csv(
    DATA_PATH + "assessments.csv"
)

student_assessment = pd.read_csv(
    DATA_PATH + "studentAssessment.csv"
)

# Convert scores to numeric
student_assessment["score"] = pd.to_numeric(
    student_assessment["score"],
    errors="coerce"
)

# Select FFF assessments
fff_assessments = assessments[
    assessments["code_module"] == "FFF"
].copy()

# Select TMA assessments only
tma_assessments = fff_assessments[
    fff_assessments["assessment_type"] == "TMA"
].copy()

# Keep only assessments with positive weight
tma_assessments = tma_assessments[
    tma_assessments["weight"] > 0
]

# Merge assessment information with student scores
target_data = student_assessment.merge(
    tma_assessments,
    on="id_assessment",
    how="inner"
)

# Remove records without actual scores
target_data = target_data.dropna(
    subset=["score"]
).copy()

# Display results
print("\n" + "=" * 60)
print("FFF TMA TARGET DATASET")
print("=" * 60)

print("\nDataset shape:")
print(target_data.shape)

print("\nAssessments per presentation:")

print(
    target_data.groupby("code_presentation")[
        "id_assessment"
    ].nunique()
)

print("\nRecords per presentation:")

print(
    target_data.groupby("code_presentation").size()
)

print("\nAssessment details:")

print(
    tma_assessments[
        ["code_presentation", "id_assessment", "date", "weight"]
    ].to_string(index=False)
)

print("\nScore statistics:")

print(target_data["score"].describe())

print("\nMissing scores:")
print(target_data["score"].isnull().sum())

# Save target records
target_data.to_csv(
    "data/processed/fff_tma_targets.csv",
    index=False
)

print("\nTarget dataset saved successfully!")
