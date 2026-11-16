import pandas as pd

DATA_PATH = "data/raw/oulad/"

# Load datasets
assessments = pd.read_csv(DATA_PATH + "assessments.csv")
student_assessment = pd.read_csv(
    DATA_PATH + "studentAssessment.csv"
)
student_info = pd.read_csv(DATA_PATH + "studentInfo.csv")

# Select FFF course
fff_assessments = assessments[
    assessments["code_module"] == "FFF"
].copy()

fff_students = student_info[
    student_info["code_module"] == "FFF"
].copy()

# Merge assessment results with assessment information
fff_results = student_assessment.merge(
    fff_assessments,
    on="id_assessment",
    how="inner"
)

# Convert scores to numeric
fff_results["score"] = pd.to_numeric(
    fff_results["score"],
    errors="coerce"
)

# 1. Assessment information
print("\nFFF ASSESSMENTS")
print("=" * 60)

print(
    fff_assessments[
        ["code_presentation", "id_assessment",
         "assessment_type", "date", "weight"]
    ].to_string(index=False)
)

# 2. Student counts
print("\nFFF STUDENT COUNTS")
print("=" * 60)

print(
    fff_students.groupby("code_presentation")[
        "id_student"
    ].nunique()
)

# 3. Assessment results per presentation
print("\nFFF ASSESSMENT RESULTS")
print("=" * 60)

print(
    fff_results.groupby(
        ["code_presentation", "id_assessment"]
    ).agg(
        total_records=("id_student", "count"),
        valid_scores=("score", "count"),
        average_score=("score", "mean")
    ).to_string()
)

# 4. Investigate missing score records
print("\nMISSING SCORE INVESTIGATION")
print("=" * 60)

raw_scores = student_assessment["score"]

print("Original score values that cannot be converted:")

print(
    raw_scores[
        pd.to_numeric(raw_scores, errors="coerce").isna()
    ].value_counts()
)

print("\nMissing scores in FFF:")

print(fff_results["score"].isnull().sum())
