import pandas as pd

DATA_PATH = "data/raw/oulad/"

# Load required datasets
courses = pd.read_csv(DATA_PATH + "courses.csv")

assessments = pd.read_csv(DATA_PATH + "assessments.csv")

student_info = pd.read_csv(DATA_PATH + "studentInfo.csv")

student_assessment = pd.read_csv(
    DATA_PATH + "studentAssessment.csv"
)

# Convert scores from strings to numeric values
student_assessment["score"] = pd.to_numeric(
    student_assessment["score"],
    errors="coerce"
)

# 1. Available course presentations
print("\nAVAILABLE COURSE PRESENTATIONS")
print("=" * 60)

print(courses)

# 2. Number of students in each presentation
print("\nSTUDENTS PER PRESENTATION")
print("=" * 60)

print(
    student_info.groupby(
        ["code_module", "code_presentation"]
    )["id_student"].nunique()
)

# 3. Assessment information
print("\nASSESSMENTS PER PRESENTATION")
print("=" * 60)

print(
    assessments.groupby(
        ["code_module", "code_presentation"]
    ).agg(
        number_of_assessments=("id_assessment", "count"),
        total_weight=("weight", "sum")
    )
)

# 4. Assessment type distribution
print("\nASSESSMENT TYPES")
print("=" * 60)

print(assessments["assessment_type"].value_counts())

# 5. Student assessment records
print("\nSTUDENT ASSESSMENT RECORDS")
print("=" * 60)

print(
    student_assessment.groupby("id_assessment")
    .agg(
        number_of_students=("id_student", "nunique"),
        average_score=("score", "mean")
    )
    .head(20)
)

print("\nInvalid or missing scores:")
print(student_assessment["score"].isnull().sum())