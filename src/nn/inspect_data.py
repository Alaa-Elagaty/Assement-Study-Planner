import pandas as pd
import os

# Dataset folder
DATA_PATH = "data/raw/oulad"

# Load all seven CSV files
files = {
    "assessments": "assessments.csv",
    "courses": "courses.csv",
    "studentAssessment": "studentAssessment.csv",
    "studentInfo": "studentInfo.csv",
    "studentRegistration": "studentRegistration.csv",
    "studentVle": "studentVle.csv",
    "vle": "vle.csv"
}

datasets = {}

for name, filename in files.items():

    path = os.path.join(DATA_PATH, filename)

    df = pd.read_csv(path)

    datasets[name] = df

    print("\n" + "=" * 60)
    print("FILE:", filename)
    print("=" * 60)

    print("Rows and columns:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nFirst 3 rows:")
    print(df.head(3))