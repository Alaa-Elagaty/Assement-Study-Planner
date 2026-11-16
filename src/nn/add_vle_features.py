import pandas as pd

DATA_PATH = "data/raw/oulad/"
OUTPUT_PATH = "data/processed/"

# Load the dataset containing historical performance features
features = pd.read_csv(
    OUTPUT_PATH + "fff_tma_features.csv"
)

# Load online learning activity
student_vle = pd.read_csv(
    DATA_PATH + "studentVle.csv"
)

# Keep FFF course records only
student_vle = student_vle[
    student_vle["code_module"] == "FFF"
].copy()

# Convert columns to numeric
student_vle["date"] = pd.to_numeric(
    student_vle["date"], errors="coerce"
)

student_vle["sum_click"] = pd.to_numeric(
    student_vle["sum_click"], errors="coerce"
)

# Remove invalid activity records
student_vle = student_vle.dropna(
    subset=["id_student", "date", "sum_click", "id_site"]
).copy()

# Prepare the three new feature columns
features["previous_clicks"] = 0
features["previous_active_days"] = 0
features["previous_resources_accessed"] = 0

# Group VLE activity by student and presentation
vle_groups = {
    key: group
    for key, group in student_vle.groupby(
        ["id_student", "code_presentation"]
    )
}

# Calculate activity before each TMA deadline
for (student_id, presentation), target_group in features.groupby(
    ["id_student", "code_presentation"]
):

    activity = vle_groups.get(
        (student_id, presentation)
    )

    # No recorded activity for this student
    if activity is None:
        continue

    # Process each TMA record for this student
    for index, target in target_group.iterrows():

        deadline = target["date"]

        # Keep activity strictly before the deadline
        previous_activity = activity[
            activity["date"] < deadline
        ]

        features.loc[index, "previous_clicks"] = (
            previous_activity["sum_click"].sum()
        )

        features.loc[index, "previous_active_days"] = (
            previous_activity["date"].nunique()
        )

        features.loc[index, "previous_resources_accessed"] = (
            previous_activity["id_site"].nunique()
        )

# Display results
print("\n" + "=" * 60)
print("ONLINE LEARNING ACTIVITY FEATURES")
print("=" * 60)

print("\nDataset shape:")
print(features.shape)

print("\nNew feature statistics:")
print(
    features[
        [
            "previous_clicks",
            "previous_active_days",
            "previous_resources_accessed"
        ]
    ].describe()
)

print("\nExample records:")
print(
    features[
        [
            "id_student",
            "code_presentation",
            "date",
            "score",
            "previous_clicks",
            "previous_active_days",
            "previous_resources_accessed"
        ]
    ].head(10).to_string(index=False)
)

# Save updated dataset
features.to_csv(
    OUTPUT_PATH + "fff_tma_features.csv",
    index=False
)

print("\nVLE features added and dataset saved successfully!")
