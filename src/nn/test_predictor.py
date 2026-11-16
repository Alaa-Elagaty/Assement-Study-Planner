from predictor import predict_mark


# Three example student profiles
students = {
    "Student A - Strong previous performance": {
        "date": 47,
        "weight": 12.5,
        "previous_tma_count": 1,
        "previous_tma_mean": 90,
        "previous_tma_last_score": 90,
        "previous_clicks": 2000,
        "previous_active_days": 20,
        "previous_resources_accessed": 100
    },

    "Student B - Average previous performance": {
        "date": 47,
        "weight": 12.5,
        "previous_tma_count": 1,
        "previous_tma_mean": 65,
        "previous_tma_last_score": 65,
        "previous_clicks": 1000,
        "previous_active_days": 12,
        "previous_resources_accessed": 50
    },

    "Student C - Limited previous activity": {
        "date": 47,
        "weight": 12.5,
        "previous_tma_count": 1,
        "previous_tma_mean": 40,
        "previous_tma_last_score": 40,
        "previous_clicks": 200,
        "previous_active_days": 4,
        "previous_resources_accessed": 15
    }
}


print("=" * 60)
print("PREDICTOR TEST - DIFFERENT STUDENT PROFILES")
print("=" * 60)

for student_name, features in students.items():

    result = predict_mark(features)

    print(f"\n{student_name}")
    print("-" * 40)
    print("Predicted mark:", result["predicted_mark"])
    print("Raw prediction:", result["raw_prediction"])

print("\nAll profile tests completed!")
