import pandas as pd
import numpy as np

# Load test predictions
df = pd.read_csv("results/nn/test_predictions.csv")

# Calculate prediction errors
df["error"] = df["predicted_score"] - df["actual_score"]
df["absolute_error"] = df["error"].abs()

print("\n" + "=" * 60)
print("TEST PREDICTION ANALYSIS")
print("=" * 60)

print("\nFirst 15 predictions:")
print(
    df[
        ["actual_score", "predicted_score", "absolute_error"]
    ].head(15).round(2).to_string(index=False)
)

print("\nPrediction statistics:")
print(df["predicted_score"].describe().round(2))

print("\nAbsolute error statistics:")
print(df["absolute_error"].describe().round(2))

# Check predictions outside the valid score range
below_zero = (df["predicted_score"] < 0).sum()
above_100 = (df["predicted_score"] > 100).sum()

print("\nPredictions outside the 0-100 range:")
print("Below 0:", below_zero)
print("Above 100:", above_100)

# Count predictions within different error ranges
print("\nAbsolute error distribution:")
print("Error <= 5 points:", (df["absolute_error"] <= 5).sum())
print("Error <= 10 points:", (df["absolute_error"] <= 10).sum())
print("Error <= 15 points:", (df["absolute_error"] <= 15).sum())

print("\nAnalysis completed!")
