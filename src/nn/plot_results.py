from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = PROJECT_ROOT / "results" / "nn"
GRAPHS_DIR = RESULTS_DIR / "graphs"

GRAPHS_DIR.mkdir(parents=True, exist_ok=True)


# Load results
history = pd.read_csv(RESULTS_DIR / "training_history.csv")
metrics = pd.read_csv(RESULTS_DIR / "evaluation_metrics.csv")
predictions = pd.read_csv(RESULTS_DIR / "test_predictions.csv")


# --------------------------------------------------
# GRAPH 1: Training vs Validation Loss
# --------------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    history["epoch"],
    history["train_loss"],
    label="Training Loss"
)

plt.plot(
    history["epoch"],
    history["val_loss"],
    label="Validation Loss"
)

best_epoch = history.loc[history["val_loss"].idxmin(), "epoch"]

plt.axvline(
    x=best_epoch,
    linestyle="--",
    label=f"Best validation epoch: {best_epoch}"
)

plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss (log scale)")
plt.title("Neural Network Training and Validation Loss")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(GRAPHS_DIR / "training_validation_loss.png", dpi=300)
plt.close()


# --------------------------------------------------
# GRAPH 2: Actual vs Predicted Marks
# --------------------------------------------------

plt.figure(figsize=(8, 7))

plt.scatter(
    predictions["actual_score"],
    predictions["predicted_score"],
    alpha=0.3
)

plt.plot(
    [0, 100],
    [0, 100],
    linestyle="--",
    label="Perfect Prediction"
)

plt.xlabel("Actual Mark")
plt.ylabel("Predicted Mark")
plt.title("Actual vs Predicted TMA Marks")
plt.xlim(0, 100)
plt.ylim(0, 110)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(GRAPHS_DIR / "actual_vs_predicted.png", dpi=300)
plt.close()


# --------------------------------------------------
# GRAPH 3: Prediction Error Distribution
# --------------------------------------------------

predictions["error"] = (
    predictions["predicted_score"] -
    predictions["actual_score"]
)

plt.figure(figsize=(9, 6))

plt.hist(
    predictions["error"],
    bins=40,
    edgecolor="black"
)

plt.axvline(
    x=0,
    linestyle="--",
    label="Zero Error"
)

plt.xlabel("Prediction Error (Predicted - Actual)")
plt.ylabel("Number of Predictions")
plt.title("Distribution of Neural Network Prediction Errors")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(GRAPHS_DIR / "prediction_error_distribution.png", dpi=300)
plt.close()


# --------------------------------------------------
# GRAPH 4: Neural Network vs Baseline
# --------------------------------------------------

comparison = metrics.set_index("Model")

metric_names = ["MAE", "RMSE"]

# Neural network first, then every baseline row written by evaluate_model.py (src/baselines.py)
model_names = ["Neural Network"] + [m for m in comparison.index if m != "Neural Network"]

positions = range(len(metric_names))
width = 0.8 / len(model_names)

plt.figure(figsize=(9, 6))

for k, name in enumerate(model_names):
    plt.bar(
        [p + (k - (len(model_names) - 1) / 2) * width for p in positions],
        comparison.loc[name, metric_names],
        width,
        label=name
    )

plt.xticks(list(positions), metric_names)
plt.ylabel("Error (Mark Points)")
plt.title("Neural Network vs Baselines Performance")
plt.legend()
plt.grid(axis="y", alpha=0.3)

# Display R-squared values for additional context
plt.figtext(
    0.5,
    0.01,
    "R²: " + " | ".join(f"{n} = {comparison.loc[n, 'R2']:.3f}" for n in model_names),
    ha="center",
    fontsize=8
)

plt.tight_layout(rect=[0, 0.05, 1, 1])

plt.savefig(GRAPHS_DIR / "model_vs_baseline.png", dpi=300)
plt.close()


# --------------------------------------------------
# FINISH
# --------------------------------------------------

print("=" * 50)
print("ALL GRAPHS GENERATED SUCCESSFULLY")
print("=" * 50)

for graph in GRAPHS_DIR.glob("*.png"):
    print(graph.name)

print("\nSaved in:", GRAPHS_DIR)
