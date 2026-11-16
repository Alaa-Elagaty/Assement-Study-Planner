import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from model import StudyPlannerNN

# Reproducibility
torch.manual_seed(42)

# Device
device = torch.device("cpu")

# Load tensors created in the previous step
# We recreate them from the scaled CSV files.
import pandas as pd

DATA_PATH = "data/processed/"

train_df = pd.read_csv(DATA_PATH + "train_scaled.csv")
val_df = pd.read_csv(DATA_PATH + "validation_scaled.csv")

feature_columns = [
    "date",
    "weight",
    "previous_tma_count",
    "previous_tma_mean",
    "previous_tma_last_score",
    "previous_clicks",
    "previous_active_days",
    "previous_resources_accessed"
]

X_train = torch.tensor(
    train_df[feature_columns].values,
    dtype=torch.float32
)

y_train = torch.tensor(
    train_df["score"].values,
    dtype=torch.float32
).reshape(-1, 1)

X_val = torch.tensor(
    val_df[feature_columns].values,
    dtype=torch.float32
)

y_val = torch.tensor(
    val_df["score"].values,
    dtype=torch.float32
).reshape(-1, 1)

# Create DataLoaders
train_dataset = TensorDataset(X_train, y_train)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

# Initialize model
model = StudyPlannerNN().to(device)

# Loss function and optimizer
criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# Early stopping settings
epochs = 100
patience = 10

best_val_loss = float("inf")
epochs_without_improvement = 0

# Store training history
history = []

print("=" * 60)
print("NEURAL NETWORK TRAINING")
print("=" * 60)

for epoch in range(epochs):

    # Training phase
    model.train()

    total_train_loss = 0

    for inputs, targets in train_loader:

        inputs = inputs.to(device)
        targets = targets.to(device)

        # Forward pass
        predictions = model(inputs)

        # Calculate loss
        loss = criterion(predictions, targets)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item() * inputs.size(0)

    avg_train_loss = total_train_loss / len(train_dataset)

    # Validation phase
    model.eval()

    with torch.no_grad():

        val_predictions = model(X_val.to(device))

        val_loss = criterion(
            val_predictions,
            y_val.to(device)
        ).item()

    history.append({
        "epoch": epoch + 1,
        "train_loss": avg_train_loss,
        "val_loss": val_loss
    })

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Train Loss: {avg_train_loss:.4f} "
        f"Validation Loss: {val_loss:.4f}"
    )

    # Early stopping
    if val_loss < best_val_loss:

        best_val_loss = val_loss
        epochs_without_improvement = 0

        torch.save(
            model.state_dict(),
            "models/study_planner_nn.pth"
        )

    else:

        epochs_without_improvement += 1

    if epochs_without_improvement >= patience:

        print("\nEarly stopping triggered!")
        break

# Save training history
pd.DataFrame(history).to_csv(
    "results/nn/training_history.csv",
    index=False
)

print("\nTraining completed!")
print("Best validation loss:", best_val_loss)
print("Best model saved to models/study_planner_nn.pth")
print("Training history saved successfully!")
