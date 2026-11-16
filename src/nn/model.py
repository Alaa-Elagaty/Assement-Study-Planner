import torch
import torch.nn as nn


class StudyPlannerNN(nn.Module):

    def __init__(self, input_size=8):
        super().__init__()

        self.network = nn.Sequential(

            # First hidden layer: 8 -> 64
            nn.Linear(input_size, 64),
            nn.ReLU(),

            # Second hidden layer: 64 -> 32
            nn.Linear(64, 32),
            nn.ReLU(),

            # Output layer: 32 -> 1
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)


# Test the model architecture
if __name__ == "__main__":

    model = StudyPlannerNN()

    # Create a sample input with 8 features
    sample_input = torch.randn(1, 8)

    # Get prediction
    prediction = model(sample_input)

    print("=" * 50)
    print("STUDY PLANNER NEURAL NETWORK")
    print("=" * 50)

    print("\nModel architecture:")
    print(model)

    print("\nSample input shape:")
    print(sample_input.shape)

    print("\nPrediction shape:")
    print(prediction.shape)

    print("\nSample prediction:")
    print(prediction.item())

    print("\nTotal trainable parameters:")
    print(
        sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )
    )
