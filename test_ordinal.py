import torch
import torch.nn as nn
import torchvision.models as models

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from dataloaders import test_loader


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("TESTING ORDINAL MODEL")
print("=" * 60)

print(f"\nDevice: {device}")


# --------------------------------------------------
# Create ordinal model
# --------------------------------------------------

model = models.resnet18(weights=None)

# Ordinal model has 4 output units
model.fc = nn.Linear(
    model.fc.in_features,
    4
)


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

model.load_state_dict(
    torch.load(
        "best_model_ordinal.pth",
        map_location=device
    )
)

model = model.to(device)
model.eval()

print("\nModel: ResNet-18")
print("Output units: 4")
print("Task: Ordinal classification")
print("Loaded: best_model_ordinal.pth")


# --------------------------------------------------
# Class names
# --------------------------------------------------

class_names = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR"
]


# --------------------------------------------------
# Testing
# --------------------------------------------------

all_labels = []
all_predictions = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        # Forward pass
        outputs = model(images)

        # Convert logits to probabilities
        probabilities = torch.sigmoid(outputs)

        # Each threshold represents:
        #
        # threshold 0 -> severity > 0
        # threshold 1 -> severity > 1
        # threshold 2 -> severity > 2
        # threshold 3 -> severity > 3
        #
        # Count how many thresholds are passed
        # to obtain the final class 0-4.

        predictions = (probabilities >= 0.5).sum(dim=1)

        # Store results
        all_labels.extend(labels.numpy())
        all_predictions.extend(
            predictions.cpu().numpy()
        )


# --------------------------------------------------
# Overall metrics
# --------------------------------------------------

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)


# --------------------------------------------------
# Print overall results
# --------------------------------------------------

print("\n" + "=" * 60)
print("TEST RESULTS")
print("=" * 60)

print(f"\nTest Accuracy:  {accuracy * 100:.2f}%")
print(f"Precision:      {precision:.4f}")
print(f"Recall:         {recall:.4f}")
print(f"F1-Score:       {f1:.4f}")


# --------------------------------------------------
# Classification report
# --------------------------------------------------

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=class_names,
        zero_division=0
    )
)


# --------------------------------------------------
# Confusion matrix
# --------------------------------------------------

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print("\nRows = Actual")
print("Columns = Predicted\n")

print(cm)


# --------------------------------------------------
# Testing complete
# --------------------------------------------------

print("\n" + "=" * 60)
print("ORDINAL TESTING COMPLETE")
print("=" * 60)