import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from model import create_model
from dataloaders import test_loader


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("TESTING START")
print("=" * 60)

print(f"\nDevice: {device}")


# --------------------------------------------------
# Load model
# --------------------------------------------------

model = create_model(num_classes=5)

model.load_state_dict(
    torch.load(
        "best_model_focal.pth",
        map_location=device
    )
)

model = model.to(device)
model.eval()

print("\nModel: ResNet-18")
print("Loaded: best_model_focal.pth")


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
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)

        # Get predicted class
        _, predictions = torch.max(outputs, 1)

        # Store results
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())


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
print("TESTING COMPLETE")
print("=" * 60)