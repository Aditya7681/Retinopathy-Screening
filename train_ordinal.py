import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

from dataloaders import train_loader, val_loader


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("ORDINAL CLASSIFICATION TRAINING START")
print("=" * 60)

print(f"\nDevice: {device}")


# --------------------------------------------------
# Create ordinal ResNet-18
# --------------------------------------------------

model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    4
)

model = model.to(device)

print("\nModel: ResNet-18")
print("Output units: 4")
print("Task: Ordinal classification")


# --------------------------------------------------
# Convert class labels to ordinal targets
# --------------------------------------------------

def create_ordinal_targets(labels):

    ordinal_targets = torch.stack(
        [
            (labels > 0).float(),
            (labels > 1).float(),
            (labels > 2).float(),
            (labels > 3).float()
        ],
        dim=1
    )

    return ordinal_targets


# --------------------------------------------------
# Loss
# --------------------------------------------------

criterion = nn.BCEWithLogitsLoss()

print("\nLoss: BCEWithLogitsLoss")
print("Ordinal thresholds: 4")


# --------------------------------------------------
# Optimizer
# --------------------------------------------------

optimizer = optim.Adam(
    model.parameters(),
    lr=0.0001
)


# --------------------------------------------------
# Convert ordinal outputs to class predictions
# --------------------------------------------------

def ordinal_predictions(outputs):

    probabilities = torch.sigmoid(outputs)

    predictions = (
        probabilities >= 0.5
    ).sum(dim=1)

    return predictions


# --------------------------------------------------
# Training
# --------------------------------------------------

num_epochs = 10
best_val_accuracy = 0.0

for epoch in range(num_epochs):

    print("\n" + "=" * 60)
    print(f"Epoch {epoch + 1}/{num_epochs}")
    print("=" * 60)


    # ==================================================
    # TRAINING
    # ==================================================

    model.train()

    running_loss = 0.0

    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        ordinal_targets = create_ordinal_targets(
            labels
        )

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            ordinal_targets
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        predictions = ordinal_predictions(
            outputs
        )

        total += labels.size(0)

        correct += (
            predictions == labels
        ).sum().item()


    train_loss = (
        running_loss / len(train_loader)
    )

    train_accuracy = (
        100 * correct / total
    )


    # ==================================================
    # VALIDATION
    # ==================================================

    model.eval()

    val_loss = 0.0

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            ordinal_targets = create_ordinal_targets(
                labels
            )

            outputs = model(images)

            loss = criterion(
                outputs,
                ordinal_targets
            )

            val_loss += loss.item()

            predictions = ordinal_predictions(
                outputs
            )

            total += labels.size(0)

            correct += (
                predictions == labels
            ).sum().item()


    val_loss = (
        val_loss / len(val_loader)
    )

    val_accuracy = (
        100 * correct / total
    )


    # ==================================================
    # RESULTS
    # ==================================================

    print(f"\nTraining Loss:     {train_loss:.4f}")
    print(f"Training Accuracy: {train_accuracy:.2f}%")

    print(f"\nValidation Loss:     {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.2f}%")


    # ==================================================
    # SAVE BEST MODEL
    # ==================================================

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            "best_model_ordinal.pth"
        )

        print("\n✅ Best Ordinal model saved!")


# --------------------------------------------------
# Complete
# --------------------------------------------------

print("\n" + "=" * 60)
print("ORDINAL CLASSIFICATION TRAINING COMPLETE")
print("=" * 60)

print(
    f"\nBest Validation Accuracy: "
    f"{best_val_accuracy:.2f}%"
)

print(
    "\nBest model saved as: "
    "best_model_ordinal.pth"
)