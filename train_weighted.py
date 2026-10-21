import torch
import torch.nn as nn
import torch.optim as optim

from model import create_model
from dataloaders import train_loader, val_loader


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("WEIGHTED CROSS-ENTROPY TRAINING START")
print("=" * 60)

print(f"\nDevice: {device}")


# --------------------------------------------------
# Model
# --------------------------------------------------

model = create_model(num_classes=5)
model = model.to(device)

print("\nModel: ResNet-18")
print("Number of classes: 5")


# --------------------------------------------------
# Weighted Cross-Entropy Loss
# --------------------------------------------------

class_weights = torch.tensor(
    [1.0, 1.0, 1.0, 1.5, 1.5],
    dtype=torch.float
).to(device)

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

print("\nLoss: Weighted Cross-Entropy")
print("Class weights:")
print("Class 0 (No DR):        1.0")
print("Class 1 (Mild):         1.0")
print("Class 2 (Moderate):     1.0")
print("Class 3 (Severe):       1.5")
print("Class 4 (PDR):          1.5")


# --------------------------------------------------
# Optimizer
# --------------------------------------------------

optimizer = optim.Adam(
    model.parameters(),
    lr=0.0001
)


# --------------------------------------------------
# Training
# --------------------------------------------------

num_epochs = 10
best_val_accuracy = 0.0

for epoch in range(num_epochs):

    print("\n" + "=" * 60)
    print(f"Epoch {epoch + 1}/{num_epochs}")
    print("=" * 60)

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total


    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    model.eval()

    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    val_loss = val_loss / len(val_loader)

    val_accuracy = 100 * correct / total


    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    print(f"\nTraining Loss:     {train_loss:.4f}")
    print(f"Training Accuracy: {train_accuracy:.2f}%")

    print(f"\nValidation Loss:     {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.2f}%")


    # --------------------------------------------------
    # Save best model
    # --------------------------------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            "best_model_weighted.pth"
        )

        print("\n✅ Best Weighted CE model saved!")


# --------------------------------------------------
# Complete
# --------------------------------------------------

print("\n" + "=" * 60)
print("WEIGHTED CROSS-ENTROPY TRAINING COMPLETE")
print("=" * 60)

print(f"\nBest Validation Accuracy: {best_val_accuracy:.2f}%")

print("\nBest model saved as: best_model_weighted.pth")