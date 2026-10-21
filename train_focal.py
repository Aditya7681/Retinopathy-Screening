import torch
import torch.nn as nn
import torch.optim as optim

from model import create_model
from dataloaders import train_loader, val_loader


# --------------------------------------------------
# Focal Loss
# --------------------------------------------------

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, outputs, labels):
        cross_entropy = nn.functional.cross_entropy(
            outputs,
            labels,
            reduction="none"
        )

        pt = torch.exp(-cross_entropy)

        focal_loss = ((1 - pt) ** self.gamma) * cross_entropy

        return focal_loss.mean()


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("FOCAL LOSS TRAINING START")
print("=" * 60)

print(f"\nDevice: {device}")


# --------------------------------------------------
# Create model
# --------------------------------------------------

model = create_model(num_classes=5)
model = model.to(device)

print("\nModel: ResNet-18")
print("Number of classes: 5")


# --------------------------------------------------
# Loss function
# --------------------------------------------------

criterion = FocalLoss(gamma=2.0)

print("Loss: Focal Loss")
print("Gamma: 2.0")


# --------------------------------------------------
# Optimizer
# --------------------------------------------------

optimizer = optim.Adam(
    model.parameters(),
    lr=0.0001
)


# --------------------------------------------------
# Training settings
# --------------------------------------------------

num_epochs = 10

best_val_accuracy = 0.0


# --------------------------------------------------
# Training loop
# --------------------------------------------------

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

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate focal loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Update model weights
        optimizer.step()

        # Track statistics
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

            # Forward pass
            outputs = model(images)

            # Calculate focal loss
            loss = criterion(outputs, labels)

            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct / total


    # --------------------------------------------------
    # Print results
    # --------------------------------------------------

    print(f"\nTraining Loss:   {train_loss:.4f}")
    print(f"Training Accuracy: {train_accuracy:.2f}%")

    print(f"\nValidation Loss:   {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.2f}%")


    # --------------------------------------------------
    # Save best model
    # --------------------------------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            "best_model_focal.pth"
        )

        print("\n✅ Best Focal Loss model saved!")


# --------------------------------------------------
# Training complete
# --------------------------------------------------

print("\n" + "=" * 60)
print("FOCAL LOSS TRAINING COMPLETE")
print("=" * 60)

print(f"\nBest Validation Accuracy: {best_val_accuracy:.2f}%")
print("\nBest model saved as: best_model_focal.pth")
