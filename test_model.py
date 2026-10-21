import torch

from dataloaders import train_loader
from model import create_model


# Create the model
model = create_model()

# Get one batch from the DataLoader
images, labels = next(iter(train_loader))

# Run the batch through the model
with torch.no_grad():
    outputs = model(images)


print("Input shape :", images.shape)
print("Label shape :", labels.shape)
print("Output shape:", outputs.shape)
print("\nModel output:")
print(outputs)