import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def create_model(num_classes=5):
    # Load ResNet-18 with pretrained ImageNet weights
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    # Get the number of inputs to the final layer
    num_features = model.fc.in_features

    # Replace the original 1000-class layer with our 5-class layer
    model.fc = nn.Linear(num_features, num_classes)

    return model