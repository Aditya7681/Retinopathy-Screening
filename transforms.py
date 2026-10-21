from torchvision import transforms


# --------------------------------------------------
# Training transformations
# --------------------------------------------------

train_transform = transforms.Compose([

    # Resize all images to the same size
    transforms.Resize((224, 224)),

    # Random horizontal flip
    transforms.RandomHorizontalFlip(p=0.5),

    # Small random rotation
    transforms.RandomRotation(degrees=10),

    # Slight brightness and contrast variation
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2
    ),

    # Convert PIL image → PyTorch tensor
    transforms.ToTensor(),

    # Normalize RGB channels
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# --------------------------------------------------
# Validation / Test transformations
# --------------------------------------------------

val_test_transform = transforms.Compose([

    # Resize
    transforms.Resize((224, 224)),

    # Convert to tensor
    transforms.ToTensor(),

    # Normalize
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])