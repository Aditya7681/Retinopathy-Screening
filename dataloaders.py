from pathlib import Path

import pandas as pd
import torch

from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset import APTOSDataset
from transforms import train_transform, val_test_transform


# --------------------------------------------------
# Paths
# --------------------------------------------------

project_data_path = Path(r"C:\Supervised Project\data")
image_dir = Path(r"E:\Supervised Project\data\train_images")

train_csv = project_data_path / "train_split.csv"
val_csv = project_data_path / "val_split.csv"
test_csv = project_data_path / "test_split.csv"


# --------------------------------------------------
# Create datasets
# --------------------------------------------------

train_dataset = APTOSDataset(
    csv_file=train_csv,
    image_dir=image_dir,
    transform=train_transform
)

val_dataset = APTOSDataset(
    csv_file=val_csv,
    image_dir=image_dir,
    transform=val_test_transform
)

test_dataset = APTOSDataset(
    csv_file=test_csv,
    image_dir=image_dir,
    transform=val_test_transform
)


# --------------------------------------------------
# Calculate class weights for training sampler
# --------------------------------------------------

train_labels = pd.read_csv(train_csv)["diagnosis"]

class_counts = train_labels.value_counts().sort_index()

print("=" * 60)
print("CLASS DISTRIBUTION")
print("=" * 60)

print("\nTraining class counts:")
print(class_counts)


# Inverse frequency weighting
class_weights = 1.0 / torch.tensor(
    class_counts.values,
    dtype=torch.float
)

print("\nClass weights:")
for class_id, weight in enumerate(class_weights):
    print(f"Class {class_id}: {weight:.6f}")


# --------------------------------------------------
# Create weight for every training sample
# --------------------------------------------------

sample_weights = torch.tensor(
    [
        class_weights[label]
        for label in train_labels
    ],
    dtype=torch.float
)


# --------------------------------------------------
# Create weighted sampler
# --------------------------------------------------

train_sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)


# --------------------------------------------------
# Create DataLoaders
# --------------------------------------------------

batch_size = 16

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    sampler=train_sampler,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)


# --------------------------------------------------
# Print information
# --------------------------------------------------

print("\n" + "=" * 60)
print("DATALOADERS CREATED")
print("=" * 60)

print(f"\nTraining samples:   {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples:       {len(test_dataset)}")

print(f"\nBatch size: {batch_size}")

print(f"\nTraining batches:   {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")
print(f"Test batches:       {len(test_loader)}")