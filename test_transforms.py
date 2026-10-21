from pathlib import Path

import torch

from dataset import APTOSDataset
from transforms import train_transform


# Paths
train_csv = Path(r"C:\Supervised Project\data\train_split.csv")
image_dir = Path(r"E:\Supervised Project\data\train_images")


# Create dataset with transformations
dataset = APTOSDataset(
    csv_file=train_csv,
    image_dir=image_dir,
    transform=train_transform
)


print("=" * 60)
print("TRANSFORMATION TEST")
print("=" * 60)

print(f"\nDataset size: {len(dataset)}")


# Get one transformed sample
image, label = dataset[0]

print("\nTransformed sample:")
print(f"Image type:   {type(image)}")
print(f"Image shape:  {image.shape}")
print(f"Image dtype:  {image.dtype}")
print(f"Label:        {label}")
print(f"Label type:   {type(label)}")


# Check numerical range
print("\nTensor statistics:")
print(f"Minimum value: {image.min().item():.3f}")
print(f"Maximum value: {image.max().item():.3f}")
print(f"Mean value:    {image.mean().item():.3f}")
print(f"Std value:     {image.std().item():.3f}")


# Verify tensor shape
expected_shape = torch.Size([3, 224, 224])

if image.shape == expected_shape:
    print("\n✅ Image successfully transformed to 3 × 224 × 224")
else:
    print("\n❌ Unexpected image shape!")


print("\n" + "=" * 60)
print("TRANSFORMATION TEST COMPLETE")
print("=" * 60)