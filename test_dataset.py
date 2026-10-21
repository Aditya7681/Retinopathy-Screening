from pathlib import Path

from dataset import APTOSDataset


# Paths
train_csv = Path(r"C:\Supervised Project\data\train_split.csv")
image_dir = Path(r"E:\Supervised Project\data\train_images")


# Create dataset
dataset = APTOSDataset(
    csv_file=train_csv,
    image_dir=image_dir
)


print("=" * 60)
print("PYTORCH DATASET TEST")
print("=" * 60)

print(f"\nDataset size: {len(dataset)}")


# Get one sample
image, label = dataset[0]

print("\nFirst sample:")
print(f"Image type:  {type(image)}")
print(f"Image size:  {image.size}")
print(f"Image mode:  {image.mode}")
print(f"Label:       {label}")
print(f"Label type:  {type(label)}")


print("\n" + "=" * 60)
print("DATASET TEST COMPLETE")
print("=" * 60)