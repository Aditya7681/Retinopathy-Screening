import torch

from dataloaders import train_loader


print("=" * 60)
print("DATALOADER TEST")
print("=" * 60)


# Get one batch
images, labels = next(iter(train_loader))


print("\nBatch information:")
print(f"Images shape: {images.shape}")
print(f"Labels shape: {labels.shape}")

print(f"\nImage dtype: {images.dtype}")
print(f"Label dtype: {labels.dtype}")

print("\nLabels in first batch:")
print(labels.tolist())


# Check expected shape
expected_image_shape = torch.Size([16, 3, 224, 224])
expected_label_shape = torch.Size([16])

if images.shape == expected_image_shape:
    print("\n✅ Image batch shape is correct")
else:
    print(f"\n⚠️ Image batch shape: {images.shape}")


if labels.shape == expected_label_shape:
    print("✅ Label batch shape is correct")
else:
    print(f"⚠️ Label batch shape: {labels.shape}")


print("\n" + "=" * 60)
print("DATALOADER TEST COMPLETE")
print("=" * 60)