from pathlib import Path
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

# Paths
data_path = Path(r"E:\Supervised Project\data")
csv_path = data_path / "train.csv"
images_path = data_path / "train_images"

# Load CSV
df = pd.read_csv(csv_path)

print("=" * 60)
print("APTOS IMAGE INSPECTION")
print("=" * 60)

# --------------------------------------------------
# 1. Inspect image dimensions
# --------------------------------------------------

print("\n[1] IMAGE DIMENSIONS")
print("-" * 60)

sample_ids = df["id_code"].head(10)

for image_id in sample_ids:
    image_path = images_path / f"{image_id}.png"

    with Image.open(image_path) as img:
        print(
            f"{image_id}.png -> "
            f"Size: {img.size}, "
            f"Mode: {img.mode}"
        )


# --------------------------------------------------
# 2. Display one image from each class
# --------------------------------------------------

print("\n[2] DISPLAYING ONE IMAGE FROM EACH CLASS")
print("-" * 60)

fig, axes = plt.subplots(1, 5, figsize=(20, 5))

for diagnosis in range(5):

    # Get first image belonging to this class
    row = df[df["diagnosis"] == diagnosis].iloc[0]

    image_id = row["id_code"]
    image_path = images_path / f"{image_id}.png"

    image = Image.open(image_path)

    axes[diagnosis].imshow(image)
    axes[diagnosis].set_title(
        f"Diagnosis {diagnosis}\n{image_id}"
    )
    axes[diagnosis].axis("off")

plt.tight_layout()
plt.show()