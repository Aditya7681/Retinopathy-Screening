from pathlib import Path
import pandas as pd

# Paths
data_path = Path(r"E:\Supervised Project\data")
csv_path = data_path / "train.csv"
images_path = data_path / "train_images"

print("=" * 60)
print("APTOS DATASET INSPECTION")
print("=" * 60)

# --------------------------------------------------
# 1. Load CSV
# --------------------------------------------------

df = pd.read_csv(csv_path)

print("\n[1] CSV INFORMATION")
print("-" * 60)

print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")

print("\nColumn names:")
print(list(df.columns))


# --------------------------------------------------
# 2. Show first few rows
# --------------------------------------------------

print("\n[2] FIRST 5 ROWS")
print("-" * 60)

print(df.head())


# --------------------------------------------------
# 3. Check diagnosis labels
# --------------------------------------------------

print("\n[3] DIAGNOSIS DISTRIBUTION")
print("-" * 60)

print(df["diagnosis"].value_counts().sort_index())


# --------------------------------------------------
# 4. Check for missing values
# --------------------------------------------------

print("\n[4] MISSING VALUES")
print("-" * 60)

print(df.isnull().sum())


# --------------------------------------------------
# 5. Check image/CSV matching
# --------------------------------------------------

print("\n[5] IMAGE MATCHING")
print("-" * 60)

# Get all image filenames
image_files = {
    file.name
    for file in images_path.iterdir()
    if file.suffix.lower() in {".png", ".jpg", ".jpeg"}
}

# Add .png to the IDs from CSV
csv_images = {
    f"{image_id}.png"
    for image_id in df["id_code"]
}

# Images mentioned in CSV but missing from folder
missing_images = csv_images - image_files

# Images present in folder but not mentioned in CSV
extra_images = image_files - csv_images

print(f"Images in folder:       {len(image_files)}")
print(f"Images in CSV:          {len(csv_images)}")
print(f"Missing image files:    {len(missing_images)}")
print(f"Unlisted image files:   {len(extra_images)}")


# --------------------------------------------------
# 6. Final result
# --------------------------------------------------

print("\n" + "=" * 60)

if len(missing_images) == 0 and len(extra_images) == 0:
    print("✅ CSV AND IMAGE DATA MATCH PERFECTLY")
else:
    print("⚠️ CSV AND IMAGE DATA DO NOT MATCH")

print("=" * 60)