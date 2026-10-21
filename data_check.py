from pathlib import Path

# Dataset paths
data_path = Path(r"E:\Supervised Project\data")
csv_path = data_path / "train.csv"
images_path = data_path / "train_images"
zip_path = data_path / "train_images.zip"

print("=" * 60)
print("APTOS DATASET CHECK")
print("=" * 60)

# 1. Check main data folder
print("\n[1] Checking data folder...")

if data_path.exists():
    print("✅ Data folder found")
else:
    print("❌ Data folder NOT found")
    exit()


# 2. Check CSV
print("\n[2] Checking CSV file...")

if csv_path.exists():
    print(f"✅ train.csv found")
    print(f"   Size: {csv_path.stat().st_size / (1024 * 1024):.2f} MB")
else:
    print("❌ train.csv NOT found")


# 3. Check extracted image folder
print("\n[3] Checking image folder...")

if images_path.exists():
    print("✅ train_images folder found")
else:
    print("❌ train_images folder NOT found")


# 4. Check ZIP
print("\n[4] Checking ZIP archive...")

if zip_path.exists():
    print(f"✅ train_images.zip found")
    print(f"   Size: {zip_path.stat().st_size / (1024 * 1024):.2f} MB")
else:
    print("❌ train_images.zip NOT found")


# 5. Count images
if images_path.exists():

    print("\n[5] Counting images...")

    image_extensions = {".png", ".jpg", ".jpeg"}

    images = [
        file for file in images_path.iterdir()
        if file.suffix.lower() in image_extensions
    ]

    print(f"✅ Images found: {len(images)}")

    # Show first 5 images
    print("\nFirst 5 images:")
    for image in sorted(images)[:5]:
        print(f"   {image.name}")


print("\n" + "=" * 60)
print("DATASET CHECK COMPLETE")
print("=" * 60)