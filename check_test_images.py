from pathlib import Path
import pandas as pd
from PIL import Image


# --------------------------------------------------
# Paths
# --------------------------------------------------

test_csv = Path(r"C:\Supervised Project\data\test_split.csv")
image_dir = Path(r"E:\Supervised Project\data\train_images")


# --------------------------------------------------
# Load test CSV
# --------------------------------------------------

df = pd.read_csv(test_csv)

print("=" * 60)
print("CHECKING TEST IMAGES")
print("=" * 60)

bad_images = []


# --------------------------------------------------
# Check every test image
# --------------------------------------------------

for index, row in df.iterrows():

    image_id = row["id_code"]
    image_path = image_dir / f"{image_id}.png"

    try:

        with Image.open(image_path) as image:
            image.verify()

    except Exception as e:

        bad_images.append(
            {
                "id_code": image_id,
                "path": str(image_path),
                "error": str(e)
            }
        )

        print("\n❌ Problem found:")
        print(f"Image ID: {image_id}")
        print(f"Path:     {image_path}")
        print(f"Error:    {e}")


# --------------------------------------------------
# Results
# --------------------------------------------------

print("\n" + "=" * 60)
print("CHECK COMPLETE")
print("=" * 60)

print(f"\nTotal test images: {len(df)}")
print(f"Bad images:        {len(bad_images)}")
print(f"Valid images:      {len(df) - len(bad_images)}")


if len(bad_images) == 0:
    print("\n✅ All test images are valid.")

else:
    print("\n⚠️ Some test images need to be replaced/recovered.")