from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# --------------------------------------------------
# Paths
# --------------------------------------------------

data_path = Path(r"E:\Supervised Project\data")
csv_path = data_path / "train.csv"

# --------------------------------------------------
# Load dataset
# --------------------------------------------------

df = pd.read_csv(csv_path)

print("=" * 60)
print("APTOS DATASET SPLIT")
print("=" * 60)

print(f"\nTotal images: {len(df)}")

# --------------------------------------------------
# First split
# 70% training
# 30% temporary
# --------------------------------------------------

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["diagnosis"],
    random_state=42
)

# --------------------------------------------------
# Second split
# Half of temporary → validation
# Half of temporary → test
# --------------------------------------------------

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["diagnosis"],
    random_state=42
)

# --------------------------------------------------
# Display sizes
# --------------------------------------------------

print("\nDataset sizes:")
print("-" * 60)

print(f"Training:   {len(train_df)}")
print(f"Validation: {len(val_df)}")
print(f"Test:       {len(test_df)}")

# --------------------------------------------------
# Display class distribution
# --------------------------------------------------

print("\nTraining class distribution:")
print(train_df["diagnosis"].value_counts().sort_index())

print("\nValidation class distribution:")
print(val_df["diagnosis"].value_counts().sort_index())

print("\nTest class distribution:")
print(test_df["diagnosis"].value_counts().sort_index())

# --------------------------------------------------
# Save the splits
# --------------------------------------------------

project_data_path = Path(r"C:\Supervised Project\data")

project_data_path.mkdir(parents=True, exist_ok=True)

train_df.to_csv(project_data_path / "train_split.csv", index=False)
val_df.to_csv(project_data_path / "val_split.csv", index=False)
test_df.to_csv(project_data_path / "test_split.csv", index=False)

print("\nSplit files saved:")
print("✅ train_split.csv")
print("✅ val_split.csv")
print("✅ test_split.csv")

print("\n" + "=" * 60)
print("SPLIT COMPLETE")
print("=" * 60)