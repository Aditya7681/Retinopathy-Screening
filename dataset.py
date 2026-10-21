from pathlib import Path

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset


class APTOSDataset(Dataset):

    def __init__(self, csv_file, image_dir, transform=None):

        # Load CSV
        self.data = pd.read_csv(csv_file)

        # Image directory
        self.image_dir = Path(image_dir)

        # Transformations
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        # Get image ID and label
        image_id = self.data.iloc[index]["id_code"]
        label = self.data.iloc[index]["diagnosis"]

        # Construct image path
        image_path = self.image_dir / f"{image_id}.png"

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        # Convert label to PyTorch tensor
        label = torch.tensor(label, dtype=torch.long)

        return image, label