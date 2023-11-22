import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import pandas as pd


class TAD66KDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.labels = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.labels.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")
        label = self.labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)

        return image, label
