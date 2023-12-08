import logging
from typing import List

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
import os
import pandas as pd

from utils.transforms import CustomTransform


class AVADataset_Split(Dataset):
    """AVA dataset with support for hlagcn and mlsp splits"""

    def __init__(
        self,
        csv_files: List[str],
        root_dir: str,
        custom_transform_options: list = None,
        default_transform: bool = True,
        split: str = "hlagcn",
    ):
        if split not in ("hlagcn", "mlsp"):
            raise ValueError(f"Unsupported split: {split}")

        self.data = self._load_and_merge_data(csv_files, split)
        self.root_dir = root_dir
        self.transform = self._build_transform(
            custom_transform_options, default_transform
        )
        self.split = split

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        max_attempts = 10  # Define a maximum number of attempts to find a valid image
        attempts = 0

        while attempts < max_attempts:
            try:
                image_id = self.data.loc[idx % len(self.data), "image_id"]
                img_name = os.path.join(self.root_dir, f"{image_id}.jpg")
                image = Image.open(img_name).convert("RGB")

                if self.transform:
                    image = self.transform(image)

                sample_data = self._get_sample_data(idx % len(self.data))

                return (image, *sample_data)
            except (OSError, UnidentifiedImageError) as e:
                logging.error(f"Error opening image: {img_name}, attempting next image.")
                idx += 1  # Move to the next index
                attempts += 1

        raise RuntimeError(f"Could not find a valid image after {max_attempts} attempts.")

    def _load_and_merge_data(self, csv_files: List[str], split: str) -> pd.DataFrame:
        data_frames = []
        for csv_file in csv_files:
            data_frames.append(pd.read_csv(csv_file))

        merged_data = pd.concat(data_frames, ignore_index=True)

        # Ensure all dataframes have the required columns for hlagcn
        if split == "hlagcn":
            for col in ["score_1", "score_2", "score_3", "score_4", "score_5",
                       "score_6", "score_7", "score_8", "score_9", "score_10"]:
                if col not in merged_data.columns:
                    merged_data[col] = 0

        return merged_data

    def _build_transform(
        self, custom_transform_options: list, default_transform: bool
    ) -> transforms.Compose:
        transform_list = [CustomTransform(custom_transform_options)]
        if default_transform:
            transform_list += [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        return transforms.Compose(transform_list)

    def _get_sample_data(self, idx):
        if self.split == "hlagcn":
            score = self.data.loc[idx, "score"]
            score_details = self.data.loc[idx, "score_1":"score_10"].to_list()
            binary_score = self.data.loc[idx, "binary_score"]
            return score, score_details, binary_score
        else:
            score = self.data.loc[idx, "score"]
            score_details = self.data.loc[idx, "score_1":"score_10"].to_list()
            return score, score_details


class TAD66KDataset_Split(Dataset):
    """TAD66K dataset with support for custom splits"""

    def __init__(self,
                 csv_file: str,
                 root_dir: str,
                 custom_transform_options: list = None,
                 default_transform: bool = True,
                 split: str = "default",
                 ):
        if split not in ("default"):
            raise ValueError(f"Unsupported split: {split}")

        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = self._build_transform(
            custom_transform_options, default_transform
        )
        self.split = split

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        max_attempts = 10  # Define a maximum number of attempts to find a valid image
        attempts = 0

        while attempts < max_attempts:
            try:
                image_id = self.data.loc[idx % len(self.data), "image_id"]
                img_name = os.path.join(self.root_dir, f"{image_id}.jpg")
                image = Image.open(img_name).convert("RGB")

                if self.transform:
                    image = self.transform(image)

                return image
            except (OSError, UnidentifiedImageError) as e:
                logging.error(f"Error opening image: {img_name}, attempting next image.")
                idx += 1  # Move to the next index
                attempts += 1

        raise RuntimeError(f"Could not find a valid image after {max_attempts} attempts.")

    def _build_transform(self, custom_transform_options: list, default_transform: bool) -> transforms.Compose:
        transform_list = [CustomTransform(custom_transform_options)]
        if default_transform:
            transform_list += [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224,
 
0.225]),
            ]
        return transforms.Compose(transform_list)