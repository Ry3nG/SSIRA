import logging
import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import pandas as pd
from utils.transforms import CustomTransform

# Constants
IMG_SIZE = 224  # Image size for resizing
MEAN = [0.485, 0.456, 0.406]  # Normalization mean
STD = [0.229, 0.224, 0.225]  # Normalization standard deviation


class OpenCVTransform:
    def __init__(self, img_size=IMG_SIZE, mean=MEAN, std=STD):
        self.img_size = img_size
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)

    def __call__(self, image_cv):
        # Check if the input is a tensor and convert it back to numpy array if necessary
        if isinstance(image_cv, torch.Tensor):
            image_cv = image_cv.permute(1, 2, 0).numpy()

        # Resize
        image_cv = cv.resize(image_cv, (self.img_size, self.img_size))

        # Normalize
        image_cv = image_cv.astype(np.float32) / 255.0
        image_cv = (image_cv - self.mean) / self.std

        # Convert to tensor
        image_tensor = torch.from_numpy(image_cv).permute(2, 0, 1)

        return image_tensor


class TAD66KDataset(Dataset):
    def __init__(
        self, csv_file, root_dir, custom_transform_options, default_transform=True
    ):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.custom_transform = CustomTransform(custom_transform_options)
        self.default_transform = OpenCVTransform() if default_transform else None

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        for _ in range(10):  # max_attempts
            img_path = os.path.join(self.root_dir, self.data_frame.iloc[idx, 0])
            image_cv = cv.imread(img_path)
            if image_cv is None:
                logging.error(f"Invalid or missing image at path: {img_path}")
                idx = (idx + 1) % len(self.data_frame)
                continue

            image_cv = cv.cvtColor(image_cv, cv.COLOR_BGR2RGB)

            try:
                degraded_image_cv = self.custom_transform(
                    image_cv
                )  # Apply custom transform first
            except ValueError as e:
                logging.error(f"Custom transform failed: {e}")
                continue

            if self.default_transform:
                degraded_image_cv = self.default_transform(
                    degraded_image_cv
                )  # Apply default transform second
                image_cv = self.default_transform(image_cv)

            return degraded_image_cv, image_cv
        logging.error("Could not find a valid image after 10 attempts.")
        return None, None


class AVADataset(Dataset):
    def __init__(
        self,
        txt_file,
        root_dir,
        custom_transform_options,
        default_transform=True,
        ids=None,
        include_ids=True,
    ):
        self.data_frame = pd.read_csv(txt_file, delim_whitespace=True, header=None)
        self.root_dir = root_dir
        self.custom_transform = CustomTransform(custom_transform_options)
        self.default_transform = OpenCVTransform() if default_transform else None
        self.ids = ids
        self.include_ids = include_ids

        if self.ids is not None:
            self.ids = [int(id) for id in self.ids]
            if self.include_ids:
                self.data_frame = self.data_frame[self.data_frame[1].isin(self.ids)]
            else:
                self.data_frame = self.data_frame[~self.data_frame[1].isin(self.ids)]

            id_set_from_list = set(self.ids)
            id_set_from_ava = set(self.data_frame[1])
            missing_ids = id_set_from_list.difference(id_set_from_ava)
            logging.info(f"Missing IDs: {missing_ids}")
            logging.info(f"Number of missing IDs: {len(missing_ids)}")
            logging.info(f"Filtered dataset size: {len(self.data_frame)}")

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        for _ in range(10):  # max_attempts
            image_id = self.data_frame.iloc[idx, 1]
            img_path = os.path.join(self.root_dir, f"{image_id}.jpg")
            image_cv = cv.imread(img_path)
            if image_cv is None:
                logging.error(
                    f"Error opening image: {img_path}, attempting next image."
                )
                idx = (idx + 1) % len(self.data_frame)
                continue

            try:
                image_cv = cv.cvtColor(image_cv, cv.COLOR_BGR2RGB)
                image_cv = self.custom_transform(
                    image_cv
                )  # Apply custom transform first
            except ValueError as e:
                logging.error(f"Custom transform failed: {e}")
                continue

            if self.default_transform:
                image_cv = self.default_transform(
                    image_cv
                )  # Apply default transform first

            ratings = self.data_frame.iloc[idx, 2:12].values
            score = sum((i + 1) * ratings[i] for i in range(10)) / sum(ratings)
            return image_cv, score
        logging.error("Could not find a valid image after 10 attempts.")
        return None, None


# Usage example for the datasets
# tad66k_dataset = TAD66KDataset(csv_file='path_to_tad66k_csv', root_dir='path_to_tad66k_images', custom_transform_options=[...])
# ava_dataset = AVADataset(txt_file='path_to_ava_txt', root_dir='path_to_ava_images', custom_transform_options=[...], ids=[...])


"""
# Additional dataset for GLINTnet Self-Supervised Learning
class GLINTnetDataset(Dataset):
    
    #Dataset for GLINTnet self-supervised learning phase. 
    #This dataset applies random transformations (degradations) to the images.
    
    def __init__(self, csv_file, root_dir, custom_transform_options, default_transform=True):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.custom_transform = CustomTransform(custom_transform_options)
        self.default_transform = default_transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx % len(self.data_frame), 0])
        try:
            image = Image.open(img_name).convert("RGB")
            # Apply custom transformation (degradation)
            degraded_image = self.custom_transform(image)
            # Apply default transformations if enabled
            if self.default_transform:
                image = default_transform(image)
                degraded_image = default_transform(degraded_image)
            return degraded_image, image  # Return both degraded and original images
        except (OSError, UnidentifiedImageError):
            logging.error(f"Error opening image: {img_name}.")
            return None, None

"""
