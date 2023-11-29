import logging
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image,UnidentifiedImageError
import os
import pandas as pd

from utils.transforms import CustomTransform

default_transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class TAD66KDataset(Dataset):
    """TAD66K Dataset for self-supervised learning"""

    def __init__(self, csv_file, root_dir, custom_transform_options,default_transform = True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        transform_list = [CustomTransform(custom_transform_options)]
        if default_transform:
            transform_list += [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        self.transform = transforms.Compose(transform_list)


    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        max_attempts = 10  # for instance
        for _ in range(max_attempts):
            try:
                img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx % len(self.data_frame), 0])
                image = Image.open(img_name).convert("RGB")
                if self.transform:
                    image = self.transform(image)
                return image
            except (OSError, UnidentifiedImageError) as e:
                logging.error(f"Error opening image: {img_name}, attempting next image.")
                idx += 1
        logging.error(f"Could not find a valid image after {max_attempts} attempts.")




class AVADataset(Dataset):
    """AVA dataset for Aesthetic Score Prediction"""

    def __init__(self, txt_file, root_dir, custom_transform_options, default_transform = True, ids = None, include_ids = True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_frame = pd.read_csv(txt_file, delim_whitespace=True, header=None)
        self.root_dir = root_dir
        transform_list = [CustomTransform(custom_transform_options)]
        if default_transform:
            transform_list += [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        self.transform = transforms.Compose(transform_list)
        self.ids = ids
        self.include_ids = include_ids

        if self.ids is not None:
            # Ensure IDs are of the same type as in the data_frame
            self.ids = [int(id) for id in self.ids]  # Convert to int if they are strings
            if self.include_ids:
                # Keep only rows with IDs in the list
                self.data_frame = self.data_frame[self.data_frame[1].isin(self.ids)]
            else:
                # Exclude rows with IDs in the list
                self.data_frame = self.data_frame[~self.data_frame[1].isin(self.ids)]

        logging.info(f"Filtered dataset size: {len(self.data_frame)}")


    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        max_attempts = 10  # Maximum number of attempts to find a valid image
        attempts = 0

        while attempts < max_attempts:
            try:
                image_id = self.data_frame.iloc[idx, 1]
                img_name = os.path.join(self.root_dir, f"{image_id}.jpg")
                image = Image.open(img_name).convert("RGB")

                if self.transform:
                    image = self.transform(image)

                # compute the aesthetic score as the average of the ratings
                ratings = self.data_frame.iloc[idx, 2:12].values
                score = sum((i + 1) * ratings[i] for i in range(10)) / sum(ratings)

                return image, score

            except (OSError, UnidentifiedImageError) as e:
                logging.error(f"Error opening image: {img_name}, attempting next image.")
                idx = (idx + 1) % len(self.data_frame)  # Move to the next index
                attempts += 1
        
        logging.error(f"Could not find a valid image after {max_attempts} attempts.")
 
# Example instantiation of the dataset
# tad66k_dataset = TAD66KDataset(csv_file='path_to_tad66k_csv', root_dir='path_to_tad66k_images', transform=transform)
# ava_dataset = AVADataset(txt_file='path_to_ava_txt', root_dir='path_to_ava_images', transform=transform)
