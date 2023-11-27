import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
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
        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image


class AVADataset(Dataset):
    """AVA dataset for Aesthetic Score Prediction"""

    def __init__(self, txt_file, root_dir, custom_transform_options, default_transform = True):
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

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        image_id = self.data_frame.iloc[idx, 1]
        img_name = os.path.join(self.root_dir, f"{image_id}.jpg")
        image = Image.open(img_name).convert("RGB")

        # compute the aesthetic score as the average of the ratings
        ratings = self.data_frame.iloc[idx, 2:12].values
        score = sum((i + 1) * ratings[i] for i in range(10)) / sum(ratings)

        if self.transform:
            image = self.transform(image)

        return image, score


# Example instantiation of the dataset
# tad66k_dataset = TAD66KDataset(csv_file='path_to_tad66k_csv', root_dir='path_to_tad66k_images', transform=transform)
# ava_dataset = AVADataset(txt_file='path_to_ava_txt', root_dir='path_to_ava_images', transform=transform)
