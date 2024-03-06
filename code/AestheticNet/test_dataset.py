import os
from data.dataset_split import AVADataset_Split
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid as make_grids
from utils.constants import PATH_PLOTS

# Define paths to the CSV files and root directory
csv_files = ["/home/zerui/SSIRA/dataset/AVA_Split/train_hlagcn.csv", "/home/zerui/SSIRA/dataset/AVA_Split/train_mlsp.csv"]
root_dir = "/home/zerui/SSIRA/dataset/AVA/AVA_dataset/image"

# Custom transform options (optional)
custom_transform_options = [24]

# Create datasets for both splits
hlagcn_dataset = AVADataset_Split(csv_files, root_dir, custom_transform_options, split="hlagcn")
mlsp_dataset = AVADataset_Split(csv_files, root_dir, custom_transform_options, split="mlsp")


def denormalize(image):
    """
    Denormalizes a torch image to its original range.
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(image.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(image.device)
    return image * std + mean

# Test loading a sample from each dataset
image, score, details, binary_score = hlagcn_dataset[0]

print(f"HLAGCN: Image loaded successfully, score: {score}, details: {details}, binary_score: {binary_score}")

image, score, details = mlsp_dataset[0]

print(f"MLSP: Image loaded successfully, score: {score}, details: {details}")

# Test using a custom transform
image, score, details, binary_score = hlagcn_dataset[1]

print(f"HLAGCN with custom transform: Image loaded successfully")
    

