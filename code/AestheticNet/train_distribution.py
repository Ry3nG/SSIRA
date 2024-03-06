import argparse
import datetime
import logging
import os

import pandas as pd
import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.cuda.amp import GradScaler, autocast
from torch.nn import DataParallel
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split

from data.dataset_split import AVADataset_Split, TAD66KDataset_Split
from models.aestheticNet import AestheticNet
from utils.constants import *
from utils.losses import ReconstructionLoss, EMDLoss
from utils.transforms import CustomTransform
from utils.setup_logging import setup_logging
from utils.argument_parser import parse_args


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_one_epoch(model, dataloader, criterion, optimizer, device, phase):
    model.train()
    total_loss = 0
    for batch in dataloader:
        if phase == 'pretext':
            images = batch.to(device)
            optimizer.zero_grad()
            reconstructed_images = model(images, phase=phase)
            loss = criterion(reconstructed_images, images)
        elif phase == 'aesthetic':
            images, _, score_distributions, _ = batch
            images = images.to(device)
            score_distributions = score_distributions.to(device)
            optimizer.zero_grad()
            outputs = model(images, phase=phase)
            loss = criterion(outputs, score_distributions)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device, phase):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            if phase == 'pretext':
                images = batch.to(device)
                reconstructed_images = model(images, phase=phase)
                loss = criterion(reconstructed_images, images)
            elif phase == 'aesthetic':
                images, _, score_distributions, _ = batch
                images = images.to(device)
                score_distributions = score_distributions.to(device)
                outputs = model(images, phase=phase)
                loss = criterion(outputs, score_distributions)

            total_loss += loss.item()
    return total_loss / len(dataloader)

def save_model(model, filename):
    torch.save(model.state_dict(), filename)

def load_model(model, filename, device):
    model.load_state_dict(torch.load(filename, map_location=device))
    return model

def main():
    # Paths and parameters
    csv_files = ["path_to_train_hlagcn.csv", "path_to_train_mlsp.csv"]
    root_dir = "path_to_image_directory"
    custom_transform_options = [24]  # Add your custom transform options

    # Device configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Model initialization
    model = AestheticNet(num_scores=10).to(device)

    # Phase 1: Pretext Training
    pretext_dataset = TAD66KDataset_Split(csv_file="path_to_pretext_dataset.csv", root_dir=root_dir, custom_transform_options=custom_transform_options, split="default")
    pretext_dataloader = DataLoader(pretext_dataset, batch_size=32, shuffle=True, num_workers=4)
    pretext_criterion = ReconstructionLoss(device=device).to(device)
    pretext_optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):  # Adjust number of epochs as needed
        train_loss = train_one_epoch(model, pretext_dataloader, pretext_criterion, pretext_optimizer, device, phase='pretext')
        logging.info(f'Pretext Phase - Epoch {epoch+1}, Train Loss: {train_loss:.4f}')

    save_model(model, 'aestheticnet_pretext.pth')

    # Phase 2: Aesthetic Training
    aesthetic_dataset = AVADataset_Split(csv_files, root_dir, custom_transform_options, split="hlagcn")
    aesthetic_dataloader = DataLoader(aesthetic_dataset, batch_size=32, shuffle=True, num_workers=4)
    aesthetic_criterion = EMDLoss().to(device)
    aesthetic_optimizer = optim.Adam(model.parameters(), lr=0.001)

    model = load_model(model, 'aestheticnet_pretext.pth', device)

    for epoch in range(50):  # Adjust number of epochs as needed
        train_loss = train_one_epoch(model, aesthetic_dataloader, aesthetic_criterion, aesthetic_optimizer, device, phase='aesthetic')
        val_loss = validate(model, aesthetic_dataloader, aesthetic_criterion, device, phase='aesthetic')
        logging.info(f'Aesthetic Phase - Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

if __name__ == "__main__":
    main()
