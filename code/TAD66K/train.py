import os

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau


import torchvision.transforms as transforms
import pandas as pd
import matplotlib.pyplot as plt

from data.dataset import TAD66KDataset
from models.contrasivemodel import ContrastiveModel
from models.contrasiveloss import ContrastiveLoss
from data.transforms import get_standard_transforms, get_degradation_transforms
from utils.utils import save_model, load_model, validate_model
import utils.constants as constants
import logging
import datetime

current_time = datetime.datetime.now()

# Setup logging
log_file = os.path.join(constants.PATH_LOGS, f"train_{current_time}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
)
logging.info("Imported packages.")
logging.info("Training script started on %s." % current_time)

# Load the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")
logging.info(torch.cuda.get_device_name(device))

# Load the dataset
train_dataset = TAD66KDataset(
    csv_file=constants.PATH_LABEL_MERGE_TAD66K_TRAIN,
    root_dir=constants.PATH_DATASET_TAD66K,
    transform=get_standard_transforms(),
    degradation_transform=get_degradation_transforms(),
)

# Split the dataset into train and validation
train_size = int(constants.TRAIN_VAL_SPLIT_RATIO * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    train_dataset, [train_size, val_size]
)

# Create the dataloaders
train_loader = DataLoader(
    train_dataset,
    batch_size=constants.BATCH_SIZE,
    shuffle=True,
    num_workers=constants.NUM_WORKERS,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=constants.BATCH_SIZE,
    shuffle=True,
    num_workers=constants.NUM_WORKERS,
)

# Load the model
model = ContrastiveModel().to(device)
logging.info("Loaded model.")
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    logging.info("Using %d GPUs for training." % torch.cuda.device_count())


# Define loss function and optimizer
criterion = ContrastiveLoss(temperature=0.5, device=device).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=constants.LEARNING_RATE)
scheduler = ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=constants.LR_DECAY_FACTOR,
    patience=constants.LR_DECAY_PATIENCE,
    threshold=constants.LR_DECAY_THRESHOLD,
    verbose=True,
)

# Train the model
scaler = GradScaler()
num_epochs = constants.NUM_EPOCHS
best_val_loss = float("inf")  # Initialize best validation loss as infinity
val_loss_history = []

early_stopping_patience = constants.EARLY_STOPPING_PATIENCE
early_stopping_counter = 0

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    for degraded_images, original_images in train_loader:
        degraded_images, original_images = degraded_images.to(
            device
        ), original_images.to(device)

        optimizer.zero_grad()  # Clear gradients

       # Inside the training loop
        with autocast():
            output1, output2 = model((degraded_images, original_images)) # tuple!
            loss = criterion(output1, output2)


        scaler.scale(loss).backward()  # Scales loss and calls backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    logging.info(f"Epoch [{epoch + 1}/{num_epochs}] Train loss: {avg_loss:.4f}")
    scheduler.step(avg_loss)

    # Validate the model
    # Inside training loop
    val_loss, avg_distance = validate_model(model, val_loader, criterion, device)
    logging.info(
        f"Epoch [{epoch + 1}/{num_epochs}] Validation Loss: {val_loss:.4f}, Avg Distance: {avg_distance:.2f}"
    )
    val_loss_history.append(val_loss)

    logging.info("----------------------------------------------")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stopping_counter = 0
        best_model_path = os.path.join(constants.PATH_MODELS, f"model_best_"+str(current_time)+".pth")
        save_model(model, optimizer, scheduler, epoch, best_model_path)
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= early_stopping_patience:
            logging.info(
                f"Validation loss did not improve for {early_stopping_patience} epochs. Training stopped."
            )
            break

    # Save model checkpoint
    checkpoint_path = os.path.join(
        constants.PATH_MODELS, f"model_epoch_{epoch + 1}_" + str(current_time)+".pth"
    )
    save_model(model, optimizer, scheduler, epoch, checkpoint_path)
