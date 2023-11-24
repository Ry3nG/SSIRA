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
from models.degradationCNN import DegradationCNN
from utils.transforms import get_standard_transforms, get_degradation_transforms
from utils.utils import save_model, load_model, validate_model
import utils.constants as constants
import logging
import datetime

tic = datetime.datetime.now()

# Setup logging
log_file = os.path.join(constants.PATH_LOGS, f"train_{tic}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
)
logging.info("Imported packages.")
logging.info("Training parameters:")
logging.info("BATCH_SIZE: %d" % constants.BATCH_SIZE)
logging.info("NUM_EPOCHS: %d" % constants.NUM_EPOCHS)
logging.info("LEARNING_RATE: %f" % constants.LEARNING_RATE)
logging.info("NUM_WORKERS: %d" % constants.NUM_WORKERS)
logging.info("TRAIN_VAL_SPLIT_RATIO: %f" % constants.TRAIN_VAL_SPLIT_RATIO)
logging.info("LR_DECAY_FACTOR: %f" % constants.LR_DECAY_FACTOR)
logging.info("LR_DECAY_PATIENCE: %d" % constants.LR_DECAY_PATIENCE)
logging.info("LR_DECAY_THRESHOLD: %f" % constants.LR_DECAY_THRESHOLD)
logging.info("EARLY_STOPPING_PATIENCE: %d" % constants.EARLY_STOPPING_PATIENCE)
logging.info("Model: DegradationCNN")
logging.info("Loss function: CrossEntropyLoss + MSELoss")
logging.info("Optimizer: Adam")
logging.info("Learning rate scheduler: ReduceLROnPlateau")
logging.info("Training script started on %s." % tic)

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
model = DegradationCNN(num_degradation_types=5).to(device)
logging.info("Loaded model.")
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    logging.info("Using %d GPUs for training." % torch.cuda.device_count())


# Define loss function and optimizer
optimizer = optim.Adam(model.parameters(), lr=constants.LEARNING_RATE)
criterion_type = nn.CrossEntropyLoss()
criterion_level = nn.MSELoss()

scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=constants.LR_DECAY_FACTOR, patience=constants.LR_DECAY_PATIENCE, threshold=constants.LR_DECAY_THRESHOLD, verbose=True)

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
    
    for original_images,degraded_images, labels_type, labels_level in train_loader:
        original_images = original_images.to(device)
        degraded_images = degraded_images.to(device)
        labels_type = labels_type.to(device, dtype=torch.long)
        labels_level = labels_level.to(device, dtype=torch.float)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        with autocast():
            type_output, level_output = model(degraded_images)
            loss_type = criterion_type(type_output, labels_type)
            loss_level = criterion_level(level_output.view(-1), labels_level)
            loss = loss_type + loss_level

        # backward
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    logging.info(f"Epoch [{epoch + 1}/{num_epochs}] Train loss: {avg_loss:.4f}")
    

    # Validate the model
    # Inside training loop
    # After validation
    val_loss = validate_model(model, val_loader, criterion_type, criterion_level, device)
    logging.info(f"Epoch [{epoch + 1}/{num_epochs}] Validation Loss: {val_loss:.4f}")
    old_lr = optimizer.param_groups[0]["lr"]
    scheduler.step(val_loss)
    new_lr = optimizer.param_groups[0]["lr"]
    if old_lr != new_lr:
        logging.info(f"Learning rate changed from {old_lr} to {new_lr}")
        # reset early stopping counter
        early_stopping_counter = 0
    if new_lr < constants.LR_DECAY_THRESHOLD:
        logging.info(
            f"Learning rate {new_lr} is below the threshold {constants.LR_DECAY_THRESHOLD}. Training stopped."
        )
        break
    val_loss_history.append(val_loss)

    logging.info("----------------------------------------------")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stopping_counter = 0
        best_model_path = os.path.join(constants.PATH_MODELS, f"model_best_{tic}.pth")
        save_model(model, optimizer, None, epoch, best_model_path)
        logging.info("Saved best model.")
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= early_stopping_patience:
            logging.info("Early stopping triggered.")
            break

    # Save model checkpoint
    if (epoch + 1) % 10 == 0:
        checkpoint_path = os.path.join(constants.PATH_MODELS, f"model_epoch_{epoch + 1}_{tic}.pth")
        save_model(model, optimizer, None, epoch, checkpoint_path)
        logging.info("Saved model checkpoint.")


# Plot the validation loss history
plt.plot(val_loss_history)
plt.title("Validation Loss History")
plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.savefig(
    os.path.join(constants.PATH_PLOTS, f"val_loss_history_" + str(tic) + ".png")
)

logging.info("Training completed.")
toc = datetime.datetime.now()

# log training parameters:

logging.info("Training time: %s." % (toc - tic))
logging.info("Training script ended on %s." % toc)
