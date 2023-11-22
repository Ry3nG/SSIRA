import os
import torch
import torch.optim as optim
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torchvision.transforms as transforms


import pandas as pd
import matplotlib.pyplot as plt


from skimage.metrics import peak_signal_noise_ratio as compare_psnr

from utils.dataset import TAD66KDataset
from models.autoencoder import ConvAutoencoder

import logging
import datetime

current_time = datetime.datetime.now()

# set up logging console output format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(f"train_{current_time}.log")],
)

logging.info("Imported packages.")
logging.info("Training script started on %s." % current_time)

# constants
PATH_DATASET_TAD66K = "/home/zerui/SSIRA/dataset/TAD66K/"
PATH_LABEL_MERGE_TAD66K_TEST = "/home/zerui/SSIRA/dataset/TAD66K/labels/merge/test.csv"
PATH_LABEL_MERGE_TAD66K_TRAIN = (
    "/home/zerui/SSIRA/dataset/TAD66K/labels/merge/train.csv"
)

# load the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")


# define transforms
transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),  # Resize to 256x256
        transforms.ToTensor(),  # Convert image to PyTorch Tensor
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # Normalize with ImageNet stats
    ]
)

degradation_transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),  # Match the original transform size
        transforms.RandomHorizontalFlip(),  # Randomly flip the images horizontally
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
        ),  # Random color jitter,
        transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2)),
        transforms.Lambda(
            transforms.Lambda(lambda x: x + 0.05 * torch.randn_like(x))
        ),  # Adding random noise
    ]
)


# load the dataset
train_dataset = TAD66KDataset(
    csv_file=PATH_LABEL_MERGE_TAD66K_TRAIN,
    root_dir=PATH_DATASET_TAD66K,
    transform=transform,
    degradation_transform=degradation_transform,
)

train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])


train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=True, num_workers=8)

logging.info(f"Train dataset size: {len(train_dataset)}")
logging.info(f"Validation dataset size: {len(val_dataset)}")

# initialize the model
model = ConvAutoencoder()

if torch.cuda.device_count() > 1:
    logging.info(f"Using {torch.cuda.device_count()} GPUs!")
    model = torch.nn.DataParallel(model)
model.to(device)

logging.info(
    f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
)
logging.info("Model initialized.")


# define the loss function and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# Set up the scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True) 
logging.info("Scheduler initialized.")

# train the model
early_stopping_patience = 5  # Number of epochs to wait for improvement before stopping
best_val_loss = float("inf")  # Initialize best validation loss as infinity
epochs_no_improve = 0  # Counter for epochs with no improvement

num_epochs = 1000
best_val_loss = float("inf")  # Initialize best validation loss as infinity
early_stopping_patience = 3  # Number of epochs to wait for improvement before stopping
epochs_no_improve = 0  # Counter for epochs with no improvement
val_loss_history = []

scaler = GradScaler()

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    for i, (degraded_images, original_images) in enumerate(train_loader):
        degraded_images, original_images = degraded_images.to(
            device
        ), original_images.to(device)

        # Enables autocasting for the forward pass
        with autocast():
            reconstructed_images = model(degraded_images)
            loss = criterion(reconstructed_images, original_images)

        # Scales loss and calls backward()
        optimizer.zero_grad()
        scaler.scale(loss).backward()

        # scaler.step() first unscales the gradients of the optimizer's assigned params.
        # If these gradients do not contain Inf or NaN, optimizer.step() is then called,
        # otherwise, optimizer.step() is skipped.
        scaler.step(optimizer)

        # Updates the scale for next iteration
        scaler.update()

        running_loss += loss.item()

        if i % 10 == 0:
            logging.info(
                f"Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}"
            )

    # Validation phase
    model.eval()
    val_loss = 0.0
    #total_psnr = 0.0
    with torch.no_grad():
        for degraded_images, original_images in val_loader:
            degraded_images, original_images = degraded_images.to(
                device
            ), original_images.to(device)

            # Run the forward pass with autocasting
            with autocast():
                reconstructed_images = model(degraded_images)
                loss = criterion(reconstructed_images, original_images)

            val_loss += loss.item()

            # Convert images to numpy for PSNR
            for i in range(len(original_images)):
                orig = original_images[i].cpu().numpy().transpose(1, 2, 0)
                reconstructed = reconstructed_images[i].cpu().numpy().transpose(1, 2, 0)

                # Normalize the images between 0 and 1 for PSNR
                orig = (orig - orig.min()) / (orig.max() - orig.min())
                reconstructed = (reconstructed - reconstructed.min()) / (
                    reconstructed.max() - reconstructed.min()
                )

                #total_psnr += compare_psnr(orig, reconstructed)

    avg_val_loss = val_loss / len(val_loader)
    val_loss_history.append(avg_val_loss)

    #avg_psnr = total_psnr / len(val_loader.dataset)
    scheduler.step(avg_val_loss)

    # Early stopping check
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    if epochs_no_improve == early_stopping_patience:
        logging.info(f"Early stopping triggered after {epoch+1} epochs!")
        break

    logging.info(
        f"Epoch {epoch+1}/{num_epochs}, Train Loss: {running_loss/len(train_loader)}, Val Loss: {avg_val_loss}"
    )
    plt.figure(figsize=(10, 5))
    plt.plot(val_loss_history, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Validation Loss Over Epochs')
    plt.savefig(f"val_loss_{current_time}.png")
    plt.close()


# Save the trained model
if isinstance(model, torch.nn.DataParallel):
    torch.save(model.module.state_dict(), f"conv_autoencoder_{current_time}.pth")
else:
    torch.save(model.state_dict(), f"conv_autoencoder_{current_time}.pth")
