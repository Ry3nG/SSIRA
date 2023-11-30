import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import logging
import datetime

from models.GLINTnet import GLINTnet, GLINTnetSelfSupervised
from data.dataset import AVADataset,TAD66KDataset
from utils.constants import *


def setup_logging(current_time):
    log_file = os.path.join(PATH_LOGS, f"GLINTnet_train_{current_time}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )


def parse_args():
    parser = argparse.ArgumentParser(description="GLINTnet training script")
    parser.add_argument(
        "--batch_size", type=int, default=BATCH_SIZE, help="Batch size for training"
    )
    parser.add_argument(
        "--pretext_num_epochs",
        type=int,
        default=PRETEXT_NUM_EPOCHS,
        help="Number of epochs for pretext training",
    )
    parser.add_argument(
        "--aes_num_epochs",
        type=int,
        default=AES_NUM_EPOCHS,
        help="Number of epochs for aesthetic training",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=LEARNING_RATE,
        help="Learning rate for training",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=NUM_WORKERS,
        help="Number of workers for DataLoader",
    )
    parser.add_argument(
        "--train_val_split_ratio",
        type=float,
        default=TRAIN_VAL_SPLIT_RATIO,
        help="Train/Validation split ratio",
    )
    parser.add_argument(
        "--save_freq",
        type=int,
        default=SAVE_FREQ,
        help="Frequency for saving the model",
    )
    return parser.parse_args()


def train_self_supervised(
    model, data_loader, optimizer, criterion, epochs, device, train_start_time
):
    model.train()

    for epoch in range(epochs):
        for degraded_images, original_images in data_loader:
            print(f"Degraded images shape: {degraded_images.shape}")
            print(f"Original images shape: {original_images.shape}")
            degraded_images = degraded_images.to(device)
            original_images = original_images.to(device)

            optimizer.zero_grad()
            reconstructed_images, _ = model(degraded_images)
            loss = criterion(reconstructed_images, original_images)
            loss.backward()
            optimizer.step()

            logging.info(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.8f}")
        val_loss = validate(model, data_loader, criterion, device)
        logging.info(f"Validation Loss Epoch [{epoch+1}/{epochs}]: {val_loss:.8f}")

        if (epoch + 1) % SAVE_FREQ == 0:
            save_model(
                model,
                epoch,
                PATH_MODEL_RESULTS,
                "GLINTnet" + "_Phase_Pretext" + str(epoch),
                train_start_time,
            )


def train_supervised(
    model, data_loader, optimizer, criterion, epochs, device, train_start_time
):
    model.train()

    for epoch in range(epochs):
        for images, scores in data_loader:
            images = images.to(device)
            scores = scores.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, scores)
            loss.backward()
            optimizer.step()

            logging.info(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.8f}")
        val_loss = validate(model, data_loader, criterion, device)
        logging.info(f"Validation Loss Epoch [{epoch+1}/{epochs}]: {val_loss:.8f}")

        if (epoch + 1) % SAVE_FREQ == 0:
            save_model(
                model,
                epoch,
                PATH_MODEL_RESULTS,
                "GLINTnet" + "_Phase_AES" + str(epoch),
                train_start_time,
            )


def validate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for data in data_loader:
            inputs, targets = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    avg_loss = total_loss / len(data_loader)
    return avg_loss


def read_ava_ids(file_path):
    with open(file_path, "r") as file:
        test_ids = [line.strip() for line in file]
    return test_ids


def save_training_checkpoint(
    model, epoch, save_dir, filename, current_time, optimizer, scheduler, scaler
):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f"{filename}_epoch_{epoch}_{current_time}.pth")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
        },
        save_path,
    )


def save_model(model, epoch, save_dir, filename, current_time):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f"{filename}_epoch_{epoch}_{current_time}.pth")
    torch.save(model.state_dict(), save_path)
    logging.info(f"Checkpoint saved: {save_path}")


def main():
    tic = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    setup_logging(tic)
    logging.info(f"GLINTnet training script started at {tic}")
    args = parse_args()

    # Now use args to get the command line arguments
    batch_size = args.batch_size
    pretext_num_epochs = args.pretext_num_epochs
    aes_num_epochs = args.aes_num_epochs
    learning_rate = args.learning_rate
    num_workers = args.num_workers
    train_val_split_ratio = args.train_val_split_ratio
    save_freq = args.save_freq

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    # Load the dataset
    full_self_supervised_dataset = TAD66KDataset(
        csv_file=PATH_LABEL_MERGE_TAD66K_TRAIN,
        root_dir=PATH_DATASET_TAD66K,
        custom_transform_options=[23],
        default_transform=True,
    )
    ava_generic_train_id = read_ava_ids(PATH_AVA_GENERIC_TRAIN_IDS)
    full_supervised_dataset = AVADataset(
        txt_file=PATH_AVA_TXT,
        root_dir=PATH_AVA_IMAGE,
        custom_transform_options=[23],
        default_transform=True,
        ids=ava_generic_train_id,
        include_ids=True,
    )

    # Split the dataset into train and validation
    self_supervised_train_size = int(
        train_val_split_ratio * len(full_self_supervised_dataset)
    )
    self_supervised_val_size = (
        len(full_self_supervised_dataset) - self_supervised_train_size
    )
    supervised_train_size = int(train_val_split_ratio * len(full_supervised_dataset))
    supervised_val_size = len(full_supervised_dataset) - supervised_train_size

    (
        self_supervised_train_dataset,
        self_supervised_val_dataset,
    ) = torch.utils.data.random_split(
        full_self_supervised_dataset,
        [self_supervised_train_size, self_supervised_val_size],
    )
    supervised_train_dataset, supervised_val_dataset = torch.utils.data.random_split(
        full_supervised_dataset, [supervised_train_size, supervised_val_size]
    )

    # Create the dataloaders
    self_supervised_train_loader = DataLoader(
        self_supervised_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    self_supervised_val_loader = DataLoader(
        self_supervised_val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    supervised_train_loader = DataLoader(
        supervised_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    supervised_val_loader = DataLoader(
        supervised_val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    # Initialize the model
    base_model = GLINTnet(
        num_classes=1, is_classification=False
    )  # example for a simple regression model
    model = GLINTnetSelfSupervised(
        base_model, input_features=2048, manipulation_options=list(range(24))
    ).to(device)

    # Define the loss function and optimizer
    criterion_self_supervised = nn.MSELoss()
    criterion_supervised = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # self-supervised training
    logging.info("Starting self-supervised training...")
    train_self_supervised(
        model,
        self_supervised_train_loader,
        optimizer,
        criterion_self_supervised,
        pretext_num_epochs,
        device,
        tic,
    )

    # supervised training
    model = base_model.to(device)
    logging.info("Starting supervised training...")
    train_supervised(
        model,
        supervised_train_loader,
        optimizer,
        criterion_supervised,
        aes_num_epochs,
        device,
        tic,
    )


if __name__ == "__main__":
    main()
