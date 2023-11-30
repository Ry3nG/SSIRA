import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose
import os
import logging
import argparse
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast

# Import your model and other components
from models.GLINTnet import GLINTnet, GLINTnetSelfSupervised
from data.dataset import TAD66KDataset, AVADataset
from utils.constants import *

# Performance boosters
torch.backends.cudnn.benchmark = True

def setup_logging(current_time):
    log_file = os.path.join(PATH_LOGS, f"train_{current_time}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )


def read_ava_ids(file_path):
    with open(file_path, "r") as file:
        test_ids = [line.strip() for line in file]
    return test_ids



def parse_args():
    parser = argparse.ArgumentParser(description="Train AestheticNet")
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
        help="Number of workers for training",
    )
    parser.add_argument(
        "--train_val_split_ratio",
        type=float,
        default=TRAIN_VAL_SPLIT_RATIO,
        help="Ratio of training to validation data",
    )
    parser.add_argument(
        "--save_freq", type=int, default=SAVE_FREQ, help="Frequency of saving model"
    )
    parser.add_argument(
        "--lr_patience",
        type=int,
        default=LR_PATIENCE,
        help="Patience for learning rate scheduler",
    )
    parser.add_argument(
        "--lr_factor",
        type=float,
        default=LR_FACTOR,
        help="Factor for learning rate scheduler",
    )
    parser.add_argument(
        "--lr_mode", type=str, default=LR_MODE, help="Mode for learning rate scheduler"
    )
    parser.add_argument(
        "--lr_verbose",
        type=bool,
        default=LR_VERBOSE,
        help="Verbosity for learning rate scheduler",
    )
    parser.add_argument(
        "--lr_min",
        type=float,
        default=LR_MIN,
        help="Minimum learning rate for learning rate scheduler",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to a saved checkpoint (default: None)",
    )

    return parser.parse_args()
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
    return running_loss / len(dataloader)

# Parse arguments
args = parse_args()
setup_logging(datetime.now().strftime("%Y%m%d_%H%M%S"))

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize datasets and dataloaders
ava_generic_train_id = read_ava_ids(PATH_AVA_GENERIC_TRAIN_IDS)
tad66k_dataset = TAD66KDataset(PATH_LABEL_MERGE_TAD66K_TRAIN, PATH_DATASET_TAD66K, custom_transform_options=[23])
ava_dataset = AVADataset(PATH_AVA_TXT, PATH_AVA_IMAGE, custom_transform_options=[23], ids=ava_generic_train_id)

# Train/Val Split for AVA dataset
train_size = int(args.train_val_split_ratio * len(ava_dataset))
val_size = len(ava_dataset) - train_size
ava_train_dataset, ava_val_dataset = random_split(ava_dataset, [train_size, val_size])

tad66k_loader = DataLoader(tad66k_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
ava_train_loader = DataLoader(ava_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
ava_val_loader = DataLoader(ava_val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

# Phase 1: Self-Supervised Learning with TAD66K Dataset
model_self_supervised = GLINTnetSelfSupervised(GLINTnet(num_classes=10), input_features=2048, manipulation_options=list(range(24))).to(device)
optimizer_self_supervised = optim.Adam(model_self_supervised.parameters(), lr=args.learning_rate)
criterion_self_supervised = nn.MSELoss() 
scaler_self_supervised = GradScaler()  # For mixed precision training

for epoch in range(args.pretext_num_epochs):
    model_self_supervised.train()
    running_loss = 0.0
    for i,(inputs, _)in enumerate(tad66k_loader):
        inputs = inputs.to(device)

        optimizer_self_supervised.zero_grad()

        with autocast():  # Enable mixed precision
            reconstructed, _ = model_self_supervised(inputs)
            loss = criterion_self_supervised(reconstructed, inputs)

        scaler_self_supervised.scale(loss).backward()
        # Gradient clipping (example: clip to max norm of 1)
        torch.nn.utils.clip_grad_norm_(model_self_supervised.parameters(), 1)

        scaler_self_supervised.step(optimizer_self_supervised)
        scaler_self_supervised.update()

        running_loss += loss.item()
        logging.info(f"Epoch {epoch+1}, Batch {i+1}, Loss: {loss.item():.8f}" )  

    val_loss = validate(model_self_supervised, ava_val_loader, criterion_self_supervised, device)
    logging.info(f"Epoch {epoch+1}, Training Loss: {running_loss / len(tad66k_loader)}, Validation Loss: {val_loss}")

    if epoch % args.save_freq == args.save_freq - 1:
        torch.save(model_self_supervised.state_dict(), os.path.join(args.path_model_results, f'self_supervised_epoch_{epoch}.pth'))

logging.info("Finished Self-Supervised Training")

# Phase 2: Supervised Learning with AVA Dataset
model_supervised = GLINTnet(num_classes=10).to(device)
optimizer_supervised = optim.Adam(model_supervised.parameters(), lr=args.learning_rate)
criterion_supervised = nn.CrossEntropyLoss()
scheduler_supervised = ReduceLROnPlateau(optimizer_supervised, mode=args.lr_mode, factor=args.lr_factor, patience=args.lr_patience, verbose=args.lr_verbose, min_lr=args.lr_min)
scaler_supervised = GradScaler()  # For mixed precision training

for epoch in range(args.aes_num_epochs):
    model_supervised.train()
    running_loss = 0.0
    for i,(inputs, labels) in enumerate(ava_train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer_supervised.zero_grad()

        with autocast():  # Enable mixed precision
            outputs = model_supervised(inputs)
            loss = criterion_supervised(outputs, labels)

        scaler_supervised.scale(loss).backward()
        scaler_supervised.step(optimizer_supervised)
        scaler_supervised.update()

        running_loss += loss.item()
        logging.info(f"Epoch {epoch+1}, Batch {i+1}, Loss: {loss.item():.8f}" )
    val_loss = validate(model_supervised, ava_val_loader, criterion_supervised, device)
    scheduler_supervised.step(val_loss)
    logging.info(f"Epoch {epoch+1}, Training Loss: {running_loss / len(ava_train_loader)}, Validation Loss: {val_loss}")

    if epoch % args.save_freq == args.save_freq - 1:
        torch.save(model_supervised.state_dict(), os.path.join(args.path_model_results, f'supervised_epoch_{epoch}.pth'))

logging.info("Finished Supervised Training")
