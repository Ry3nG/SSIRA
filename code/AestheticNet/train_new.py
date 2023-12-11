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
from utils.losses import ReconstructionLoss, AestheticScoreLoss
from utils.transforms import CustomTransform
from utils.setup_logging import setup_logging
from utils.argument_parser import parse_args


def main():
    # Training start time
    tic = datetime.datetime.now()
    global_start_time = tic.strftime("%Y-%m-%d_%H-%M-%S")

    # setup logging
    setup_logging(target_path=PATH_LOGS, current_time=global_start_time)

    # Parse arguments ----------------------------------------------------------
    tic = datetime.datetime.now()
    args = parse_args()
    BATCH_SIZE = args.batch_size
    PRETEXT_NUM_EPOCHS = args.pretext_num_epochs
    AES_NUM_EPOCHS = args.aes_num_epochs
    LEARNING_RATE_PRETEXT = args.lr_pretext
    LEARNING_RATE_AES = args.lr_aesthetic
    NUM_WORKERS = args.num_workers
    TRAIN_VAL_SPLIT_RATIO = args.train_val_split_ratio
    SAVE_FREQ = args.save_freq
    LR_PATIENCE = args.lr_patience
    LR_FACTOR = args.lr_factor
    LR_MODE = args.lr_mode
    LR_VERBOSE = args.lr_verbose

    toc = datetime.datetime.now()
    logging.info(f"Time elapsed for argument parsing: {toc - tic}")
    # log hyperparameters
    logging.info(f"Hyperparameters:")
    logging.info(f"Batch size: {BATCH_SIZE}")
    logging.info(f"Pretext num epochs: {PRETEXT_NUM_EPOCHS}")
    logging.info(f"Aesthetic num epochs: {AES_NUM_EPOCHS}")
    logging.info(f"Learning rate pretext: {LEARNING_RATE_PRETEXT}")
    logging.info(f"Learning rate aesthetic: {LEARNING_RATE_AES}")

    # Set device ----------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")
    logging.info(torch.cuda.get_device_name(device))

    # Define transforms ---------------------------------------------------------
    pretext_augmentation_options = list(range(24))
    aesthetics_augmentation_options = [22, 23]

    # Initialize model and loss function ----------------------------------------
    model = AestheticNet()
    if torch.cuda.device_count() > 1:
        logging.info(f"Using {torch.cuda.device_count()} GPUs")
        model = DataParallel(model)
    model = model.to(device)

    criterion_pretext = ReconstructionLoss().to(device)
    criterion_aes = AestheticScoreLoss.to(device)

    # initialize the optimizer
    optimizer_pretext = AdamW(model.parameters(), lr=LEARNING_RATE_PRETEXT)
    optimizer_aesthetic = AdamW(model.parameters(), lr=LEARNING_RATE_AES)

    # Initialize the learning rate scheduler
    # reduce LR on plateau
    scheduler_pretext = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer_pretext,
        mode=LR_MODE,
        factor=LR_FACTOR,
        patience=LR_PATIENCE,
        verbose=LR_VERBOSE,
        min_lr=LR_MIN,
    )
    scheduler_aesthetic = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer_aesthetic,
        mode=LR_MODE,
        factor=LR_FACTOR,
        patience=LR_PATIENCE,
        verbose=LR_VERBOSE,
        min_lr=LR_MIN,
    )

    # Initialize GradScaler for mixed precision
    scaler_pretext = GradScaler()
    scaler_aesthetic = GradScaler()

    # Initialize the dataset and split for training and validation
    full_train_dataset_pretext = TAD66KDataset_Split(
        csv_file=PATH_LABEL_MERGE_TAD66K_TRAIN,
        root_dir=PATH_DATASET_TAD66K,
        custom_transform_options=pretext_augmentation_options,
        default_transform=True,
        split="default",
    )
    train_size_pretext = int(TRAIN_VAL_SPLIT_RATIO * len(full_train_dataset_pretext))
    val_size_pretext = len(full_train_dataset_pretext) - train_size_pretext
    train_dataset_pretext, val_dataset_pretext = random_split(
        full_train_dataset_pretext, [train_size_pretext, val_size_pretext]
    )
    full_train_dataset_aesthetic = AVADataset_Split(
        [PATH_AVA_HLAGCN],
        PATH_AVA_IMAGE,
        custom_transform_options=aesthetics_augmentation_options,
        split="hlagcn",
    )
    train_size_aesthetic = int(
        TRAIN_VAL_SPLIT_RATIO * len(full_train_dataset_aesthetic)
    )
    val_size_aesthetic = len(full_train_dataset_aesthetic) - train_size_aesthetic
    train_dataset_aesthetic, val_dataset_aesthetic = random_split(
        full_train_dataset_aesthetic, [train_size_aesthetic, val_size_aesthetic]
    )
    # todo: logging dataset info


    # create dataloaders
    train_loader_pretext = DataLoader(
        train_dataset_pretext,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    val_loader_pretext = DataLoader(
        val_dataset_pretext,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    train_loader_aesthetic = DataLoader(
        train_dataset_aesthetic,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    val_loader_aesthetic = DataLoader(
        val_dataset_aesthetic,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
