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
    aesthetics_augmentation_options = [22,23]

    # Initialize model and loss function ----------------------------------------
    model = AestheticNet()
    if torch.cuda.device_count() > 1:
        logging.info(f"Using {torch.cuda.device_count()} GPUs")
        model = DataParallel(model)
    model = model.to(device)

    criterion_pretext = ReconstructionLoss().to(device)
    criterion_aes = AestheticScoreLoss().to(device)

