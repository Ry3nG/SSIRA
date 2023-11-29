from matplotlib import pyplot as plt
import torch
import torch.optim as optim
from torch.optim import AdamW
from torch.nn import DataParallel
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, random_split
from data.dataset import TAD66KDataset, AVADataset
from models.aestheticNet import AestheticNet
from utils.losses import ReconstructionLoss, AestheticScoreLoss
from utils.constants import *
from utils.transforms import CustomTransform

import argparse
import logging
import datetime
import os


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


def train(model, dataloader, criterion, optimizer, scaler, device, phase, epoch, total_epochs):
    model.train()  # set model to training mode
    total_loss = 0.0

    for batch_idx, batch in enumerate(dataloader):
        if phase == "pretext":
            inputs = batch.to(device)
            optimizer.zero_grad()
            outputs = model(inputs, phase=phase)
            loss = criterion(outputs, inputs)
        elif phase == "aesthetic":
            images, scores = batch
            images, scores = images.to(device), scores.to(device)
            optimizer.zero_grad()
            outputs = model(images, phase=phase)
            loss = criterion(outputs, scores)

        # Scale loss and call backward
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        if batch_idx % NUM_WORKERS == 0:
            logging.info(
                f"Epoch {epoch+1}/{total_epochs}, Phase: {phase}, Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.8f}"
            )

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device, phase):
    model.eval()  # set model to evaluation mode
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            if phase == "pretext":
                inputs = batch.to(device)
                outputs = model(inputs, phase=phase)
                loss = criterion(outputs, inputs)
            elif phase == "aesthetic":
                images, scores = batch
                images, scores = images.to(device), scores.to(device)
                outputs = model(images, phase=phase)
                loss = criterion(outputs, scores)
            total_loss += loss.item()

    return total_loss / len(dataloader)


def save_model(model, epoch, save_dir, filename, current_time):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f"{filename}_epoch_{epoch}_{current_time}.pth")
    torch.save(model.state_dict(), save_path)


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

def load_checkpoint(model, optimizer, scheduler, scaler, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])
    epoch = checkpoint["epoch"]
    return model, optimizer, scheduler, scaler, epoch

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


def main():
    # save training start time
    tic = datetime.datetime.now()
    tic = tic.strftime("%Y-%m-%d_%H-%M-%S")

    # setup logging
    setup_logging(tic)

    # parse arguments
    logging.info("Parsing arguments...")
    args = parse_args()
    BATCH_SIZE = args.batch_size
    PRETEXT_NUM_EPOCHS = args.pretext_num_epochs
    AES_NUM_EPOCHS = args.aes_num_epochs
    LEARNING_RATE = args.learning_rate
    NUM_WORKERS = args.num_workers
    TRAIN_VAL_SPLIT_RATIO = args.train_val_split_ratio
    SAVE_FREQ = args.save_freq
    LR_PATIENCE = args.lr_patience
    LR_FACTOR = args.lr_factor
    LR_MODE = args.lr_mode
    LR_VERBOSE = args.lr_verbose
    LR_MIN = args.lr_min

    # training start message
    logging.info(f"Training started at {tic} ------------------------------")

    # Logging the hyperparameters
    logging.info(f"Batch Size: {BATCH_SIZE}")
    logging.info(f"Number of Epochs for Pretext Phase: {PRETEXT_NUM_EPOCHS}")
    logging.info(f"Number of Epochs for Aesthetic Phase: {AES_NUM_EPOCHS}")
    logging.info(f"Learning Rate: {LEARNING_RATE}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    logging.info(torch.cuda.get_device_name(device))

    # define transforms
    custom_transform_options = list(range(24))

    # initialize the model and loss function
    model = AestheticNet()
    if torch.cuda.device_count() > 1:
        model = DataParallel(model)
    model.to(device)
    criterion_pretext = ReconstructionLoss().to(device)
    criterion_aesthetic = AestheticScoreLoss().to(device)

    # Initialize GradScaler for mixed precision
    scaler_pretext = GradScaler()
    scaler_aesthetic = GradScaler()

    # initialize the optimizer
    optimizer_pretext = AdamW(model.parameters(), lr=LEARNING_RATE)
    optimizer_aesthetic = AdamW(model.parameters(), lr=LEARNING_RATE)

    # Initialize the learning rate scheduler
    scheduler_pretext = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_pretext,
        mode=LR_MODE,
        factor=LR_FACTOR,
        patience=LR_PATIENCE,
        verbose=LR_VERBOSE,
        min_lr=LR_MIN,
    )
    scheduler_aesthetic = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_aesthetic,
        mode=LR_MODE,
        factor=LR_FACTOR,
        patience=LR_PATIENCE,
        verbose=LR_VERBOSE,
        min_lr=LR_MIN,
    )

    logging.info("Model initialized.")

    # initialize and split the datasets for training and validation

    ava_generic_train_id = read_ava_ids(PATH_AVA_GENERIC_TRAIN_IDS)

    full_train_dataset_pretext = TAD66KDataset(
        csv_file=PATH_LABEL_MERGE_TAD66K_TRAIN,
        root_dir=PATH_DATASET_TAD66K,
        custom_transform_options=custom_transform_options,
    )

    train_size_pretext = int(TRAIN_VAL_SPLIT_RATIO * len(full_train_dataset_pretext))
    val_size_pretext = len(full_train_dataset_pretext) - train_size_pretext
    train_dataset_pretext, val_dataset_pretext = random_split(
        full_train_dataset_pretext, [train_size_pretext, val_size_pretext]
    )

    full_train_dataset_aesthetic = AVADataset(
        txt_file=PATH_AVA_TXT,
        root_dir=PATH_AVA_IMAGE,
        custom_transform_options=custom_transform_options,
        ids=ava_generic_train_id,
        include_ids=True,
    )
    train_size_aesthetic = int(
        TRAIN_VAL_SPLIT_RATIO * len(full_train_dataset_aesthetic)
    )
    val_size_aesthetic = len(full_train_dataset_aesthetic) - train_size_aesthetic
    train_dataset_aesthetic, val_dataset_aesthetic = random_split(
        full_train_dataset_aesthetic, [train_size_aesthetic, val_size_aesthetic]
    )

    logging.info("Datasets initialized.")
    # log dataset sizes
    logging.info(f"Pretext Train Dataset Size: {len(train_dataset_pretext)}")
    logging.info(f"Pretext Validation Dataset Size: {len(val_dataset_pretext)}")
    logging.info(f"Aesthetic Train Dataset Size: {len(train_dataset_aesthetic)}")
    logging.info(f"Aesthetic Validation Dataset Size: {len(val_dataset_aesthetic)}")

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

    val_losses_pretext = []
    val_losses_aesthetic = []

    start_epoch = 0
    start_mode = "pretext"
    if args.checkpoint_path:
        # decide it is pretext or aesthetic based on the checkpoint path
        if "pretext" in args.checkpoint_path:
            # pretext
            model, optimizer_pretext, scheduler_pretext, scaler_pretext, start_epoch = load_checkpoint(model, optimizer_pretext, scheduler_pretext, scaler_pretext, args.checkpoint_path)
            start_mode = "pretext"
            logging.info(f"Loaded pretext checkpoint from {args.checkpoint_path}")
        elif "aesthetic" in args.checkpoint_path:
            # aesthetic
            model, optimizer_aesthetic, scheduler_aesthetic, scaler_aesthetic, start_epoch = load_checkpoint(model, optimizer_aesthetic, scheduler_aesthetic, scaler_aesthetic, args.checkpoint_path)
            start_mode = "aesthetic"
            logging.info(f"Loaded aesthetic checkpoint from {args.checkpoint_path}")
        else:
            raise ValueError("Invalid checkpoint path")


    if start_mode == "pretext":
        # training and validation loop
        for epoch in range(start_epoch,PRETEXT_NUM_EPOCHS):
            # Train in pretext phase
            train_loss_pretext = train(
                model,
                train_loader_pretext,
                criterion_pretext,
                optimizer_pretext,
                scaler_pretext,
                device,
                "pretext",
                epoch,
                PRETEXT_NUM_EPOCHS,
            )
            val_loss_pretext = validate(
                model, val_loader_pretext, criterion_pretext, device, "pretext"
            )
            old_lr = optimizer_pretext.param_groups[0]["lr"]
            scheduler_pretext.step(val_loss_pretext)
            new_lr = optimizer_pretext.param_groups[0]["lr"]
            if old_lr != new_lr:
                logging.info(
                    f"Epoch {epoch+1}/{PRETEXT_NUM_EPOCHS}, Pretext Phase, Learning Rate Changed from {old_lr} to {new_lr}"
                )

            val_losses_pretext.append(val_loss_pretext)

            logging.info(
                f"Epoch {epoch+1}/{PRETEXT_NUM_EPOCHS}, Pretext Phase, Train Loss: {train_loss_pretext:.8f}, Val Loss: {val_loss_pretext:.8f}"
            )
            # Save model at specified frequency
            if (epoch + 1) % SAVE_FREQ == 0 or epoch == PRETEXT_NUM_EPOCHS - 1:
                logging.info("Saving checkpoint...")
                save_training_checkpoint(
                    model,
                    epoch,
                    PATH_MODEL_RESULTS,
                    "aestheticNet-checkpoint-pretext",
                    tic,
                    optimizer_pretext,
                    scheduler_pretext,
                    scaler_pretext,
                )
    for epoch in range(start_epoch,AES_NUM_EPOCHS):
        # Train in aesthetic phase
        train_loss_aesthetic = train(
            model,
            train_loader_aesthetic,
            criterion_aesthetic,
            optimizer_aesthetic,
            scaler_aesthetic,
            device,
            "aesthetic",
            epoch,
            AES_NUM_EPOCHS,
        )
        val_loss_aesthetic = validate(
            model, val_loader_aesthetic, criterion_aesthetic, device, "aesthetic"
        )
        old_lr = optimizer_aesthetic.param_groups[0]["lr"]
        scheduler_aesthetic.step(val_loss_aesthetic)
        new_lr = optimizer_aesthetic.param_groups[0]["lr"]
        if old_lr != new_lr:
            logging.info(
                f"Epoch {epoch+1}/{AES_NUM_EPOCHS}, Aesthetic Phase, Learning Rate Changed from {old_lr} to {new_lr}"
            )
        val_losses_aesthetic.append(val_loss_aesthetic)

        logging.info(
            f"Epoch {epoch+1}/{AES_NUM_EPOCHS}, Aesthetic Phase, Train Loss: {train_loss_aesthetic:.8f}, Val Loss: {val_loss_aesthetic:.8f}"
        )
        # Save model at specified frequency
        if (epoch + 1) % SAVE_FREQ == 0 or epoch == AES_NUM_EPOCHS - 1:
            logging.info("Saving checkpoint...")
            save_training_checkpoint(
                model,
                epoch,
                PATH_MODEL_RESULTS,
                "aestheticNet-checkpoint-aesthetic",
                tic,
                optimizer_aesthetic,
                scheduler_aesthetic,
                scaler_aesthetic,
            )

    # Plotting the validation loss (2 plots)
    plt.figure(figsize=(10, 5))
    plt.plot(val_losses_pretext, label="Pretext Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Validation Losses Over Epochs")
    plt.legend()
    plt.savefig(os.path.join(PATH_PLOTS, "val_losses_pretext.png"))

    plt.figure(figsize=(10, 5))
    plt.plot(val_losses_aesthetic, label="Aesthetic Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Validation Losses Over Epochs")
    plt.legend()
    plt.savefig(os.path.join(PATH_PLOTS, "val_losses_aesthetic.png"))

if __name__ == "__main__":
    main()
