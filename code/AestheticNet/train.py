from matplotlib import pyplot as plt
import pandas as pd
import torch
import torch.optim as optim
from torch.optim import AdamW
from torch.nn import DataParallel
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from data.dataset import TAD66KDataset, AVADataset
from data.dataset_split import AVADataset_Split

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
        "--learning_rate_pretext",
        type=float,
        default=LEARNING_RATE,
        help="Learning rate for pretext training",
    )
    parser.add_argument(
        "--learning_rate_aesthetic",
        type=float,
        default=LEARNING_RATE,
        help="Learning rate for aesthetic training",
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
        help="Path to a saved checkpoint (default: None)", # /home/zerui/SSIRA/code/AestheticNet/results/Models/aestheticNet-ready-for-aesthetics.pth
    )

    return parser.parse_args()


def train(
    model,
    dataloader,
    criterion,
    optimizer,
    scaler,
    device,
    phase,
    epoch,
    total_epochs,
):
    model.train()  # set model to training mode
    total_loss = 0.0
    if phase == "pretext":
        logging.info(f"Training in pretext phase, Epoch {epoch+1}/{total_epochs}")
        for batch_idx, batch in enumerate(dataloader):
            inputs = batch.to(device)
            optimizer.zero_grad()
            outputs = model(inputs, phase=phase)
            loss = criterion(outputs, inputs)

            # Scale loss and call backward
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            if batch_idx % NUM_WORKERS == 0:
                logging.info(
                    f"Epoch {epoch+1}/{total_epochs}, Phase: {phase}, Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.8f}"
                )    
    elif phase == "aesthetic":
        
        logging.info(f"Training in aesthetic phase, Epoch {epoch+1}/{total_epochs}")
        for batch_idx, data in enumerate(dataloader):
            images = data[0]
            scores = data[1]
            images, scores = images.to(device), scores.to(device)
            optimizer.zero_grad()
            outputs = model(images, phase=phase)
            loss = criterion(outputs, scores)

            # Scale loss and call backward
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            # Logging batch level information
            if batch_idx % NUM_WORKERS == 0:
                logging.info(
                    f"Epoch {epoch+1}/{total_epochs}, Phase: {phase}, Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.8f}"
                )


    return total_loss / len(dataloader)

    """
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
    """


def validate(model, dataloader, criterion, device, phase):
    model.eval()  # set model to evaluation mode
    total_loss = 0.0

    with torch.no_grad():
        if phase == "pretext":
            for batch in dataloader:
                inputs = batch.to(device)
                outputs = model(inputs, phase=phase)
                loss = criterion(outputs, inputs)
                total_loss += loss.item()
        if phase == "aesthetic":
            for data in dataloader:
                images = data[0]
                scores = data[1]
                images, scores = images.to(device), scores.to(device)
                outputs = model(images, phase=phase)
                loss = criterion(outputs, scores)
                total_loss += loss.item()

    return total_loss / len(dataloader)


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
    return save_path


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

    # Tensorboard writer
    writer = SummaryWriter(log_dir=os.path.join(PATH_LOGS, "tensorboard", tic))

    # parse arguments
    logging.info("Parsing arguments...")
    args = parse_args()
    BATCH_SIZE = args.batch_size
    PRETEXT_NUM_EPOCHS = args.pretext_num_epochs
    AES_NUM_EPOCHS = args.aes_num_epochs
    LEARNING_RATE_AES = args.learning_rate_aesthetic
    LEARNING_RATE_PRETEXT = args.learning_rate_pretext
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
    logging.info(f"Learning Rate for Pretext Phase: {LEARNING_RATE_PRETEXT}")
    logging.info(f"Learning Rate for Aesthetic Phase: {LEARNING_RATE_AES}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    logging.info(torch.cuda.get_device_name(device))

    # define transforms
    custom_transform_options = list(range(24))
    only_horizontal_flip = [22, 23]

    # initialize the model and loss function
    model = AestheticNet()
    if torch.cuda.device_count() > 1:
        model = DataParallel(model)
    model.to(device)

    # initialize the loss function
    criterion_pretext = ReconstructionLoss(device=device).to(device)
    criterion_aesthetic = AestheticScoreLoss().to(device)

    # initialize the optimizer
    optimizer_pretext = AdamW(model.parameters(), lr=LEARNING_RATE_PRETEXT)
    optimizer_aesthetic = AdamW(model.parameters(), lr=LEARNING_RATE_AES)

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

    # Initialize GradScaler for mixed precision
    scaler_pretext = GradScaler()
    scaler_aesthetic = GradScaler()

    logging.info("Model initialized.")


    # "/home/zerui/SSIRA/dataset/TAD66K/labels/merge/train_first_1000.csv"
    full_train_dataset_pretext = TAD66KDataset(
        csv_file= PATH_LABEL_MERGE_TAD66K_TRAIN,
        root_dir=PATH_DATASET_TAD66K,
        custom_transform_options=custom_transform_options,
    )

    train_size_pretext = int(TRAIN_VAL_SPLIT_RATIO * len(full_train_dataset_pretext))
    val_size_pretext = len(full_train_dataset_pretext) - train_size_pretext
    train_dataset_pretext, val_dataset_pretext = random_split(
        full_train_dataset_pretext, [train_size_pretext, val_size_pretext]
    )
    
    """
    full_train_dataset_aesthetic = AVADataset(
        txt_file=PATH_AVA_TXT,
        root_dir=PATH_AVA_IMAGE,
        custom_transform_options=only_horizontal_flip,
        #ids=ava_generic_train_id,
        #include_ids=True,
    )
    train_size_aesthetic = int(
        TRAIN_VAL_SPLIT_RATIO * len(full_train_dataset_aesthetic)
    )
    val_size_aesthetic = len(full_train_dataset_aesthetic) - train_size_aesthetic
    train_dataset_aesthetic, val_dataset_aesthetic = random_split(
        full_train_dataset_aesthetic, [train_size_aesthetic, val_size_aesthetic]
    )
    """

    full_train_dataset_aesthetic = AVADataset_Split(
        csv_files = ["/home/zerui/SSIRA/dataset/AVA_Split/train_hlagcn.csv"],
        root_dir = PATH_AVA_IMAGE,
        custom_transform_options=[23],
        split="hlagcn",
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

    # create logging plot save directory
    if not os.path.exists(PATH_PLOTS):
        os.makedirs(PATH_PLOTS)
    saving_dir = os.path.join(PATH_PLOTS, "train_plot_" + tic)
    os.makedirs(saving_dir)

    val_losses_pretext = []
    val_losses_aesthetic = []
    train_losses_pretext = []
    train_losses_aesthetic = []

    start_epoch = 0
    start_mode = "pretext"
    if args.checkpoint_path:
        # decide it is pretext or aesthetic based on the checkpoint path
        if "pretext" in args.checkpoint_path:
            # pretext
            (
                model,
                optimizer_pretext,
                scheduler_pretext,
                scaler_pretext,
                start_epoch,
            ) = load_checkpoint(
                model,
                optimizer_pretext,
                scheduler_pretext,
                scaler_pretext,
                args.checkpoint_path,
            )
            start_mode = "pretext"
            logging.info(f"Loaded pretext checkpoint from {args.checkpoint_path}")
        elif "aesthetic" in args.checkpoint_path:
            # aesthetic
            (
                model,
                optimizer_aesthetic,
                scheduler_aesthetic,
                scaler_aesthetic,
                start_epoch,
            ) = load_checkpoint(
                model,
                optimizer_aesthetic,
                scheduler_aesthetic,
                scaler_aesthetic,
                args.checkpoint_path,
            )
            start_mode = "aesthetic"
            logging.info(f"Loaded aesthetic checkpoint from {args.checkpoint_path}")
        else:
            raise ValueError("Invalid checkpoint path")

    if start_mode == "pretext":
        logging.info("Starting pretext phase training...")
        for epoch in range(start_epoch, PRETEXT_NUM_EPOCHS):
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
                model,
                val_loader_pretext,
                criterion_pretext,
                device,
                "pretext",
            )
            old_lr = optimizer_pretext.param_groups[0]["lr"]
            scheduler_pretext.step(val_loss_pretext)
            new_lr = optimizer_pretext.param_groups[0]["lr"]
            if old_lr != new_lr:
                logging.info(
                    f"Epoch {epoch+1}/{PRETEXT_NUM_EPOCHS}, Pretext Phase, Learning Rate Changed from {old_lr} to {new_lr}"
                )

            val_losses_pretext.append(val_loss_pretext)
            train_losses_pretext.append(train_loss_pretext)

            logging.info(
                f"Epoch {epoch+1}/{PRETEXT_NUM_EPOCHS}, Pretext Phase, Train Loss: {train_loss_pretext:.8f}, Val Loss: {val_loss_pretext:.8f}"
            )
            writer.add_scalar("Validation Loss pretext", val_loss_pretext, epoch)
            writer.add_scalar("Training Loss pretext", train_loss_pretext, epoch)

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
        pretext_checkpoint_path = save_training_checkpoint(
            model,
            PRETEXT_NUM_EPOCHS,
            PATH_MODEL_RESULTS,
            "aestheticNet-pretext",
            tic,
            optimizer_pretext,
            scheduler_pretext,
            scaler_pretext,
        )
        logging.info(f"Pretext phase model saved at {pretext_checkpoint_path}")

    plt.figure(figsize=(10, 5))
    plt.plot(val_losses_pretext, label="Pretext Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Validation Losses Over Epochs")
    plt.legend()
    plt.savefig(os.path.join(saving_dir, "val_losses_pretext" + tic + ".png"))

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses_pretext, label="Pretext Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Losses Over Epochs")
    plt.legend()
    plt.savefig(os.path.join(saving_dir, "train_losses_pretext" + tic + ".png"))

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses_pretext, label="Pretext Training Loss")
    plt.plot(val_losses_pretext, label="Pretext Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Losses Over Epochs")
    plt.legend()
    plt.savefig(os.path.join(saving_dir, "train_val_losses_pretext" + tic + ".png"))

    # ---------------------------------Aesthetic Phase---------------------------------#

    # handle checkpoint
    if start_mode == 'aesthetic':
        pretext_checkpoint_path = args.checkpoint_path
    # load the model for aesthetic phase
    if os.path.exists(pretext_checkpoint_path):
        (
            model,
            optimizer_aesthetic,
            scheduler_aesthetic,
            scaler_aesthetic,
            _,
        ) = load_checkpoint(
            model,
            optimizer_aesthetic,
            scheduler_aesthetic,
            scaler_aesthetic,
            pretext_checkpoint_path,
        )
        logging.info("Loaded model from pretext phase for aesthetic training.")
        logging.info(f"Loaded model: {pretext_checkpoint_path}")
    else:
        logging.warning("Pretext model not found. Continuing with current model state.")

    for epoch in range(start_epoch, AES_NUM_EPOCHS):
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
            model,
            val_loader_aesthetic,
            criterion_aesthetic,
            device,
            "aesthetic",
        )
        old_lr = optimizer_aesthetic.param_groups[0]["lr"]
        scheduler_aesthetic.step(val_loss_aesthetic)
        new_lr = optimizer_aesthetic.param_groups[0]["lr"]
        if old_lr != new_lr:
            logging.info(
                f"Epoch {epoch+1}/{AES_NUM_EPOCHS}, Aesthetic Phase, Learning Rate Changed from {old_lr} to {new_lr}"
            )
        val_losses_aesthetic.append(val_loss_aesthetic)
        train_losses_aesthetic.append(train_loss_aesthetic)

        writer.add_scalar("Validation Loss aesthetic", val_loss_aesthetic, epoch)
        writer.add_scalar("Training Loss aesthetic", train_loss_aesthetic, epoch)

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

    # At the end of aesthetic phase, save the final model
    final_model_path = save_training_checkpoint(
        model,
        AES_NUM_EPOCHS,
        PATH_MODEL_RESULTS,
        "aestheticNet-final",
        tic,
        optimizer_aesthetic,
        scheduler_aesthetic,
        scaler_aesthetic,
    )
    logging.info(f"Final model saved at {final_model_path}")

    plt.figure(figsize=(10, 5))
    plt.plot(val_losses_aesthetic, label="Aesthetic Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Validation Losses Over Epochs")
    plt.legend()
    plt.savefig(os.path.join(saving_dir, "val_losses_aesthetic" + tic + ".png"))

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses_aesthetic, label="Aesthetic Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Losses Over Epochs")
    plt.legend()
    plt.savefig(os.path.join(saving_dir, "train_losses_aesthetic" + tic + ".png"))

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses_aesthetic, label="Aesthetic Training Loss")
    plt.plot(val_losses_aesthetic, label="Aesthetic Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Losses Over Epochs")
    plt.legend()
    plt.savefig(os.path.join(saving_dir, "train_val_losses_aesthetic" + tic + ".png"))
    writer.close()

    # save the raw data of losses to a csv file
    pretext_losses = {
        "train_losses": train_losses_pretext,
        "val_losses": val_losses_pretext,
    }
    pretext_losses_df = pd.DataFrame(pretext_losses)
    pretext_losses_df.to_csv(
        os.path.join(saving_dir, "pretext_losses" + tic + ".csv"), index=False
    )

    aesthetic_losses = {
        "train_losses": train_losses_aesthetic,
        "val_losses": val_losses_aesthetic,
    }

    aesthetic_losses_df = pd.DataFrame(aesthetic_losses)
    aesthetic_losses_df.to_csv(
        os.path.join(saving_dir, "aesthetic_losses" + tic + ".csv"), index=False
    )


if __name__ == "__main__":
    main()
