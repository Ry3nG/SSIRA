from constants import *
import argparse

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
        "--lr_pretext",
        type=float,
        default=LEARNING_RATE,
        help="Learning rate for pretext training",
    )
    parser.add_argument(
        "--lr_aesthetic",
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
        help="Path to a saved checkpoint (default: None)",
    )

    return parser.parse_args()
