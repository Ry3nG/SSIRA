"""
This file contains the constants used in the project.
"""
import torch
# dataset constants
PATH_DATASET_TAD66K = "/home/zerui/SSIRA/dataset/TAD66K/"
PATH_LABEL_MERGE_TAD66K_TEST = "/home/zerui/SSIRA/dataset/TAD66K/labels/merge/test.csv"
PATH_LABEL_MERGE_TAD66K_TRAIN = (
    "/home/zerui/SSIRA/dataset/TAD66K/labels/merge/train.csv"
)

PATH_AVA_TXT = "/home/zerui/SSIRA/dataset/AVA/AVA_dataset/AVA.txt"
PATH_AVA_IMAGE = "/home/zerui/SSIRA/dataset/AVA/AVA_dataset/image"


# training constants
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
NUM_WORKERS = 8
TRAIN_VAL_SPLIT_RATIO = 0.9

# logging constants
PATH_LOGS = "/home/zerui/SSIRA/code/AestheticNet/logs"
PATH_MODEL_RESULTS = "/home/zerui/SSIRA/code/AestheticNet/results/Models"
