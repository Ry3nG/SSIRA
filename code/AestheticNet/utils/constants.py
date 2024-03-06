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


# PATH AVA
PATH_AVA_TXT = "/home/zerui/SSIRA/dataset/AVA/AVA_dataset/AVA.txt"
PATH_AVA_IMAGE = "/home/zerui/SSIRA/dataset/AVA/AVA_dataset/image"
PATH_AVA_TEST_IDS = (
    "/home/zerui/SSIRA/dataset/AVA/AVA_dataset/aesthetics_image_lists/generic_test.jpgl"
)

PATH_AVA_GENERIC_TRAIN_IDS = "/home/zerui/SSIRA/dataset/AVA/AVA_dataset/aesthetics_image_lists/generic_ls_train_clean.jpgl"
PATH_AVA_GENERIC_TEST_IDS = "/home/zerui/SSIRA/dataset/AVA/AVA_dataset/aesthetics_image_lists/generic_test_clean.jpgl"

# PATH AVA SPLITS
PATH_AVA_HLAGCN = "/home/zerui/SSIRA/dataset/AVA_Split/train_hlagcn.csv"
PATH_AVA_MLSP = " /home/zerui/SSIRA/dataset/AVA_Split/train_mlsp.csv"

# training constants
BATCH_SIZE = 64

PRETEXT_NUM_EPOCHS = 100
AES_NUM_EPOCHS = 100

LEARNING_RATE = 1e-7
NUM_WORKERS = 32
TRAIN_VAL_SPLIT_RATIO = 0.95

SAVE_FREQ = 5

# logging constants
PATH_LOGS = "/home/zerui/SSIRA/code/AestheticNet/logs"
PATH_MODEL_RESULTS = "/home/zerui/SSIRA/code/AestheticNet/results/Models"
PATH_PLOTS = "/home/zerui/SSIRA/code/AestheticNet/results/Plots"

# learning rate scheduler constants
LR_PATIENCE = 10
LR_FACTOR = 0.1
LR_MODE = "min"
LR_VERBOSE = True
LR_MIN = 1e-8
