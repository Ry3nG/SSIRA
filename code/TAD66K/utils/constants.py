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

#result constants
PATH_LOGS = "/home/zerui/SSIRA/code/TAD66K/results/logs/"
PATH_MODELS = "/home/zerui/SSIRA/code/TAD66K/results/models/"
PATH_PLOTS = "/home/zerui/SSIRA/code/TAD66K/results/plots/"

# training constants
BATCH_SIZE = 128
NUM_EPOCHS = 1000
LEARNING_RATE = 0.001
NUM_WORKERS = 8
TRAIN_VAL_SPLIT_RATIO = 0.9

# Learning rate scheduler constants
LR_DECAY_FACTOR = 0.1
LR_DECAY_PATIENCE = 5
LR_DECAY_THRESHOLD = 1*1e-5

# Early stopping constants
EARLY_STOPPING_PATIENCE = 10
