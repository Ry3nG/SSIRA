
import pandas as pd
import numpy as np
import os
import random

# dataset constants
PATH_DATASET_TAD66K = "/home/zerui/SSIRA/dataset/TAD66K/"
PATH_LABEL_MERGE_TAD66K_TEST = "/home/zerui/SSIRA/dataset/TAD66K/labels/merge/test.csv"
PATH_LABEL_MERGE_TAD66K_TRAIN = (
    "/home/zerui/SSIRA/dataset/TAD66K/labels/merge/train.csv"
)

PATH_AVA_TXT = "/home/zerui/SSIRA/dataset/AVA/AVA_dataset/AVA.txt"
PATH_AVA_IMAGE = "/home/zerui/SSIRA/dataset/AVA/AVA_dataset/image"
PATH_AVA_TEST_IDS = "/home/zerui/SSIRA/dataset/AVA/AVA_dataset/aesthetics_image_lists/generic_test.jpgl"

PATH_AVA_GENERIC_TRAIN_IDS = "/home/zerui/SSIRA/dataset/AVA/AVA_dataset/aesthetics_image_lists/generic_ls_train_clean.jpgl"
PATH_AVA_GENERIC_TEST_IDS = "/home/zerui/SSIRA/dataset/AVA/AVA_dataset/aesthetics_image_lists/generic_test_clean.jpgl"


# read the csv file
original_csv = pd.read_csv(PATH_AVA_GENERIC_TRAIN_IDS)
print(original_csv.head())

# save the top 1000 images as a new csv file
new_csv = original_csv.head(1000)
new_csv.to_csv('top_1000.csv', index=False)


