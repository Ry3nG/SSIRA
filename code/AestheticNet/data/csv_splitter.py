
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




import csv

input_file = PATH_LABEL_MERGE_TAD66K_TRAIN
output_file = '/home/zerui/SSIRA/dataset/TAD66K/labels/merge/train_first_1000.csv'

with open(input_file, mode='r', newline='', encoding='utf-8') as infile, \
     open(output_file, mode='w', newline='', encoding='utf-8') as outfile:

    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    # Write header
    writer.writerow(next(reader))

    # Write first 1000 rows
    for i, row in enumerate(reader):
        if i < 1000:
            writer.writerow(row)
        else:
            break

print(f"First 1000 rows saved to '{output_file}'")
