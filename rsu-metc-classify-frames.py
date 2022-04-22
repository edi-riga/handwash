#!/usr/bin/env python3

from classify_dataset import evaluate
from dataset_utilities import get_datasets

# make sure to provide correct paths to the folders on your machine
data_dir = 'dataset-metc/RSU_METC_dataset_preprocessed/frames/trainval'
test_data_dir = 'dataset-metc/RSU_METC_dataset_preprocessed/frames/test'

train_ds, val_ds, test_ds, weights_dict = get_datasets(data_dir, test_data_dir)

evaluate("rsu-metc-single-frame", train_ds, val_ds, test_ds, weights_dict)
