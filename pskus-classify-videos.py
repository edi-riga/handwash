#!/usr/bin/env python3

from classify_dataset import evaluate, get_time_distributed_model, IMG_SIZE, N_CLASSES, num_frames, batch_size
from dataset_utilities import get_weights_dict
from generator_timedistributed import timedistributed_dataset_from_directory

# make sure to provide correct paths to the folders on your machine
data_dir = 'dataset-pskus/PSKUS_dataset_preprocessed/frames/trainval'
test_data_dir = 'dataset-pskus/PSKUS_dataset_preprocessed/frames/test'

FPS = 16

CLASS_NAMES = [str(i) for i in range(N_CLASSES)]

train_ds = timedistributed_dataset_from_directory(
    data_dir,
    num_frames=num_frames,
    frame_step=FPS // num_frames,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    shuffle=True,
    label_mode='categorical',
    crop_to_aspect_ratio=False,
    batch_size=batch_size)

val_ds = timedistributed_dataset_from_directory(
    data_dir,
    num_frames=num_frames,
    frame_step=FPS // num_frames,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    shuffle=True,
    label_mode='categorical',
    crop_to_aspect_ratio=False,
    batch_size=batch_size)

test_ds = timedistributed_dataset_from_directory(
    test_data_dir,
    num_frames=num_frames,
    frame_step=FPS // num_frames,
    seed=123,
    image_size=IMG_SIZE,
    shuffle=False,
    label_mode='categorical',
    crop_to_aspect_ratio=False,
    batch_size=batch_size)

weights_dict = get_weights_dict(data_dir, CLASS_NAMES)

model = get_time_distributed_model()

evaluate("pskus-videos", train_ds, val_ds, test_ds, weights_dict, model=model)
