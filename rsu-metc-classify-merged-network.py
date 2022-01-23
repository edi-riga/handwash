#!/usr/bin/env python3

#
# This work is inspired by this NN architecture:
# https://github.com/Realtime-Action-Recognition/Realtime-Action-Recognition/blob/master/three_stream_model.py
#

from classify_dataset import evaluate, get_merged_model, IMG_SIZE, N_CLASSES, batch_size
from dataset_utilities import get_weights_dict
from generator_rgb_with_of import merged_dataset_from_directories
import os

import tensorflow as tf

# make sure to provide correct paths to the folders on your machine
rgb_dir = '/data/handwash/RSU_METC_dataset_preprocessed/frames/trainval'
of_dir = '/data/handwash/RSU_METC_dataset_preprocessed/of/trainval'

test_rgb_dir = '/data/handwash/RSU_METC_dataset_preprocessed/frames/test'
test_of_dir = '/data/handwash/RSU_METC_dataset_preprocessed/of/test'

CLASS_NAMES = [str(i) for i in range(N_CLASSES)]

train_ds = merged_dataset_from_directories(
    rgb_dir,
    of_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    shuffle=True,
    label_mode='categorical',
    crop_to_aspect_ratio=False,
    batch_size=batch_size)

val_ds = merged_dataset_from_directories(
    rgb_dir,
    of_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    shuffle=True,
    label_mode='categorical',
    crop_to_aspect_ratio=False,
    batch_size=batch_size)

test_ds = merged_dataset_from_directories(
    test_rgb_dir,
    test_of_dir,
    seed=123,
    image_size=IMG_SIZE,
    shuffle=False,
    label_mode='categorical',
    crop_to_aspect_ratio=False,
    batch_size=batch_size)

# to improve performance, use buffered prefetching to load images
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

weights_dict = get_weights_dict(rgb_dir, CLASS_NAMES)

model = get_merged_model()

evaluate("rsu-metc-merged", train_ds, val_ds, test_ds, weights_dict, model=model)
