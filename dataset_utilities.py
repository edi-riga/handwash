#
# This script loads and separates the target dataset in train, validation, and test components.
# It also has a function for getting a weights dictionary to the used in the NN training,
# in case the dataset is not balanced.
#

import os
import tensorflow as tf
import numpy as np

import classify_dataset

def get_datasets(data_dir, test_data_dir, batch_size=None):
    if batch_size is None:
        batch_size = classify_dataset.batch_size

    IMG_SIZE = classify_dataset.IMG_SIZE

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=classify_dataset.IMG_SIZE,
        label_mode='categorical',
        crop_to_aspect_ratio=False,
        batch_size=batch_size)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMG_SIZE,
        label_mode='categorical',
        crop_to_aspect_ratio=False,
        batch_size=batch_size)

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_data_dir,
        seed=123,
        image_size=IMG_SIZE,
        label_mode='categorical',
        crop_to_aspect_ratio=False,
        batch_size=batch_size)

    weights_dict = get_weights_dict(data_dir, train_ds.class_names)

    # to improve performance, use buffered prefetching to load images
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds, weights_dict


def get_weights_dict(data_dir, class_names):
    # As the dataset is imbalanced, is is necessary to get weights for each class
    # get the number of trainval images for each class
    images_by_labels = []
    for i in range(len(class_names)):
        for subdir, dirs, files in os.walk(os.path.join(data_dir, str(i))):
            n_of_files = sum([f.endswith(".jpg") or f.endswith(".mp4") for f in files])
            images_by_labels.append(n_of_files)

    # calculate weights
    images_by_labels = np.array(images_by_labels)
    avg = np.average(images_by_labels)
    weights = avg / images_by_labels

    # create dictionary with weights as required for keras fit() function
    weights_dict = {}
    for item in range(len(weights)):
        weights_dict[int(class_names[item])] = weights[item]
    print("weights_dict=", weights_dict)

    return weights_dict
