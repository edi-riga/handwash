#!/usr/bin/env python3

#
# This script trains a simple network for washing / non-washing classification
#

import matplotlib.pyplot as plt
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import pathlib
import tensorflow as tf

#from sklearn.utils import class_weight

from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# make sure to provide correct paths to the folders on your machine
data_dir = pathlib.Path('./trainval_washing')
test_data_dir = pathlib.Path('./test_washing')

# Get the total number of images for training and validation
trainval_img_count = 0
for subdir, dirs, files in os.walk(data_dir):
    for image_file in files:
        if image_file.endswith("jpg"):
            trainval_img_count += 1

print('Number of trainval images: ', trainval_img_count)

# Get the total number of images in test dataset
test_img_count = 0
for subdir, dirs, files in os.walk(test_data_dir):
    for image_file in files:
        if image_file.endswith("jpg"):
            test_img_count += 1

print('Number of images in the test dataset: ', test_img_count)

# define parameters for the dataset loader. 
# Adjust batch size according to the memory volume of your GPU; 256 works well on NVIDIA RTX 3090 with 24 GB VRAM
#batch_size = 256
batch_size = 16
img_width = 320
img_height = 240
IMG_SIZE = (img_height, img_width)
IMG_SHAPE = IMG_SIZE + (3,)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=IMG_SIZE,
  label_mode='categorical',
  batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=IMG_SIZE,
  label_mode='categorical',
  batch_size=batch_size)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
  test_data_dir,
  seed=123,
  image_size=IMG_SIZE,
  label_mode='categorical',
  batch_size=batch_size)

# check the names of the classes
class_names = train_ds.class_names
print(class_names)

# As the dataset is imbalanced, is is necessary to get weights for each class

# get the number of trainval images for each class
images_by_labels = []
for i in range(len(class_names)): 
    for subdir, dirs, files in os.walk(os.path.join(data_dir,str(i))):
        n_of_files = 0
        for image_file in files:
            if image_file.endswith("jpg"):
                n_of_files += 1
    images_by_labels.append(n_of_files)

# calculate weights
images_by_labels = np.array(images_by_labels)
avg = np.average(images_by_labels)
weights = avg / images_by_labels

# create dictionary with weights as required for keras fit() function
weights_dict = {}
for item in range(len(weights)):
    weights_dict[int(class_names[item])] = weights[item]

# to improve performance, use buffered prefetching to load images 
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_dataset = train_ds.prefetch(buffer_size=AUTOTUNE)
validation_dataset = val_ds.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_ds.prefetch(buffer_size=AUTOTUNE)



# data augmentation 
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
  tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])



# rescale pixel values
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset= -1)


base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

# freeze the convolutional base
base_model.trainable = False

# Build the model
inputs = tf.keras.Input(shape=IMG_SHAPE)
x = inputs
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = tf.keras.layers.Flatten()(x)
outputs = tf.keras.layers.Dense(len(class_names), activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)

print("compiling the model...")
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.008),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

#number_of_epochs = 30
number_of_epochs = 10

# callbacks to implement early stopping and saving the model
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
mc = ModelCheckpoint(monitor='val_accuracy', mode='max', 
                     verbose=1, save_freq='epoch', 
                     filepath='MobileNetV2_Handwashing_dataset.{epoch:02d}-{val_accuracy:.2f}.h5')

print("fitting the model...")
history = model.fit(train_ds,
                    epochs=number_of_epochs,
                    validation_data=val_ds,
                    class_weight=weights_dict,
                    callbacks=[es, mc])

# visualise accuracy
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(train_acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')
plt.savefig("accuracy.pdf", format="pdf")
