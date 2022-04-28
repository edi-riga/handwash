#!/bin/bash

#
# This script get the data from GitHub, extracts it, and preprocesses it
# to make it suitable for the machine learning scripts.
#
# Requirements:
# - cv2 Python module installed
#

echo "Getting the dataset..."
wget https://github.com/atiselsts/data/raw/master/kaggle-dataset-6classes.tar

echo "Extracting the dataset..."
tar -xf kaggle-dataset-6classes.tar

echo "Preprocessing..."
./separate-frames.py
