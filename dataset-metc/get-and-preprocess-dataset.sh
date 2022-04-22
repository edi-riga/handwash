#!/bin/bash

#
# This script get the data from Zenodo, extracts it, and preprocesses it
# to make it suitable for the machine learning scripts.
#
# Requirements:
# - cv2 Python module installed
# - ffmpeg installed and executable from command line
#

mkdir RSU_METC_dataset
cd RSU_METC_dataset

echo "Getting the dataset..."
wget https://zenodo.org/record/5808789/files/Interface_number_1.zip?download=1 --output-document=Interface_number_1.zip
wget https://zenodo.org/record/5808789/files/Interface_number_2.zip?download=1 --output-document=Interface_number_2.zip
wget https://zenodo.org/record/5808789/files/Interface_number_3.zip?download=1 --output-document=Interface_number_3.zip

echo "Extracting the dataset..."
unzip Interface_number_1.zip
unzip Interface_number_2.zip
unzip Interface_number_3.zip
cd -

echo "Preprocessing..."
./separate-frames.py
