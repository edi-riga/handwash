#!/bin/bash

#
# This script get the data from Zenodo, extracts it, and preprocesses it
# to make it suitable for the machine learning scripts.
#

mkdir PSKUS_dataset
cd PSKUS_dataset

echo "Getting the dataset..."
wget https://zenodo.org/record/4537209/files/DataSet1.zip?download=1 --output-document=DataSet1.zip
wget https://zenodo.org/record/4537209/files/DataSet2.zip?download=1 --output-document=DataSet2.zip
wget https://zenodo.org/record/4537209/files/DataSet3.zip?download=1 --output-document=DataSet3.zip
wget https://zenodo.org/record/4537209/files/DataSet4.zip?download=1 --output-document=DataSet4.zip
wget https://zenodo.org/record/4537209/files/DataSet5.zip?download=1 --output-document=DataSet5.zip
wget https://zenodo.org/record/4537209/files/DataSet6.zip?download=1 --output-document=DataSet6.zip
wget https://zenodo.org/record/4537209/files/DataSet7.zip?download=1 --output-document=DataSet7.zip
wget https://zenodo.org/record/4537209/files/DataSet8.zip?download=1 --output-document=DataSet8.zip
wget https://zenodo.org/record/4537209/files/DataSet9.zip?download=1 --output-document=DataSet9.zip
wget https://zenodo.org/record/4537209/files/DataSet10.zip?download=1 --output-document=DataSet10.zip
wget https://zenodo.org/record/4537209/files/DataSet11.zip?download=1 --output-document=DataSet11.zip

echo "Extracting the dataset..."
unzip DataSet1.zip
unzip DataSet2.zip
unzip DataSet3.zip
unzip DataSet4.zip
unzip DataSet5.zip
unzip DataSet6.zip
unzip DataSet7.zip
unzip DataSet8.zip
unzip DataSet9.zip
unzip DataSet10.zip
unzip DataSet11.zip
cd -

cp statistics-with-locations.csv PSKUS_dataset/statistics-with-locations.csv

echo "Preprocessing..."
./separate-frames.py
