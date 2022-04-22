# Hand washing movement classification

This repository contains code for hand washing movement classification based on the six key movements defined by the World Health Organization (WHO).

The work has been done as part of the projects:

* "Integration of reliable technologies for protection against Covid-19 in healthcare and high-risk areas", project No. VPP-COVID-2020/1-0004.
* Latvian Council of Science project: “Automated hand washing quality control and quality evaluation system with real-time feedback”, No: lzp-2020/2-0309


# Quick start guide

You are going to need a Linux OS with TensorFlow, Keras, OpenCV, and NumPy installed to run the scripts, and a modern GPU to train the neural networks.

To start working with the data, follow these steps:

0. Install prerequisites:  Python modules and the `ffmpeg` application. Python modules can be installed with:

       sudo pip install -r requirements.txt 

1. Download and extract the required datasets. For this, you can use the scripts:

       dataset-kaggle/get-and-preprocess-dataset.sh
       dataset-pskus/get-and-preprocess-dataset.sh
       dataset-metc/get-and-preprocess-dataset.sh

2. Preprocess the datasets by extracting frames from the video data, separating them in classes, and further separating them in test/trainval subsets. If you used the scripts in the previous point to download the data, this was already done automatically. Otherwise use the `separate-frames.py` scripts.

3. (Optional.) Calculate optical flow on the datasets.

4. Train the neural network classifiers on the data.


# Detailed instructions

## 1. Datasets

The code supports the following publicly available datasets:

* Real-world hospital data sets, collected at the Pauls Stradins Clinical University Hospital (abbreviated as PSKUS or PSCUH): https://zenodo.org/record/4537209
* Lab-based dataset collected at the Medical Education Technology Center (METC) of Riga Stradins University: https://zenodo.org/record/5808789
* The publicly available subset of the [Kaggle Hand Wash Dataset](https://www.kaggle.com/realtimear/hand-wash-dataset) with video files resorted in 7 classes to match the class structure of the other datasets: https://github.com/atiselsts/data/raw/master/kaggle-dataset-6classes.tar

Follow the links above to download the datasets.

Once you have downloaded them, extract the archives, and organize them so that each dataset is located in a single folder.

* The PSKUS dataset should have files "summary.csv", "statistics.csv" and folders named "DataSet1", "DataSet2" etc. in the top-level directory. Also copy the file `statistics-with-locations.csv` from this repository to the PSKUS dataset folder. This will ensure that videos from the same camera location will be mixed in the test and trainval folders, making the neural network generalization requirements more challenging.
* The METC dataset should have files "summary.csv", "statistics.csv" and folders named "Interface_number_1", "Interface_number_2", "Interface_number_3" in the top-level directory.
* For the Kaggle dataset, we provide an intermediate version named `kaggle-dataset-6classes` at [GitHub](https://github.com/atiselsts/data/raw/master/kaggle-dataset-6classes.tar). We have re-sort the Kaggle video files so that they are all put in just 7 classes. This is because the other datasets do not distinguish between right and left hand washing. The wrist-washing videos are placed the class 0 ("Other") folder.


## 2. Preprocessing the data

The folders `dataset-kaggle`, `dataset-metc` and `dataset-pskus` have `separate-frames.py` scripts in them. Fix the paths in these scripts to match the locations of your datasets, and run the script to separate the video datasets in frames, video snippets, as well as separate these frames and shorter videos in `trainval` and `test` folders.


## 3. Calculate optical flow

This step is optional and only required if the merged neural network architecture is used.

Run the `calculate-optical-flow.py` script, and pass the target dataset's folder names as the command line argument to this script.


## 4. Train the classifiers.


For each dataset, three training scripts are provided. (Replace `xxx` with the name of the dataset.)

* `xxx-classify-frames.py` — the baseline single-frame architecture
* `xxx-classify-videos.py` — the time-distributed network architecture with GRU elements
* `xxx-classify-merged-network.py` — the two-stream network architecture with both RGB and Optical Flow inputs.

These scripts rely on a number of environmental variables to set training parameters for the neural networks.
Unless you're fine with the default values, you should set these parameters before running the scripts, e.g. with:

     export HANDWASH_NUM_EPOCHS=40

The variables are:

* `HANDWASH_NN` — the base model name, default "MobileNetV2"
* `HANDWASH_NUM_LAYERS` — the number of trainable layers (of the base model), default 0
* `HANDWASH_NUM_EPOCHS` — the max number of epochs. Early termination is still possible! Default: 20.
* `HANDWASH_NUM_FRAMES` — how many frames to concatenate as input to the TimeDistributed network? Default: 5.
* `HANDWASH_SUFFIX` — user-defined additional suffix of the result files of the experiment. Default: empty string.
* `HANDWASH_PRETRAINED_MODEL` — the path to a pretrained model. Used for transfer-learning experiments. Default: empty string (pretrained model not used, the base model with ImageNet weights is loaded instead.)
* `HANDWASH_EXTRA_LAYERS` — the number of additional dense layers to add to the network before the top layer. Default: 0.


# References

For more detailed information, see the following articles:

* A. Elsts, M. Ivanovs, R. Kadikis, O. Sabelnikovs. CNN for Hand Washing Movement Classification: What Matters More — the Approach or the Dataset? In Proceedings of International Conference on Image Processing Theory, Tools and Applications (IPTA) 2022.
* M. Lulla, A. Rutkovskis, A. Slavinska, A. Vilde, A. Gromova, M. Ivanovs, A. Skadins, R. Kadikis and A. Elsts. Hand Washing Video Dataset Annotated According to the World Health Organization’s Handwashing Guidelines. Data, 6(4), p.38. [[HTML]](https://www.mdpi.com/2306-5729/6/4/38/htm)
* M. Ivanovs, R. Kadikis, M. Lulla, A. Rutkovskis, A. Elsts, Automated Quality Assessment of Hand Washing Using Deep Learning, arXiv preprint, 2020. [[PDF]](https://arxiv.org/pdf/2011.11383.pdf)


# Contacts

The main author of this code can be reached via email for questions: atis.elsts@edi.lv or atis.elsts@gmail.com
