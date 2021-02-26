#!/usr/bin/env python

#
# This script takes input folders with video files and annotations
# and converts them to output folders.
# There are two main outputfolder:
# * "test" for testing dataset
# * "trainval" for training and validation data
# Each of the main output folders contains subfolders named "0", "1", ... "7".
# The subfolders have jpeg images of the individual frames.
#

import cv2
import os
import json
import random

# Change these duire
output_folder = './'
input_folder = 'handwash_data/'

# movement 0 is large part of the dataset; use only 20% of all frames
MOVEMENT_0_PROPORTION = 0.2
# use 10% for the test dataset, rest for training and validation
TEST_SET_PROPORTION = 0.1

# the movement codes are from 0 to 7
TOTAL_MOVEMENTS = 8

TOTAL_ANNOTATORS = 8

def majority_vote(lst):
    """Returns the element present in majority of the list, or -1 otherwise
    """
    counts = [0] * TOTAL_MOVEMENTS
    for el in lst:
        counts[int(el)] += 1
    best = 0
    for i in range(1, TOTAL_MOVEMENTS):
        if counts[best] < counts[i]:
            best = i
    majority = (len(lst) + 2) // 2
    if counts[best] < majority:
        return -1
    return best

def find_frame_labels(fullpath):
    """Returns `is_washing` status and movement codes for each frame
    """
    filename = os.path.basename(fullpath)
    annotators_dir = os.path.join(os.path.dirname(os.path.dirname(fullpath)), "Annotations")

    annotations = []

    for a in range(1, TOTAL_ANNOTATORS + 1):
        annotator_dir = os.path.join(annotators_dir, "Annotator" + str(a))
        json_filename = os.path.join(annotator_dir, filename.split(".")[0] + ".json")

        if os.access(json_filename, os.R_OK):
            with open(json_filename, "r") as f:
                try:
                  data = json.load(f)
                except:
                  print("failed to load {}".format(json_filename))
                  continue
                a_annotations = [(data['labels'][i]['is_washing'], data['labels'][i]['code']) for i in range(len(data['labels']))]
                annotations.append(a_annotations)

    num_annotators = len(annotations)
    num_frames = len(annotations[0])
    is_washing = []
    codes = []
    for frame_num in range(num_frames):
        frame_annotations = [annotations[a][frame_num] for a in range(num_annotators)]
        frame_is_washing = any([frame_annotations[a][0] for a in range(num_annotators)])
        frame_codes = [frame_annotations[a][1] for a in range(num_annotators)]
        # treat movement 7 as movement 0
        frame_codes = [0 if code == 7 else code for code in frame_codes]

        is_washing.append(frame_is_washing)
        if frame_is_washing:
            codes.append(majority_vote(frame_codes))
        else:
            codes.append(-1)

    return is_washing, codes

def get_frames(folder):
    """Get and save frames from matching videos
    """
    # to get the summary afterwards
    N_of_videofiles = 0
    N_of_matching_files = 0
    N_of_matching_frames = 0 # frames that were labelled identically
    N_of_frames_total = 0  # frames labelled as 'is washing'
    used_files = 0

    print('Processing folder: ' + folder + ' ...')

    file_locations = {}
    for subdir, dirs, files in os.walk(os.path.join(input_folder, folder)):
        for videofile in files:
            if videofile.endswith(".mp4"):
                if videofile not in file_locations:
                   file_locations[videofile] = [subdir]
                else:
                   print("file {} also in {}".format(videofile, ",".join(file_locations[videofile])))
                   file_locations[videofile].append(subdir)
                N_of_videofiles += 1

                fullpath = os.path.join(input_folder, subdir, videofile)
                used_files +=1
                is_washing, codes = find_frame_labels(fullpath)

                vidcap = cv2.VideoCapture(fullpath)
                is_success, image = vidcap.read()
                frame_number = 0

                r = random.random()
                if r < TEST_SET_PROPORTION:
                    traintest = "test"
                else:
                    traintest = "trainval"

                while is_success:
                    if is_washing[frame_number]:
                        N_of_frames_total += 1
                        if codes[frame_number] >= 0:
                            N_of_matching_frames += 1

                            if codes[frame_number] == 0 and random.random() > MOVEMENT_0_PROPORTION:
                                # skip some movement 0 frames
                                is_success, image = vidcap.read()
                                frame_number += 1
                                continue

                            subfolder = str(codes[frame_number])
                            filename = 'frame{}_file_{}.jpg'.format(frame_number, os.path.splitext(videofile)[0])

                            # the name of the file storing the frames includes the frame number and the videofile name
                            save_path_and_name = os.path.join(output_folder, traintest, subfolder, filename)
                            cv2.imwrite(save_path_and_name, image)
                    is_success, image = vidcap.read()
                    frame_number += 1


    N_of_nonmatching_files = N_of_videofiles - N_of_matching_files
    N_of_nonmatching_frames = N_of_frames_total - N_of_matching_frames

    print('Number of processed videofiles: ', N_of_videofiles)
    print('Number of saved frames: ', N_of_matching_frames)
    print('Number of frames marked as IS WASHING that did not match: ', N_of_nonmatching_frames)
    print('Percentage of matching frames: ', 1.0 * N_of_matching_frames / (N_of_matching_frames + N_of_nonmatching_frames) * 100)
    print('')


def main():
    random.seed(0)
    try:
       os.mkdir(output_folder)
    except:
       pass
    try:
       os.mkdir(os.path.join(output_folder, "test"))
    except:
       pass
    try:
       os.mkdir(os.path.join(output_folder, "trainval"))
    except:
       pass
    for movement in range(TOTAL_MOVEMENTS):
       try:
          os.mkdir(os.path.join(output_folder, "test", str(movement)))
       except:
          pass
       try:
          os.mkdir(os.path.join(output_folder, "trainval", str(movement)))
       except:
          pass

    list_of_folders = [d for d in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, d))]
    for folder in sorted(list_of_folders):
        get_frames(folder)

# ----------------------------------------------
if __name__ == "__main__":
    main()
