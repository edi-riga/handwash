#!/usr/bin/env python3

#
# This script takes input folders with video files and annotations
# and converts them to output folders.
# There are two output folders for movement classification:
# * "test" for testing dataset;
# * "trainval" for training and validation data.
# Each of these output folders contains subfolders named "0", "1", ... "7".
# There also are two output folders for the is/isn't washing classification:
# * "test_washing" for testing dataset
# * "trainval_washing" for training and validation data
# Each of them contains "0" and "1" subfolders.
# The subfolders have jpeg images of the individual frames.
#

import cv2
import os
import json
import random

# Change these directories to your own locations
output_folder = './'
input_folder = '../merged_handwash_data'

# movement 0 is large part of the dataset; use only 20% of all frames
MOVEMENT_0_PROPORTION = 0.2
# use 10% for the test dataset, rest for training and validation
TEST_SET_PROPORTION = 0.1

# the movement codes are from 0 to 7
TOTAL_MOVEMENTS = 8

# the Annotator directories go from Annotator1 to Annotator8
TOTAL_ANNOTATORS = 8

# most of the dataset describes washing; reduce to improve class balance in the output
IS_WASHING_PROPORTION = 1/1.8

# reduce the number of frames taken to improve training speed for is washing classification
FRAME_POOLING = 10

# if supplementary info is present, and this parameter set to True,
# then for the is/isn't washing classification only images where hands have been detected are used.
ONLY_NONWASHING_WITH_HANDS = True

if ONLY_NONWASHING_WITH_HANDS:
    # reduce the frame pooling
    FRAME_POOLING = 4
    # reduce the proportion
    IS_WASHING_PROPORTION = 0.35

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


def mk(directory):
    folders = directory.split(os.path.sep)
    for i in range(len(folders)):
        so_far = str(os.path.sep).join(folders[:i+1])
        try:
            os.mkdir(so_far)
        except:
            pass


def find_frame_labels(fullpath):
    """Returns `is_washing` status and movement codes for each frame
    """
    filename = os.path.basename(fullpath)
    annotators_dir = os.path.join(os.path.dirname(os.path.dirname(fullpath)), "Annotations")

    annotations = []

    # Load the supplementary info, if present.
    # This info is not part of the public dataset;
    # it is hand presence information, generated using the Mediapipe hand tracking neural network
    supplementary_dir = os.path.join(os.path.dirname(os.path.dirname(fullpath)), "Supplementary")
    supplementary_filename = os.path.join(supplementary_dir, "hands-" + filename + ".txt")
    frames_with_hands = []
    if os.access(supplementary_filename, os.R_OK):
        with open(supplementary_filename, "r") as f:
            for line in f.readlines():
                try:
                    i = int(line)
                except Exception as ex:
                    break
                frames_with_hands.append(i)

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
        frame_is_washing_any = any([frame_annotations[a][0] for a in range(num_annotators)])
        frame_is_washing_all = all([frame_annotations[a][0] for a in range(num_annotators)])
        frame_codes = [frame_annotations[a][1] for a in range(num_annotators)]
        # treat movement 7 as movement 0
        frame_codes = [0 if code == 7 else code for code in frame_codes]

        if frame_is_washing_all:
            frame_is_washing = 2
        elif frame_is_washing_any:
            frame_is_washing = 1
        else:
            frame_is_washing = 0

        is_washing.append(frame_is_washing)
        if frame_is_washing:
            codes.append(majority_vote(frame_codes))
        else:
            codes.append(-1)

    if len(frames_with_hands):
        if len(frames_with_hands) != len(is_washing):
            if len(frames_with_hands) > len(is_washing):
                print("Incorrect dimensions of the supplementary information: {} vs {}".format(len(frames_with_hands), len(is_washing)))
                frames_with_hands = []
            else:
                # pad with zeroes (no hands detected)
                pad_len = len(is_washing) - len(frames_with_hands)
                frames_with_hands += [0] * pad_len

    return is_washing, codes, frames_with_hands, num_annotators


def get_frames(folder):
    """Get and save frames from matching videos
    """
    # to get the summary afterwards
    N_of_videofiles = 0
    N_of_frames_considered = 0
    N_of_frames_washing = 0  # frames labelled as 'is washing'
    N_of_matching_frames = 0 # frames that had a clear majority label
    N_of_saved_movement_classification_frames = 0
    N_of_saved_is_washing_classification_frames = 0
    N_of_saved_no_washing_with_hands_frames = 0
    N_of_saved_no_washing_no_hands_frames = 0
    used_files = 0

    print('Processing folder: ' + folder + ' ...')

    for subdir, dirs, files in os.walk(os.path.join(input_folder, folder)):
        for videofile in files:
            if videofile.endswith(".mp4"):
                N_of_videofiles += 1

                fullpath = os.path.join(subdir, videofile)
                used_files +=1
                is_washing, codes, frames_with_hands, num_annotators = find_frame_labels(fullpath)

                vidcap = cv2.VideoCapture(fullpath)
                is_success, image = vidcap.read()
                frame_number = 0

                r = random.random()
                if r < TEST_SET_PROPORTION:
                    traintest = "test"
                else:
                    traintest = "trainval"

                while is_success:
                    N_of_frames_considered += 1

                    # for multi-annotator files with agreement,
                    # potentially save the frame in the washing/not washing set
                    if num_annotators > 1 and is_washing[frame_number] in [0, 2]:
                        if frame_number % FRAME_POOLING != 0:
                            is_ok = False
                        else:
                            if len(frames_with_hands):
                                if bool(frames_with_hands[frame_number]) != bool(is_washing[frame_number]):
                                    if is_washing[frame_number]:
                                        #print("warning: hands not detected in frame, but is washing!")
                                        is_ok = False
                                    else:
                                        # keep all frames with hands, but no washing
                                        #print("nice: hands in frame, but not washing!")
                                        is_ok = True
                                        N_of_saved_no_washing_with_hands_frames += 1
                                else:
                                    if ONLY_NONWASHING_WITH_HANDS:
                                        if is_washing[frame_number]:
                                            # is washing and has hands
                                            is_ok = random.random() < IS_WASHING_PROPORTION
                                        else:
                                            # is not washing and has no hands
                                            is_ok = False
                                    else:
                                        # both equal; skip most of such frames
                                        is_ok = random.random() < IS_WASHING_PROPORTION
                                        if is_washing[frame_number]:
                                            # apply the filter double
                                            if is_ok:
                                                is_ok = random.random() < IS_WASHING_PROPORTION
                                        else:
                                            # a single filter, but remember that this is an ordinary no-washing frame
                                            if is_ok:
                                                N_of_saved_no_washing_no_hands_frames += 1

                            # no supplementary info
                            else:
                                # skip most of the is_washing frames
                                is_ok = is_washing[frame_number] == 0 or random.random() < IS_WASHING_PROPORTION
                        if is_ok:
                            filename = 'frame{}_file_{}.jpg'.format(frame_number, os.path.splitext(videofile)[0])
                            subfolder = "1" if is_washing[frame_number] else "0"
                            save_path_and_name = os.path.join(output_folder, traintest + "_washing", subfolder, filename)
                            cv2.imwrite(save_path_and_name, image)
                            N_of_saved_is_washing_classification_frames += 1

                    # for frames with multiple annotators and washing on,
                    # potentially save the frame in its repective movement class set
                    if num_annotators > 1 and is_washing[frame_number] == 2:
                        N_of_frames_washing += 1
                        if codes[frame_number] >= 0:
                            N_of_matching_frames += 1

                            # skip some movement 0 frames
                            if (codes[frame_number] != 0 or random.random() < MOVEMENT_0_PROPORTION):
                                subfolder = str(codes[frame_number])
                                filename = 'frame{}_file_{}.jpg'.format(frame_number, os.path.splitext(videofile)[0])
                                # the name of the file storing the frames includes the frame number and the videofile name
                                save_path_and_name = os.path.join(output_folder, traintest, subfolder, filename)
                                cv2.imwrite(save_path_and_name, image)
                                N_of_saved_movement_classification_frames += 1

                    is_success, image = vidcap.read()
                    frame_number += 1


    N_of_nonmatching_frames = N_of_frames_washing - N_of_matching_frames
    print('Number of processed videofiles: ', N_of_videofiles)
    print('Number of considered frames: ', N_of_frames_considered)
    print('Number of frames marked as IS WASHING: ', N_of_frames_washing)
    print('Number of frames marked as IS WASHING that did not have a majority label: ', N_of_nonmatching_frames)
    print('Percentage of frames with a majority label: ', 100.0 * N_of_matching_frames / N_of_frames_washing if N_of_frames_washing else 0)
    print('Number of frames saved for is/isn\'t washing classification: ', N_of_saved_is_washing_classification_frames)
    print('Number of frames saved for movement classification: ', N_of_saved_movement_classification_frames)
    # if the videos have supplementary info extracted
    if N_of_saved_no_washing_with_hands_frames or N_of_saved_no_washing_no_hands_frames:
        print('Number of frames with no washing and with hands: ', N_of_saved_no_washing_with_hands_frames, 'with no hands: ', N_of_saved_no_washing_no_hands_frames)
    print('')


def main():
    random.seed(0)

    for movement in range(TOTAL_MOVEMENTS):
        mk(os.path.join(output_folder, "test", str(movement)))
        mk(os.path.join(output_folder, "trainval", str(movement)))

    for onoff in ["0", "1"]:
        mk(os.path.join(output_folder, "test_washing", onoff))
        mk(os.path.join(output_folder, "trainval_washing", onoff))

    list_of_folders = [d for d in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, d))]
    for folder in sorted(list_of_folders):
        get_frames(folder)

# ----------------------------------------------
if __name__ == "__main__":
    main()
