#!/usr/bin/env python3

#
# This script takes input folders with video files and annotations
# and converts them to output folders.
# There are two output folders for movement classification:
# * "test" for testing dataset;
# * "trainval" for training and validation data.
# Each of these output folders contains subfolders named "0", "1", ... "6.
# The subfolders have jpeg images of the individual frames.
#

import cv2
import os
import json
import random

# Change these directories to your own locations
input_folder = './RSU_METC_dataset'
output_folder = './RSU_METC_dataset_preprocessed/frames/'
video_output_folder = './RSU_METC_dataset_preprocessed/videos/'

TEST_PROPORTION = 0.25

# the movement codes are from 0 to 6
TOTAL_MOVEMENTS = 7

# there is only one Annotator
TOTAL_ANNOTATORS = 1

FPS = 16

# allow up to +-0.5 seconds for reaction time
REACTION_TIME_FRAMES = FPS // 2

subdirs = ["Interface_number_1", "Interface_number_2", "Interface_number_3"]

snippet_extraction_file = "create_snippets.sh"

FULL_PROCESSING = True


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

def discount_reaction_indeterminacy(labels):
    new_labels = [u for u in labels]
    n = len(labels) - 1
    for i in range(n):
        if i == 0 or labels[i] != labels[i+1] or i == n - 1:
            start = max(0, i - REACTION_TIME_FRAMES)
            end = i
            for j in range(start, end):
                new_labels[j] = -1
            start = i
            end = min(n + 1, i + REACTION_TIME_FRAMES)
            for j in range(start, end):
                new_labels[j] = -1
    return new_labels


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
        annotator_dir = os.path.join(annotators_dir, "Annotator_" + str(a))
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

    if True:
        is_washing = discount_reaction_indeterminacy(is_washing)
        codes = discount_reaction_indeterminacy(codes)

    return is_washing, codes, frames_with_hands, num_annotators


def select_frames_to_save(is_washing, codes):
    old_code = -1
    old_saved = False
    num_snippets = 0
    mapping = {}
    current_snippet = {}
    for i in range(len(is_washing)):
        new_code = codes[i]
        new_saved = (is_washing[i] == 2 and new_code != -1)
        if new_saved != old_saved:
            if new_saved:
                num_snippets += 1
                current_snippet = {}
            else:
                # save current snippet
                for key in current_snippet:
                    mapping[key] = current_snippet[key]

        if new_saved:
            current_snippet_frame = len(current_snippet)
            current_snippet[i] = (current_snippet_frame, num_snippets, new_code)
        old_saved = new_saved
        old_code = new_code

    if old_saved:
        # save current snippet
        for key in current_snippet:
            mapping[key] = current_snippet[key]

    return mapping


def get_frames(folder):
    print('Processing folder: ' + folder + ' ...')

    for subdir, dirs, files in os.walk(folder):
        for videofile in files:
            if videofile.endswith(".mp4"):
                fullpath = os.path.join(subdir, videofile)
                is_washing, codes, frames_with_hands, num_annotators = find_frame_labels(fullpath)

                #print(fullpath)
                with open(snippet_extraction_file, "a+") as f:
                    f.write("echo processing {}...\n".format(fullpath))

                if random.random() < TEST_PROPORTION:
                    traintest = "test"
                else:
                    traintest = "trainval"

                # mapping from old frame number to (frame_num, snippet_num, code)
                new_frame_numbers = select_frames_to_save(is_washing, codes)

                # ffmpeg -r 16 -i frame_%d_snippet_2_2020-07-08_09-32-34_camera105.jpg -filter:v scale=320x240 out.mp4
                snippets = set()
                for key in new_frame_numbers:
                    _, num, code = new_frame_numbers[key]
                    snippets.add((num, code))
                snippets = sorted(list(snippets))

                vf = os.path.splitext(videofile)[0]
                with open(snippet_extraction_file, "a+") as f:
                    for snippet, code in snippets:
                        #print(snippet, code)
                        subfolder = str(code)
                        ipath = os.path.join(output_folder, traintest, subfolder)
                        opath = os.path.join(video_output_folder, traintest, subfolder)
                        command = "ffmpeg -r {} -i {}/frame_%d_snippet_{}_{}.jpg -filter:v scale=320x240 {}/snippet_{}_{}.mp4 >/dev/null 2>&1".format(FPS, ipath, snippet, vf, opath, snippet, vf)
                        f.write(command + "\n")

                if not FULL_PROCESSING:
                    continue

                vidcap = cv2.VideoCapture(fullpath)
                is_success, image = vidcap.read()
                frame_number = 0

                old_code = -1
                while is_success:
                    if frame_number in new_frame_numbers:
                        new_frame_num, snippet_num, code = new_frame_numbers[frame_number]
                        #print("frame {} mapped to {} snippet {}".format(frame_number, new_frame_num, snippet_num))

                        assert code == codes[frame_number]

                        subfolder = str(codes[frame_number])
                        filename = 'frame_{}_snippet_{}_{}.jpg'.format(new_frame_num, snippet_num, os.path.splitext(videofile)[0])
                        # the name of the file storing the frames includes the frame number and the videofile name
                        save_path_and_name = os.path.join(output_folder, traintest, subfolder, filename)
                        cv2.imwrite(save_path_and_name, image)
                    is_success, image = vidcap.read()
                    frame_number += 1

def main():
    random.seed(0)

    for movement in range(TOTAL_MOVEMENTS - 1):
        mk(os.path.join(output_folder, "test", str(movement)))
        mk(os.path.join(output_folder, "trainval", str(movement)))
        mk(os.path.join(video_output_folder, "test", str(movement)))
        mk(os.path.join(video_output_folder, "trainval", str(movement)))

    with open(snippet_extraction_file, "wt") as f:
        f.write("#!/bin/bash\n")

    for subdir in subdirs:
        get_frames(os.path.join(input_folder, subdir))

# ----------------------------------------------
if __name__ == "__main__":
    main()
