#!/usr/bin/env python3

#
# This creates both videos and frames separated in trainval and test sets
#

import os
import random
import cv2

TEST_PROPORTION = 0.3

input_dir         = './kaggle-dataset-6classes'
output_videos_dir = './kaggle-dataset-6classes-preprocessed/videos'
output_frames_dir = './kaggle-dataset-6classes-preprocessed/frames'

N_CLASSES = 7
subdirs = [str(i) for i in range(N_CLASSES)]


def mk(directory):
    folders = directory.split(os.path.sep)
    for i in range(len(folders)):
        so_far = str(os.path.sep).join(folders[:i+1])
        try:
            os.mkdir(so_far)
        except:
            pass

def process_video(classdir, filename, target_subset):
    input_fullname = os.path.join(input_dir, classdir, filename)
    output_fullname = os.path.join(output_videos_dir, target_subset, classdir, filename)
    print(input_fullname, target_subset)

    with open(input_fullname, "rb") as f:
        data = f.read()
    with open(output_fullname, "wb") as f:
        f.write(data)

    vidcap = cv2.VideoCapture(input_fullname)
    is_success, image = vidcap.read()
    frame_number = 0

    while is_success:
        out_filename = "frame_{}_{}.jpg".format(
            frame_number, os.path.splitext(filename)[0])
        save_path_and_name = os.path.join(output_frames_dir, target_subset, classdir, out_filename)
        cv2.imwrite(save_path_and_name, image)
        is_success, image = vidcap.read()
        frame_number += 1
   

def main():
    random.seed(123) # make it repeatable

    for movement in range(N_CLASSES):
        mk(os.path.join(output_videos_dir, "test", str(movement)))
        mk(os.path.join(output_videos_dir, "trainval", str(movement)))
        mk(os.path.join(output_frames_dir, "test", str(movement)))
        mk(os.path.join(output_frames_dir, "trainval", str(movement)))

    for classdir in subdirs:
        for filename in os.listdir(os.path.join(input_dir, classdir)):
            if ".mp4" not in filename:
                continue
            if random.random() < TEST_PROPORTION:
                target_subset = "test"
            else:
                target_subset = "trainval"
            process_video(classdir, filename, target_subset)

if __name__ == "__main__":
    main()
    print("all done")
