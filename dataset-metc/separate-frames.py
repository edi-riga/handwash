#!/usr/bin/env python3

#
# This script puts frames from the RSU METC dataset into a format that
# is directly usable for NN training, and separated them in test and trainval subsets.
# It also extracts video snippets and does the same.
# Different videos are put in completely different subsets.
#

import os
import glob
import random
from subprocess import PIPE, DEVNULL, Popen

TEST_PROPORTION = 0.25

# The average inter-frame interval is such that the framereate is around 16.5
# As it has to be an integer (I think!), we slow it down a bit.
FRAMERATE = 16

metc_data_dir = '../RSU_METC_dataset'
out_frame_data_dir = '../RSU_METC_dataset_preprocessed/frames'
out_video_data_dir = '../RSU_METC_dataset_preprocessed/videos'

subdirs = ["interface_number_1", "interface_number_2", "interface_number_3"]


# do not include some frames before the code changes - this may be because of a slow human reaction time
REACTION_TIME_FRAMES = FRAMERATE # 1 second


def mk(directory):
    folders = directory.split(os.path.sep)
    for i in range(len(folders)):
        so_far = str(os.path.sep).join(folders[:i+1])
        try:
            os.mkdir(so_far)
        except:
            pass

def write_video(data, filename):
    if os.path.exists(filename):
        try:
            os.remove(filename)
        except Exception as ex:
            print(ex)
    process = Popen(["ffmpeg", "-r", str(FRAMERATE), "-f", "image2pipe", "-i", "-", filename], stdin=PIPE, stdout=PIPE, stderr=DEVNULL)
    # pipe all files to ffmpeg
    out = process.communicate(data)
    if len(out) and len(out[0]):
        print("out=", out)


def rename_frames(target_dir, target_subset, frame_filenames):
    print(target_dir, target_subset)
    frame_numbers = [int(os.path.splitext(os.path.basename(u))[0][6:]) for u in frame_filenames]
    frame_dirs = [os.path.dirname(u) for u in frame_filenames]
    # treat _not_ok as movement 0
    frame_codes = [0 if "_" in os.path.basename(u) else int(os.path.basename(u)) for u in frame_dirs]

    frame_info = list(zip(frame_numbers, frame_dirs, frame_codes))
    frame_info.sort()

    frame_codes = [u[2] for u in frame_info]

    # set movement code during reaction time to -1: it will not be saved
    frame_codes_new = []
    for i in range(len(frame_codes)):
        code_subset = frame_codes[i:i+REACTION_TIME_FRAMES]
        first_code = code_subset[0]
        if not all([code == first_code for code in code_subset]):
            frame_codes_new.append(-1)
        else:
            frame_codes_new.append(first_code)

    for i in range(len(frame_codes_new)):
        frame_info[i] = (frame_info[i][0], frame_info[i][1], frame_codes_new[i])

    #print(len(frame_codes), len(frame_codes_new))
    #print(frame_codes)
    #print(frame_codes_new)

    video_dir = os.path.basename(target_dir)

    # read all files in a buffer
    video_data = b""
    prev_code = -1
    video_number = 0
    for fn, fdirname, code in frame_info:
        fdirname = os.path.basename(fdirname)

        if prev_code != -1 and prev_code != code:
            output_video_filename = "snippet_{}_{}.mp4".format(video_number, video_dir)
            output_video_fullname = os.path.join(out_video_data_dir, target_subset, str(prev_code), output_video_filename)
            #print("write video", video_number, output_video_fullname)
            write_video(video_data, output_video_fullname)
            video_number += 1
            video_data = b""

        if code != -1:
            input_filename = "frame_{}.jpg".format(fn)
            input_fullname = os.path.join(target_dir, "phone", fdirname, input_filename)
            output_filename = "frame_{}_{}.jpg".format(fn, video_dir)
            output_fullname = os.path.join(out_frame_data_dir, target_subset, str(code), output_filename)
            #print("rename {} to {}".format(input_fullname, output_fullname))
            with open(input_fullname, "rb") as f:
                data = f.read()
                video_data += data
            with open(output_fullname, "wb") as f:
                f.write(data)

        prev_code = code

    if len(video_data):
        # save the last chunk
        output_video_filename = "snippet_{}_{}.mp4".format(video_number, video_dir)
        output_video_fullname = os.path.join(out_video_data_dir, target_subset, str(prev_code), output_video_filename)
        #print("write last video", video_number, output_video_fullname)
        write_video(video_data, output_video_fullname)
   

def main():
    random.seed(123) # make it repeatable

    for movement in range(N_CLASSES):
        mk(os.path.join(output_videos_dir, "test", str(movement)))
        mk(os.path.join(output_videos_dir, "trainval", str(movement)))
        mk(os.path.join(output_frames_dir, "test", str(movement)))
        mk(os.path.join(output_frames_dir, "trainval", str(movement)))

    for sd in subdirs:
        for subdir in os.listdir(os.path.join(metc_data_dir, sd)):
            d = os.path.join(metc_data_dir, sd, subdir)
            if random.random() < TEST_PROPORTION:
                target_subset = "test"
            else:
                target_subset = "trainval"
            image_dir = os.path.join(d, "phone")
            jpg_files = os.path.join(image_dir, "*/*.jpg")
            frame_filenames = glob.glob(jpg_files)
            rename_frames(d, target_subset, frame_filenames)

if __name__ == "__main__":
    main()
    print("all done")
