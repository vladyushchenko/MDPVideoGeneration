"""
Extract UCF dataset to folders.

After moving all the files using the 1_ file, we run this one to extract
the images from the videos and also create a data file we can use for
training and testing later.
"""

import argparse
import glob
import os
import os.path
from subprocess import call
from typing import Tuple

from tqdm import tqdm


def extract_files(root_path: str, ext: str) -> None:
    """
    Extract file to folders.

    [train|test], class, filename, nb frames

    Extracting can be done with ffmpeg:
    `ffmpeg -i video.mpg image-%04d.png`
    """
    nested_folders = ["train", "test"]

    for folder in nested_folders:
        folder = os.path.join(root_path, folder)
        videos_paths = glob.glob(folder + "/**/*.avi", recursive=True)

        for item in videos_paths:

            print(item)
            # Get the parts of the file.
            video_parts = get_video_parts(item)

            train_or_test, classname, video_folder, _, filename = video_parts

            # Only extract if we haven't done it yet.
            # Otherwise, just get the info.
            if not check_already_extracted(video_parts):
                src = os.path.join(train_or_test, classname, video_folder, filename)
                dest = os.path.join(train_or_test, classname, video_folder, "image_%d.{}".format(ext))
                call(["ffmpeg", "-i", src, dest])
                os.remove(src)


def get_nb_frames_for_video(video_parts: Tuple) -> int:
    """
    Return the number of frames that were extracted.
    """
    train_or_test, classname, video_folder, filename_no_ext, _ = video_parts
    generated_files = glob.glob(train_or_test + "/" + classname + "/" + video_folder + "/" + filename_no_ext + "*.png")
    return len(generated_files)


def get_video_parts(video_path: str) -> Tuple:
    """
    Return parts, given a full path to a video.
    """
    parts = video_path.split("/")
    filename = parts[-1]
    video_folder = parts[-2]
    filename_no_ext = filename.split(".")[0]
    classname = parts[-3]
    train_or_test = "/" + os.path.join(*parts[:-3])

    return train_or_test, classname, video_folder, filename_no_ext, filename


def check_already_extracted(video_parts: Tuple) -> bool:
    """
    Check to see if we created the -0001 frame of this file.
    """
    train_or_test, classname, video_folder, filename_no_ext, _ = video_parts
    return bool(
        os.path.exists(train_or_test + "/" + classname + "/" + video_folder + "/" + filename_no_ext + "-0001.png")
    )


def unpack_videos(root_dir: str, ext: str) -> None:
    """
    Unpack videos in folders.
    """
    videos_paths = glob.glob(root_dir + "**/*.avi", recursive=True)

    for item in tqdm(videos_paths, total=len(videos_paths)):
        # Get the parts of the file.
        parts = item.split("/")
        filename = parts[-1]
        filename_no_ext = filename.split(".")[0]

        src = item
        dest = os.path.join(root_dir, "{}_%d.{}".format(filename_no_ext, ext))
        call(["ffmpeg", "-i", src, dest])
        os.remove(src)


if __name__ == "__main__":
    # Extract images from videos and build a new file that we can use as
    # our data input file. It can have format:
    #
    # [train|test], class, filename, nb frames

    PARSER = argparse.ArgumentParser("UCF extracting part 2.")
    PARSER.add_argument("root_dir")
    PARSER.add_argument("--ext", default="jpg")
    ARGS = PARSER.parse_args()

    # unpack_videos(args.root_dir, args.ext)
    extract_files(ARGS.root_dir, ARGS.ext)
