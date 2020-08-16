import argparse
import os
import re

import numpy as np
from PIL import Image
from tqdm import tqdm


# pylint: disable=R0914
def convert_images(root_path: str, output_dir: str, ext: str = "png", start: str = "") -> None:
    """
    Convert images to one imagese sequence.
    """
    class_folders = sorted(os.listdir(root_path))
    for classname in tqdm(class_folders, total=len(class_folders)):
        classname_path = os.path.join(root_path, classname)
        if not os.path.isdir(classname_path):
            continue

        video_list = sorted(os.listdir(classname_path), key=lambda x: int(re.findall(r"\d+", x)[0]))
        for counter, video_dir in enumerate(video_list):
            video_path = os.path.join(root_path, classname, video_dir)

            if not os.path.exists(video_path) or not video_dir.startswith(start):
                continue

            out_dir = os.path.join(output_dir, "Class_{}".format(classname), "Video_{}".format(counter))
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)

            image = Image.open(video_path)
            image = image.convert("RGB")

            shorter, longer = min(image.width, image.height), max(image.width, image.height)
            length = longer // shorter

            image = np.asarray(image)
            images = np.split(image, length, axis=image.shape.index(longer))

            for img_counter, image in enumerate(images):
                save_path = os.path.join(out_dir, "image_{}.{}".format(img_counter, ext))
                image = Image.fromarray(image)
                image.save(save_path)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser("Train Converter Mocogan concatenated to Raw Images ")
    PARSER.add_argument("dataset")
    PARSER.add_argument("out_dir")
    PARSER.add_argument("--ext", default="jpg")
    PARSER.add_argument("--start", default="")
    ARGS = PARSER.parse_args()

    convert_images(ARGS.dataset, output_dir=ARGS.out_dir, ext=ARGS.ext, start=ARGS.start)
