import argparse
import os
import re

import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


# pylint: disable=R0914
def convert_images(root_path: str, output_dir: str, img_size: int = 240, ext: str = "png", name: str = "image") -> None:
    """
    Convert images to one imagese sequence.
    """
    img_transform = transforms.CenterCrop(img_size)
    directories = sorted(os.listdir(root_path))

    for classname in tqdm(directories, total=len(directories)):
        classname_path = os.path.join(root_path, classname)
        if not os.path.isdir(classname_path):
            continue

        video_list = sorted(os.listdir(classname_path), key=lambda x: int(re.findall(r"\d+", x)[0]))
        for counter, video_dir in enumerate(video_list):
            video_dir = os.path.join(root_path, classname, video_dir)
            stacked_image_list = []

            images_list = sorted(os.listdir(video_dir), key=lambda x: int(re.findall(r"\d+", x)[0]))
            for image_path in images_list:
                image_path = os.path.join(video_dir, image_path)

                if not os.path.exists(image_path):
                    continue

                image = Image.open(image_path)
                image = image.convert("RGB")
                image = img_transform(image)
                image = np.asarray(image)
                stacked_image_list.append(image)

            stacked_image = np.concatenate(stacked_image_list, axis=1)
            stacked_image = Image.fromarray(stacked_image)

            out_dir = os.path.join(output_dir, classname)
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)

            save_path = os.path.join(out_dir, "{}_{}.{}".format(name, counter, ext))
            stacked_image.save(save_path)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser("Train Converter Raw Images to Mocogan concatenated")
    PARSER.add_argument("dataset")
    PARSER.add_argument("out_dir")
    PARSER.add_argument("--image_size", default=240, type=int)
    PARSER.add_argument("--ext", default="png")
    PARSER.add_argument("--name", default="image")
    ARGS = PARSER.parse_args()

    convert_images(ARGS.dataset, output_dir=ARGS.out_dir, img_size=ARGS.image_size, ext=ARGS.ext, name=ARGS.name)
