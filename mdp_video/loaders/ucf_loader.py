import functools
import json
import os
import re
import warnings
from typing import Tuple, Callable, Optional, Any, List

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image

IMG_EXTENSIONS = [
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
    ".ppm",
    ".PPM",
    ".bmp",
    ".BMP",
]


def is_image_file(filename: str) -> bool:
    """
    Check if file is image.

    :param filename: filename
    :return: bool
    """
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def load_value_file(file_path: str) -> float:
    """
    Load file.
    """
    with open(file_path, "r") as input_file:
        value = float(input_file.read().rstrip("\n\r"))
    return value


def pil_loader(path: str) -> Image:
    """
    Load image.
    """
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        with Image.open(f) as img:
            return img.convert("RGB")


def video_loader(video_dir_path: str, image_paths: List[str], image_loader: Any = pil_loader) -> List[Any]:
    """
    Get video loader.
    """
    return [image_loader(os.path.join(video_dir_path, image_path)) for image_path in image_paths]


def get_default_video_loader() -> Any:
    """
    Get default loader.
    """
    return functools.partial(video_loader, image_loader=pil_loader)


def load_annotation_data(data_file_path: str) -> Any:
    """
    Load data.
    """
    with open(data_file_path, "r") as data_file:
        return json.load(data_file)


def find_classes(root_dir: str) -> Tuple:
    """
    Find classes.
    """
    classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(root_path: str) -> Tuple:
    """
    Return dataset.
    """
    classes, classes_to_idx = find_classes(root_path)
    dataset = []
    for class_item in sorted(classes):

        d = os.path.join(root_path, class_item)
        if not os.path.isdir(d):
            continue

        video_names = os.listdir(d)
        for name in video_names:
            video_path = os.path.join(d, name)
            if not os.path.exists(video_path):
                continue

            video_image_paths = [item for item in os.listdir(video_path) if is_image_file(item)]
            video_image_paths = sorted(video_image_paths, key=lambda x: int(re.findall(r"\d+", x)[0]))
            video_length = len(video_image_paths)
            if video_length <= 0:
                continue

            sample = {
                "video_name": video_path,
                "video_frames": np.asarray(video_image_paths),
                "video_length": video_length,
                "label": classes_to_idx[class_item],
            }
            dataset.append(sample)

    return dataset, classes


class SplittedVideoDataset(data.Dataset):
    """
    Video dataset.
    """

    # pylint: disable=R0913
    def __init__(
        self,
        root_path: str,
        n_samples: int = 1,
        spatial_transform: Optional[Callable] = None,
        temporal_transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        sample_duration: int = 16,
        get_loader: Any = get_default_video_loader,
        squeezed: int = 0,
    ):
        """
        Init call.

        :param root_path:
        :param n_samples:
        :param spatial_transform:
        :param temporal_transform:
        :param target_transform:
        :param sample_duration:
        :param get_loader:
        :param squeezed:
        """
        self.data, self.class_names = make_dataset(root_path)

        self.n_samples = n_samples
        self.sample_duration = sample_duration
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.squeezed = squeezed
        self.data = [item for item in self.data if item["video_length"] >= self.sample_duration * self.n_samples]
        self.loader = get_loader()

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get item from dataset.

        :param index: Index
        :return: (image, target) where target is class_index of the target class
        :raises RuntimeWarning: in case video length is invalid
        """
        video_root = self.data[index]["video_name"]
        video_length = self.data[index]["video_length"]
        positions = self.data[index]["video_frames"]

        if video_length >= self.sample_duration * self.n_samples:
            frames_needed = self.n_samples * (self.sample_duration - 1)
            possible_start_bound = video_length - frames_needed
            start = 0 if possible_start_bound == 0 else np.random.randint(0, possible_start_bound, 1)[0]
            subsequence_idx = np.linspace(
                start, start + frames_needed, self.sample_duration, endpoint=True, dtype=np.int32,
            )
        elif video_length >= self.sample_duration:
            subsequence_idx = np.arange(0, self.sample_duration)
            warnings.warn("Data length {} of {} changed to {}".format(video_length, video_root, self.sample_duration))
        else:
            raise RuntimeWarning("Data length {} of video {} is invalid".format(video_length, video_root))

        clip = self.loader(video_root, positions[subsequence_idx])
        if self.spatial_transform is not None:
            clip = [self.spatial_transform(img) for img in clip]
            clip = torch.stack(clip).permute(1, 0, 2, 3)

        if self.temporal_transform is not None:
            clip = self.temporal_transform(clip)

        if self.squeezed:
            clip = clip.squeeze(self.squeezed)
        target = self.data[index]["label"]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return clip, target

    def __len__(self) -> int:
        """
        Get dataset length.
        """
        return len(self.data)
