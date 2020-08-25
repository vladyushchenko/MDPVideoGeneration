"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.

Licensed under the CC BY-NC-ND 4.0 license
(https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode).

Sergey Tulyakov, Ming-Yu Liu, Xiaodong Yang, Jan Kautz, MoCoGAN: Decomposing Motion and Content for Video Generation
https://arxiv.org/abs/1707.04993

Generates multiple videos given a model and saves them as video files using ffmpeg

Usage:
    generate_videos.py [options] <model>

Options:
    -n, --num_videos=<count>                number of videos to generate [default: 10]
    -o, --output_format=<ext>               save videos as [default: gif]
    -f, --number_of_frames=<count>          generate videos with that many frames [default: 16]

    --save_images                           save images as video
    --ffmpeg=<str>                          ffmpeg executable (on windows should be ffmpeg.exe). Make sure
                                            the executable is in your PATH [default: ffmpeg]
    --seed                                  seed for reproducable results
    --mosaic                                save mosaic
    --col_videos=<count>                    Mosaic videos in each row/col [default: 10]
    --diff                                  Calculates the difference between frame pairs
    --framewise                             saves framewise comparison
    --chunks                                save chunks
    --chunk_size=<count>                    Length for video as chunk [default: 8]
    --add_counter                           Add counter for mosaic
    --img_ext=<str>                         Encoder for images [default: jpg]
    --type=<str>                            [default: generator]
    --artifact=<str>                        [default: ]
    --add_noise_artifacts
    --add_video_break
    --border_list_good=<arg>                [default: ]
    --border_list_bad=<arg>                 [default: ]
"""
import os
import shutil
import subprocess as sp
from functools import partial
from typing import Tuple, Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from PIL import Image
from tqdm import tqdm
import docopt

from mdp_video.loaders.ucf_loader import make_dataset, video_loader
from mdp_video.loaders.video_loaders import Loader
from mdp_video.util import to_numpy, RealBatchSampler


def save_video(video: Any, folder: str, filename: str, ffmpeg: str, ext: str) -> None:
    """
    Create video with ffmpeg from image sequence.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)

    command = [
        ffmpeg,
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        "64x80",
        "-pix_fmt",
        "rgb24",
        "-r",
        "8",
        "-i",
        "-",
        "-c:v",
        ext,
        "-q:v",
        "3",
        "-an",
        os.path.join(folder, "{}.{}".format(filename, ext)),
    ]

    pipe = sp.Popen(command, stdin=sp.PIPE)
    pipe.stdin.write(video.tostring())
    pipe.communicate()


def save_images(video_seq: Any, out_folder: str, video_name: str, img_ext: str, nested: bool = True) -> None:
    """
    Save image sequence.
    """
    for counter, image in enumerate(video_seq):
        pil_img = Image.fromarray(image)
        name = "image_{}.{}".format(video_name, img_ext)
        folder = out_folder
        if nested:
            folder = os.path.join(out_folder, "Video_{}".format(video_name))
            name = "image_{:04d}.{}".format(counter, img_ext)
        if not os.path.exists(folder):
            os.makedirs(folder)
        pil_img.save(os.path.join(folder, name))


def save_video_sequence(video_seq: Any, out_folder: str, img_ext: str, video_name: str) -> None:
    """
    Save concatenated video.
    """
    video_seq = video_seq.transpose((0, 2, 1, 3))
    rows, height, width, channels = video_seq.shape
    video = video_seq.reshape(rows * height, width, channels)
    video = video.transpose((1, 0, 2))
    save_images([video], out_folder, video_name, img_ext, nested=False)


# pylint: disable=R0914
def generate_video(args: docopt.docopt, generator: Any, output_folder: str, saving_func: Any) -> None:
    """
    Create videos from generator samples.
    """

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    for counter in tqdm(range(args.num_videos), total=args.num_videos):
        seed = None
        if args.use_seed:
            seed = counter
        v, categories = generator.sample_videos(1, args.n_frames, seed)

        if not isinstance(categories, tuple):
            if categories is not None:
                categories = categories.data.item()
        else:
            if categories[1] is not None:
                categories = categories[1].data.item()

        video = to_numpy(v).squeeze().transpose((1, 2, 3, 0))

        if args.add_video_break:
            fill = np.ones(shape=(4, *video.shape[1:]), dtype=np.uint8) * 0
            video = np.concatenate((video, fill), axis=0)

        _, height, width, _ = video.shape

        pads = []
        for i, image in enumerate(video):
            pad = np.ones(shape=(image.shape[0] // 4, image.shape[1], image.shape[2]), dtype=np.uint8) * 255
            cv2.putText(
                pad, "Frame {}".format(i), (width // 2 - 20, height // 6), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1
            )
            pads.append(pad)

        pads = np.asarray(pads)
        video = np.concatenate((pads, video), axis=1)

        saving_func(video, os.path.join(output_folder, "Cat_{}".format(categories)), str(counter))

        # pylint: disable = W0143
        if saving_func.func != save_images:
            save_video_sequence(
                video,
                os.path.join(output_folder, "Cat_{}".format(categories)),
                video_name=str(counter),
                img_ext=args.img_ext,
            )


def generate_chunk(args: docopt.docopt, generator: Any, output_folder: str) -> None:
    """
    Create videos from generator samples.
    """

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    for counter in tqdm(range(args.num_videos), total=args.num_videos):
        seed = None
        if args.use_seed:
            seed = counter
        v, categories = generator.sample_videos(1, args.n_frames, seed)

        if not isinstance(categories, tuple):
            if categories is not None:
                categories = categories.data.item()
        else:
            if categories[1] is not None:
                categories = categories[1].data.item()

        video = to_numpy(v).squeeze().transpose((1, 2, 3, 0))
        video = video.transpose((0, 2, 1, 3))
        video = video[:: args.chunks_stride]
        rows, height, width, channels = video.shape
        video = video.reshape(rows * height, width, channels)
        video = video.transpose((1, 0, 2))
        if rows < args.chunk_size:
            return
        indices = [
            height * args.chunk_size * (i + 1)
            for i in range(0, (rows // args.chunk_size) - (1 if rows % args.chunk_size == 0 else 0))
        ]
        video = np.split(video, indices, axis=1)
        save_images(
            video,
            os.path.join(output_folder, "Cat_{}".format(categories)),
            video_name=str(counter),
            img_ext=args.img_ext,
            nested=True,
        )


# pylint: disable=R0914
def generate_mosaic(args: docopt.docopt, generator: Any, output_folder: str) -> None:
    """
    Create mosaic videos from generator samples.
    """

    img_out_folder = os.path.join(output_folder, "MosaicImages")
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)
    os.makedirs(img_out_folder)

    videos_list = []
    for counter in tqdm(range(args.col_videos), total=args.col_videos):
        seed = None
        if args.use_seed:
            seed = counter
        video, _ = generator.sample_videos(args.col_videos, args.n_frames, seed)

        video = to_numpy(video).squeeze()
        videos_list.append(video)

    videos = np.stack(videos_list)
    cols, rows, channels, length, height, width = videos.shape
    videos = videos.transpose(3, 0, 4, 1, 5, 2)
    videos = videos.reshape(length, height * cols, width * rows, channels)

    pads = []
    if args.add_counter:
        for i, image in enumerate(videos):
            pad = np.ones(shape=(height // 2, width * rows, channels)) * 255
            cv2.putText(
                pad,
                "Frame {}".format(i),
                (width * rows // 2 - 40, height // 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 0),
                1,
            )
            pads.append(pad)

        pads = np.asarray(pads)
        videos = np.concatenate((pads, videos), axis=1)

    length, total_height, total_width, channels = videos.shape

    # Add black frame to denote start of the video
    pad = np.ones(shape=(4, total_height, total_width, channels)) * 255
    videos = np.concatenate((videos, pad), axis=0)

    for counter, image in enumerate(videos):
        image = image.astype(np.uint8)
        Image.fromarray(image).save("{}/{}.{}".format(img_out_folder, counter, args.img_ext))

    sp.call(
        [
            "ffmpeg",
            "-r",
            "10",
            "-i",
            "{}/%d.{}".format(img_out_folder, args.img_ext),
            "-qscale",
            "0",
            "-s",
            "{}x{}".format(total_width, total_height),
            "{}/Mosaic.gif".format(output_folder),
        ]
    )


# pylint: disable=R0914
def generate_border_mosaic(args: docopt.docopt, generator: Any, output_folder: str) -> None:
    """
    Create mosaic videos from generator samples.
    """

    img_out_folder = os.path.join(output_folder, "MosaicImages")
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)
    os.makedirs(img_out_folder)

    videos_list = []
    for counter in tqdm(range(args.col_videos * args.col_videos), total=args.col_videos * args.col_videos):
        seed = None
        if args.use_seed:
            seed = counter
        video, _ = generator.sample_videos(1, args.n_frames, seed)

        video = to_numpy(video).squeeze()
        video = video.transpose(1, 0, 2, 3)
        length, channels, height, width = video.shape

        border_set_good = set(args.border_list_good)
        border_set_bad = set(args.border_list_bad)

        new_video = np.zeros((length, channels, height + 2 * args.border_size, width + 2 * args.border_size)).astype(
            np.long
        )
        for i, image in enumerate(video):
            # color = args.border_color_good
            if counter in border_set_good:
                color = args.border_color_good
            elif counter in border_set_bad:
                color = args.border_color_bad
            else:
                color = (0, 0, 0)

            image = image.transpose(1, 2, 0)
            border = cv2.copyMakeBorder(
                image,
                top=args.border_size,
                bottom=args.border_size,
                left=args.border_size,
                right=args.border_size,
                borderType=cv2.BORDER_CONSTANT,
                value=color,
            )
            border = border.transpose(2, 0, 1)
            new_video[i, :, :, :] = border

        videos_list.append(new_video)

    videos = np.stack(videos_list)
    videos = videos.reshape(
        (
            args.col_videos,
            args.col_videos,
            length,
            channels,
            height + 2 * args.border_size,
            width + 2 * args.border_size,
        )
    )
    videos = videos.transpose(0, 1, 3, 2, 4, 5)

    cols, rows, channels, length, height, width = videos.shape
    videos = videos.transpose(3, 0, 4, 1, 5, 2)
    videos = videos.reshape(length, height * cols, width * rows, channels)

    pads = []
    if args.add_counter:
        for i, image in enumerate(videos):
            pad = np.ones(shape=(height // 2, width * rows, channels)) * 255
            cv2.putText(
                pad,
                "Frame {}".format(i),
                (width * rows // 2 - 40, height // 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 0),
                1,
            )
            pads.append(pad)

        pads = np.asarray(pads)
        videos = np.concatenate((pads, videos), axis=1)

    length, total_height, total_width, channels = videos.shape

    # Add black frame to denote start of the video
    pad = np.ones(shape=(4, total_height, total_width, channels)) * 255
    videos = np.concatenate((videos, pad), axis=0)

    for counter, image in enumerate(videos):
        image = image.astype(np.uint8)
        Image.fromarray(image).save("{}/{}.{}".format(img_out_folder, counter, args.img_ext))

    sp.call(
        [
            "ffmpeg",
            "-r",
            "10",
            "-i",
            "{}/%d.{}".format(img_out_folder, args.img_ext),
            "-qscale",
            "0",
            "-s",
            "{}x{}".format(total_width, total_height),
            "{}/Mosaic.gif".format(output_folder),
        ]
    )


# pylint: disable=R0915
def generate_diff(args: docopt.docopt, generator: Any, output_folder: str) -> None:
    """
    Create difference frames and confusion heatmap.
    """

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    for counter in tqdm(range(args.num_videos), total=args.num_videos):
        seed = None
        if args.use_seed:
            seed = counter
        v, categories = generator.sample_videos(1, args.n_frames, seed)

        if not isinstance(categories, tuple):
            if categories is not None:
                categories = categories.data.item()
        else:
            if categories[1] is not None:
                categories = categories[1].data.item()

        video = to_numpy(v).squeeze().transpose((1, 2, 3, 0))

        row_stack_list = []
        for row_counter in range(video.shape[0]):
            row_image = video[row_counter, :, :, :].squeeze().astype("float32")

            col_stack = []
            for col_counter in range(video.shape[0]):
                col_image = video[col_counter, :, :, :].squeeze().astype("float32")

                # (score, diff_img) = skimage.measure.compare_ssim(row_image, col_image, full=True, multichannel=True)
                # diff_img = (diff_img * 255).astype("uint8")
                row_image = np.asarray(Image.fromarray(row_image.astype("uint8")).convert("L"), dtype=np.float)
                col_image = np.asarray(Image.fromarray(col_image.astype("uint8")).convert("L"), dtype=np.float)
                diff_img = np.abs(row_image - col_image)
                score = np.mean(diff_img)
                # score = (1 - meas.compare_ssim(row_image, col_image)) / 2
                diff_img = 255 - diff_img

                col_stack.append((diff_img, score))

            col_stack, hist_vals = list(zip(*col_stack))
            col_stack = np.stack(col_stack)
            row_stack_list.append((col_stack, hist_vals))

        row_stack, hist_vals = list(zip(*row_stack_list))
        row_stack = np.stack(row_stack)
        hist_vals = np.asarray(hist_vals)

        assert np.all(row_stack[0, 1] == row_stack[1, 0])

        if len(row_stack.shape) == 5:
            cols, rows, height, width, channels = row_stack.shape
            row_stack = row_stack.transpose(0, 2, 1, 3, 4)
        else:
            cols, rows, height, width = row_stack.shape
            row_stack = row_stack.transpose(0, 2, 1, 3)
            channels = 1

        row_stack = row_stack.reshape((height * cols, width * rows, channels)).squeeze()
        diff_img_table = Image.fromarray(row_stack).convert("L")

        folder = os.path.join(output_folder, "Cat_{}".format(categories))
        if not os.path.exists(folder):
            os.makedirs(folder)

        diff_img_table.save(os.path.join(folder, "VideoDiff_{:04d}.{}".format(counter, args.img_ext)))
        with open(
            (
                os.path.join(
                    folder,
                    "ConfusionMatrix_{:04d}_rank_{}_of_{}.txt".format(
                        counter, np.linalg.matrix_rank(hist_vals, hermitian=True), args.n_frames
                    ),
                )
            ),
            "w",
        ) as f:
            np.savetxt(f, hist_vals, fmt="%.4f")

        plt.rcParams["font.size"] = 20
        fig = plt.figure()
        mask = np.ones_like(hist_vals)
        mask[np.tril_indices_from(mask)] = False
        sns.heatmap(hist_vals, square=True, mask=mask, xticklabels=10, yticklabels=10)

        plt.xlabel("i. video frame number", fontsize=24)
        plt.ylabel("k. video frame number", fontsize=24)
        fig.savefig(
            os.path.join(folder, "Heatmap_{:04d}.{}".format(counter, args.img_ext)),
            dpi=fig.dpi,
            bbox_inches="tight",
            transparent=True,
        )
        plt.close()


# pylint: disable=R0914
def generate_framewise_comparison(args: docopt.docopt, generator: Any, output_folder: str) -> None:
    """
    Create comparison between the most similar video difference.
    """
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    for counter in tqdm(range(args.num_videos), total=args.num_videos):
        seed = None
        if args.use_seed:
            seed = counter
        v, categories = generator.sample_videos(1, args.n_frames, seed)

        if not isinstance(categories, tuple):
            if categories is not None:
                categories = categories.data.item()
        else:
            if categories[1] is not None:
                categories = categories[1].data.item()

        original_video = to_numpy(v).squeeze().transpose((1, 2, 3, 0))
        start_frame = original_video[0]
        original_video = original_video.transpose((0, 2, 1, 3))
        rows, height, width, channels = original_video.shape
        video = original_video.reshape(rows * height, width, channels)
        video = video.transpose((1, 0, 2))
        # plt.imshow(video)

        root_dir = ""
        for dataset_name in args.datasets:
            if dataset_name in args.model:
                data_dir = os.path.join(args.datasets_dir, "{}_raw".format(dataset_name))
                if os.path.isdir(data_dir):
                    root_dir = data_dir
                    break
        # create dataset
        dataset, _ = make_dataset(root_dir)

        global_score, global_min_index, global_min_item = np.inf, -1, None
        for item_dataset in dataset:
            item_dir = item_dataset["video_name"]

            item_frames = item_dataset["video_frames"]
            distances = [
                np.abs(start_frame - np.asarray(Image.open(os.path.join(item_dir, item)).convert("RGB")))
                for item in item_frames
            ]

            distances = np.asarray(distances)
            distances = np.mean(distances, axis=(1, 2, 3))
            min_index = np.argmin(distances)
            min_score = distances[min_index]

            if min_score < global_score:
                global_score = min_score
                global_min_index = min_index
                global_min_item = item_dataset

        # load global images and stack them on top

        sampling_sequence = [
            (global_min_index + args.every_nth * i) % global_min_item["video_length"]  # type: ignore
            for i in range(args.n_frames)
        ]
        sampling_sequence = np.asarray(sampling_sequence)

        closest_video_list = video_loader(
            global_min_item["video_name"], global_min_item["video_frames"][sampling_sequence]  # type: ignore
        )
        closest_video_list = [np.asarray(item) for item in closest_video_list]
        closest_video = np.stack(closest_video_list)

        closest_video = closest_video.transpose((0, 2, 1, 3))
        rows, height, width, channels = closest_video.shape
        closest_video = closest_video.reshape(rows * height, width, channels)
        closest_video = closest_video.transpose((1, 0, 2))

        video = np.concatenate((closest_video, video), axis=0)
        save_images(
            [video],
            os.path.join(output_folder, "Cat_{}".format(categories)),
            str(counter),
            nested=False,
            img_ext=args.img_ext,
        )


def launch_generation(args: docopt.docopt) -> None:
    """
    Launch generation.
    """
    if os.path.exists(args.output_folder):
        shutil.rmtree(args.output_folder)

    print("Saving Videos ...")
    output_folder = os.path.join(args.output_folder, "Video")
    saving_func = partial(save_video, ffmpeg=args.ffmpeg, ext=args.output_format)

    if args.mode == "generator":

        generator = torch.load(args.model)
        generator.eval()

    elif args.mode == "image" or args.mode == "artifact":
        generator = WrapperGenerator(args)
    else:
        raise RuntimeError("Generator wrong type")

    generate_video(args, generator, output_folder, saving_func)

    if args.save_images:
        print("Saving Images ...")
        output_folder = os.path.join(args.output_folder, "Images")
        saving_func = partial(save_images, img_ext=args.img_ext)
        generate_video(args, generator, output_folder, saving_func)

    if args.save_chunks:
        print("Saving Chunks ...")
        output_folder = os.path.join(args.output_folder, "Chunks")
        generate_chunk(args, generator, output_folder)

    if args.diff:
        print("Saving Diff")
        output_folder = os.path.join(args.output_folder, "Diff")
        generate_diff(args, generator, output_folder)

    if args.framewise:
        print("Saving Framewise Comparison")
        output_folder = os.path.join(args.output_folder, "Framewise")
        generate_framewise_comparison(args, generator, output_folder)

    if args.mosaic:
        print("Saving Mosaic ...")
        output_folder = os.path.join(args.output_folder, "Mosaic")
        generate_mosaic(args, generator, output_folder)


class Arguments:
    """
    Dataholder for docopt args.
    """

    def __init__(self, args: docopt.docopt) -> None:
        """
        Init call.

        :param args: program args
        """
        self.model = args["<model>"]
        self.location = args["<model>"]
        self.mode = args["--type"]
        self.num_videos = int(args["--num_videos"])
        self.n_frames = int(args["--number_of_frames"])
        self.calc_iter = self.num_videos
        self.video_length = self.n_frames
        self.output_folder = os.path.join(os.path.dirname(self.model), "SamplesLength_{}".format(self.n_frames))
        self.save_images = args["--save_images"]
        self.ffmpeg = args["--ffmpeg"]
        self.output_format = args["--output_format"]
        self.col_videos = int(args["--col_videos"])
        self.use_seed = args["--seed"]
        self.seed = 0
        self.mosaic = args["--mosaic"]
        self.diff = args["--diff"]
        self.framewise = args["--framewise"]
        self.datasets = ["actions", "shapes", "ucf_train", "ucf"]
        self.datasets_dir = "/home/vlad/datasets"
        self.every_nth = 2
        self.save_chunks = args["--chunks"]
        self.chunk_size = int(args["--chunk_size"])
        self.add_counter = args["--add_counter"]
        self.img_ext = args["--img_ext"]
        self.image_size = 64
        self.batch_size = 1
        self.cuda = True
        self.num_workers = 0
        self.chunks_stride = 8
        self.artifact = args["--artifact"]
        self.non_artifact_length = 16
        self.add_noise_artifacts = args["--add_noise_artifacts"]
        self.noise_artifacts_std = 0.01
        self.add_video_break = args["--add_video_break"]
        self.border_list_good = (
            list(map(int, args["--border_list_good"].split(","))) if args["--border_list_good"] else []
        )
        self.border_list_bad = list(map(int, args["--border_list_bad"].split(","))) if args["--border_list_bad"] else []
        self.border_color_good = (0, 0, 255)
        self.border_color_bad = (255, 0, 0)
        self.border_size = 2


class WrapperGenerator:
    """
    Wrapper for generator.
    """

    def __init__(self, args: docopt.docopt) -> None:
        """
        Init call.
        """
        self._args = args
        self._loader = RealBatchSampler(Loader(self._args)(), args=None)
        self._seed = -1

    # pylint: disable=W0613
    def sample_videos(self, num_samples: int = 0, video_len: int = 0, seed: int = -1) -> Tuple:
        """
        Generate sample videos.

        :param num_samples: number of samples
        :param video_len: length of video
        :param seed: seed number
        :return:
        """
        if (num_samples != self._args.batch_size) or (self._seed != seed):
            self._args.batch_size = num_samples
            if seed is not None:
                self._args.seed = seed
                self._seed = seed
            self._loader = RealBatchSampler(Loader(self._args)(), args=None)

        videos, _ = next(self._loader)

        return videos, None


if __name__ == "__main__":
    ARGS = docopt.docopt(__doc__)
    ARGS = Arguments(ARGS)

    launch_generation(ARGS)
