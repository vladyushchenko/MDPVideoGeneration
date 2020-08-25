#
# Credit:
#
import argparse
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

from mdp_video.loaders.ucf_loader import make_dataset, video_loader
from mdp_video.loaders.video_loaders import Loader
from mdp_video.util import to_numpy, RealBatchSampler


def get_parser() -> argparse.ArgumentParser:
    """
    Get program parser.
    """
    parser = argparse.ArgumentParser("Interence for MDP.")
    parser.add_argument("--model", type=str, required=True, help="path to model")
    parser.add_argument(
        "--mode", type=str, default="generator", choices=["artifact", "generator", "image"], help="generation mode"
    )
    parser.add_argument("--artifact_name", type=str, default="", help="artifact name")

    parser.add_argument("--num_videos", type=int, default=10, help="number of videos")
    parser.add_argument("--n_frames", type=int, default=16, help="video length")
    parser.add_argument("--col_videos", type=int, default=10, help="videos in each row/col of mosaic")
    parser.add_argument("--chunks_size", type=int, default=8, help="length of video chunk")

    parser.add_argument("--output_format", type=str, default="gif", help="output format")
    parser.add_argument("--img_ext", type=str, default="png", help="image encoder")

    parser.add_argument("--save_images", action="store_true", help="flag to save images")
    parser.add_argument("--save_mosaic", action="store_true", help="flag to save mosaic")
    parser.add_argument("--save_diff", action="store_true", help="flag to save image diff")
    parser.add_argument("--save_framewise", action="store_true", help="saves framewise comparison")
    parser.add_argument("--save_chunks", action="store_true", help="flag to save video chunks")

    parser.add_argument("--add_counter", action="store_true", help="add counter to mosaic header")
    parser.add_argument("--add_noise_artifacts", action="store_true", help="add hand crafted artifacts")
    parser.add_argument("--add_video_break", action="store_true", help="add video breaks to video")
    parser.add_argument("--fix_seed", action="store_true", help="fixes the seed across the application")
    parser.add_argument("--cuda", action="store_true", help="enables cuda")

    parser.add_argument("--border_list_good", type=int, nargs="+", help="mark listed items on a mosaic grid as correct")
    parser.add_argument(
        "--border_list_bad", type=int, nargs="+", help="mark listed items on a mosaic grid as incorrect"
    )

    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--image_size", type=int, default=64, help="target image size")

    parser.add_argument("--n_iterations", type=int, default=100000, help="# of training iterations")
    parser.add_argument("--n_channels", type=int, default=3, help="# of channels")
    parser.add_argument("--every_nth", type=int, default=2, help="# of subsampling rate for video")
    parser.add_argument("--num_workers", type=int, default=0, help="dataloader workers")

    parser.add_argument("--non_artifact_length", type=int, default=16, help="non_artifact_length")
    parser.add_argument("--noise_artifacts_std", type=float, default=0.01, help="std for noise artifacts")

    parser.add_argument("--border_size", type=int, default=2, help="border_size in pixels")
    parser.add_argument(
        "--border_color_good", type=int, nargs="+", default=(0, 0, 255), help="border color for correct samples"
    )
    parser.add_argument(
        "--border_color_bad", type=int, nargs="+", default=(255, 0, 0), help="border color for incorrect samples"
    )

    return parser


def save_video(video: Any, folder: str, filename: str, ext: str) -> None:
    """
    Create video with ffmpeg from image sequence.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)

    command = [
        "ffmpeg",
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
    pipe.stdin.write(video.tostring())  # type: ignore
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
def generate_video(args: argparse.Namespace, generator: Any, output_folder: str, saving_func: Any) -> None:
    """
    Create videos from generator samples.
    """

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    for counter in tqdm(range(args.num_videos), total=args.num_videos):
        seed = None
        if args.fix_seed:
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


def generate_chunk(args: argparse.Namespace, generator: Any, output_folder: str) -> None:
    """
    Create videos from generator samples.
    """

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    for counter in tqdm(range(args.num_videos), total=args.num_videos):
        seed = None
        if args.fix_seed:
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
        video = video[:: args.chunks_size]
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
def generate_mosaic(args: argparse.Namespace, generator: Any, output_folder: str) -> None:
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
        if args.fix_seed:
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
def generate_border_mosaic(args: argparse.Namespace, generator: Any, output_folder: str) -> None:
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
        if args.fix_seed:
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
def generate_diff(args: argparse.Namespace, generator: Any, output_folder: str) -> None:
    """
    Create difference frames and confusion heatmap.
    """

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    for counter in tqdm(range(args.num_videos), total=args.num_videos):
        seed = None
        if args.fix_seed:
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
def generate_framewise_comparison(args: argparse.Namespace, generator: Any, output_folder: str) -> None:
    """
    Create comparison between the most similar video difference.
    """
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    for counter in tqdm(range(args.num_videos), total=args.num_videos):
        seed = None
        if args.fix_seed:
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
        for dataset_name in ["actions", "shapes", "ucf_train", "ucf"]:
            if dataset_name in args.model:
                data_dir = os.path.join(args.datasets_root, "{}_raw".format(dataset_name))
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


def launch_generation(args: argparse.Namespace) -> None:
    """
    Launch generation.
    """
    if os.path.exists(args.output_folder):
        shutil.rmtree(args.output_folder)

    print("Saving Videos ...")
    output_folder = os.path.join(args.output_folder, "Video")
    saving_func = partial(save_video, ext=args.output_format)

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

    def __init__(self, args: argparse.Namespace) -> None:
        """
        Init call.

        :param args: program args
        """


class WrapperGenerator:
    """
    Wrapper for generator.
    """

    def __init__(self, args: argparse.Namespace) -> None:
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
    PARSER = get_parser()
    ARGS = PARSER.parse_args()
    ARGS.output_folder = os.path.join(os.path.dirname(ARGS.model), "SamplesLength_{}".format(ARGS.n_frames))

    assert len(ARGS.border_color_good) == 3
    assert len(ARGS.border_color_bad) == 3

    launch_generation(ARGS)
