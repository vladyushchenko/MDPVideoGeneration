import argparse
import os
from functools import partial
from typing import Callable, Any

import matplotlib.pyplot as plt
import numpy as np
import skimage.measure as meas
from tqdm import tqdm

from mdp_video.loaders import Loader
from mdp_video.util import to_numpy

plt.rcParams.update({"font.size": 20})


def video_distance_batch(videos: np.ndarray, metric: Callable, order_func: Callable) -> Any:
    """
    Calculate video distance for videos.

    :param videos: videos to process
    :param metric: metric func
    :param order_func: ordering function min/max
    :return: list with distances
    """
    video_length = videos.shape[1]

    distances = []
    for video in videos:
        distance = []
        for i in range(1, video_length):
            anchor_frame = video[i]
            scores = [metric(anchor_frame, video[j]) for j in range(i)]
            value = order_func(scores)

            assert value >= 0

            distance.append(value)
        if len(distance) == video_length - 1:
            distances.append(distance)

    return distances


def wrapper(func: Callable) -> Callable:
    """
    Wrap func.

    :param func: function to wrap
    :return: dissimilarity
    """

    def wrapper_dissimilarity(*args, **kwargs) -> Any:  # type: ignore
        return (1 - func(*args, **kwargs)) / 2

    return wrapper_dissimilarity


def temporal_metrics(args: argparse.Namespace) -> None:
    """
    Calculate temporal metrics.

    :param args: program args
    """
    fake_loader = Loader(args)()

    if args.metric == "dssim":
        metric: Any = partial(meas.compare_ssim, multichannel=True)
        metric = wrapper(metric)
        order_func = min
    elif args.metric == "ssim":
        metric = partial(meas.compare_ssim, multichannel=True)
        order_func = min

    elif args.metric == "psnr":
        metric = meas.compare_psnr
        order_func = max
    else:
        metric = meas.compare_mse
        order_func = min

    overall_distances = []

    for videos, _ in tqdm(fake_loader, total=args.calc_iter // args.batch_size):
        fake_videos = to_numpy(videos).transpose(0, 2, 3, 4, 1)
        video_distances = video_distance_batch(fake_videos, metric=metric, order_func=order_func)
        overall_distances.append(video_distances)

    # # For mocogan comparison
    # for _ in tqdm(range(args.calc_iter // args.batch_size), total=args.calc_iter // args.batch_size):
    #     videos = next(fake_loader)
    #     videos = videos[0]
    #     fake_videos = to_numpy(videos).transpose(0, 2, 3, 4, 1)
    #     video_distances = video_distance_batch(fake_videos, metric=metric, order_func=order_func)
    #     if video_distances:
    #         overall_distances.append(np.asarray(video_distances))

    overall_distances = np.concatenate(overall_distances)
    overall_distances = np.mean(overall_distances, axis=0)
    score = np.mean(overall_distances)
    print(score)

    if not os.path.isdir(args.out_folder):
        os.makedirs(args.out_folder)

    fig = plt.figure()
    ticks = np.arange(1, len(overall_distances) + 1, 1)
    plt.plot(ticks, overall_distances)

    # naming the x axis
    plt.xlabel("Frames")
    # naming the y axis
    plt.ylabel("{} value".format(str(args.metric).upper()))
    plt.gca()

    # giving a title to my graph
    plt.title("t{}: {:3f}".format(str(args.metric).upper(), score))

    name = "{}_{}_TemporalDistance_Iter{}_Score_{:3f}.{}".format(
        order_func.__name__, str(args.metric).upper(), args.calc_iter, score, args.img_ext
    )
    if args.save_data:
        np.save(os.path.join(args.out_folder, name), overall_distances)

    # function to show the plot
    plt.show()
    fig.savefig(os.path.join(args.out_folder, name), dpi=fig.dpi, bbox_inches="tight")
    plt.close()


def get_parser() -> argparse.ArgumentParser:
    """
    Get program parser.

    :return: program parser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--location")
    parser.add_argument("--save_data", action="store_true")
    parser.add_argument("--mode", default="generator")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--cuda", action="store_false")
    parser.add_argument("--calc_iter", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--video_length", type=int, default=64)
    parser.add_argument("--every_nth", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--metric", default="ssim")
    parser.add_argument("--launches", type=int, default=4)
    parser.add_argument("--out_folder")
    parser.add_argument("--img_ext", default="png")
    parser.add_argument("--artifact", default="")
    parser.add_argument("--non_artifact_length", type=int, default=8)
    parser.add_argument("--add_noise_artifacts", action="store_true")
    parser.add_argument("--noise_artifacts_std", type=float, default=0.01)
    parser.add_argument("--n_frames", type=int, default=16)

    return parser


if __name__ == "__main__":
    PARSER = get_parser()

    ARGS = PARSER.parse_args()
    ARGS.out_folder = os.path.join(os.path.dirname(ARGS.location), "TemporalMetrics_{}".format(ARGS.video_length))

    temporal_metrics(ARGS)
