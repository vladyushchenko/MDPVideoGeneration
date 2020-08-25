"""
Minimal Reference implementation for the Frechet Video Distance (FVD).

FVD is a metric for the quality of video generation models. It is
inspired by the FID (Frechet Inception Distance) used for images, but
uses a different embedding to be better suitable for videos.


Code is taken from following repository (Apache 2.0 License)
https://github.com/google-research/google-research/blob/master/frechet_video_distance/frechet_video_distance.py
"""
import argparse
from copy import deepcopy
from typing import Any, Tuple

import numpy as np
from tqdm import tqdm
import tensorflow as tf
import tensorflow_hub as hub

from mdp_video.loaders import Loader
from mdp_video.util import to_numpy, RealBatchSampler


def preprocess(videos: tf.Tensor, target_resolution: Tuple[int, int]) -> Any:
    """
    Run some preprocessing on the videos for I3D model.

    :param videos: <T>[batch_size, num_frames, height, width, depth] The videos to be
      preprocessed. We don't care about the specific dtype of the videos, it can
      be anything that tf.image.resize_bilinear accepts. Values are expected to
      be in the range 0-255.
    :param target_resolution: (width, height): target video resolution
    :return: videos: <float32>[batch_size, num_frames, height, width, depth]
    """

    videos_shape = videos.shape.as_list()
    all_frames = tf.reshape(videos, [-1] + videos_shape[-3:])
    resized_videos = tf.image.resize_bilinear(all_frames, size=target_resolution)
    target_shape = [videos_shape[0], -1] + list(target_resolution) + [3]
    output_videos = tf.reshape(resized_videos, target_shape)
    scaled_videos = 2.0 * tf.cast(output_videos, tf.float32) / 255.0 - 1
    return scaled_videos


def _is_in_graph(tensor_name: tf.Tensor) -> bool:
    """
    Check whether a given tensor does exists in the graph.
    """
    try:
        tf.get_default_graph().get_tensor_by_name(tensor_name)
    except KeyError:
        return False
    return True


def create_id3_embedding(videos: tf.Tensor, batch_size: int) -> tf.Tensor:
    """
    Embed the given videos using the Inflated 3D Convolution network.

    Downloads the graph of the I3D from tf.hub and adds it to the graph on the
    first call.

    :param videos: <float32>[batch_size, num_frames, height=224, width=224, depth=3]. Expected range is [-1, 1].
    :param batch_size: batch size
    :return: <float32>[batch_size, embedding_size]. embedding_size depends on the model used.
    :raises ValueError: when a provided embedding_layer is not supported.
    """

    module_spec = "https://tfhub.dev/deepmind/i3d-kinetics-400/1"

    # Making sure that we import the graph separately for
    # each different input video tensor.
    module_name = "fvd_kinetics-400_id3_module_" + videos.name.replace(":", "_")

    assert_ops = [
        tf.Assert(tf.reduce_max(videos) <= 1.001, ["max value in frame is > 1", videos]),
        tf.Assert(tf.reduce_min(videos) >= -1.001, ["min value in frame is < -1", videos]),
        tf.assert_equal(tf.shape(videos)[0], batch_size, ["invalid frame batch size: ", tf.shape(videos)], summarize=6),
    ]
    with tf.control_dependencies(assert_ops):
        videos = tf.identity(videos)

    module_scope = "%s_apply_default/" % module_name

    # To check whether the module has already been loaded into the graph, we look
    # for a given tensor name. If this tensor name exists, we assume the function
    # has been called before and the graph was imported. Otherwise we import it.
    # Note: in theory, the tensor could exist, but have wrong shapes.
    # This will happen if create_id3_embedding is called with a frames_placehoder
    # of wrong size/batch size, because even though that will throw a tf.Assert
    # on graph-execution time, it will insert the tensor (with wrong shape) into
    # the graph. This is why we need the following assert.
    video_batch_size = int(videos.shape[0])
    assert video_batch_size in [batch_size, -1, None], "Invalid batch size"
    tensor_name = module_scope + "RGB/inception_i3d/Mean:0"
    if not _is_in_graph(tensor_name):
        i3d_model = hub.Module(module_spec, name=module_name)
        i3d_model(videos)

    # gets the kinetics-i3d-400-logits layer
    tensor_name = module_scope + "RGB/inception_i3d/Mean:0"
    tensor = tf.get_default_graph().get_tensor_by_name(tensor_name)
    return tensor


def calculate_fvd(real_activations: tf.Tensor, generated_activations: tf.Tensor) -> tf.Tensor:
    """
    Return a list of ops that compute metrics as funcs of activations.

    :param real_activations: <float32>[num_samples, embedding_size]
    :param generated_activations: <float32>[num_samples, embedding_size]
    :return: FVD score
    """
    return tf.contrib.gan.eval.frechet_classifier_distance_from_activations(real_activations, generated_activations)


# pylint: disable=R0914
def start(args: argparse.Namespace) -> None:
    """
    Start calculation.

    :param args: program arguments
    """
    print("Noise {}".format(args.add_noise_artifacts))
    print("Length {}".format(args.video_length))

    real_args = deepcopy(args)
    real_args.mode = "image"
    real_args.location = args.dataset_loc
    real_args.add_noise_artifacts = False
    real_args.n_frames = real_args.video_length
    real_loader = RealBatchSampler(Loader(real_args)(), real_args)

    x = tf.placeholder(tf.float16, shape=(args.batch_size, args.video_length, 64, 64, 3))
    item1 = create_id3_embedding(preprocess(x, (224, 224)), batch_size=args.batch_size)

    scores = []
    seed_offset = args.seed
    for i in range(args.launches):
        if seed_offset >= 0:
            print("Setting seed {}".format(i + seed_offset))
            args.seed = i + seed_offset
        fake_args = args
        fake_loader = Loader(fake_args)()
        real_embeds, fake_embeds = [], []

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())

        for _ in tqdm(range(args.calc_iter // args.batch_size), total=args.calc_iter // args.batch_size):
            fake_tuple, real_tuple = next(iter(fake_loader)), next(real_loader)

            # if fake_args.add_noise_artifacts:
            #     x, x_orig, _ = fake_tuple
            #     fake_videos = to_numpy(fake_tuple[0]).transpose(0, 2, 3, 4, 1)
            #
            # else:
            #     fake_videos = to_numpy(fake_tuple[0]).transpose(0, 2, 3, 4, 1)

            fake_videos = to_numpy(fake_tuple[0]).transpose(0, 2, 3, 4, 1)
            real_videos = to_numpy(real_tuple[0]).transpose(0, 2, 3, 4, 1)

            lol1 = sess.run(item1, feed_dict={x: fake_videos})
            lol2 = sess.run(item1, feed_dict={x: real_videos})

            fake_embeds.append(lol1)
            real_embeds.append(lol2)

        fake_videos = np.concatenate(fake_embeds)
        real_videos = np.concatenate(real_embeds)
        print(fake_videos.shape)
        fake_videos = tf.convert_to_tensor(fake_videos, np.float32)
        real_videos = tf.convert_to_tensor(real_videos, np.float32)
        result = calculate_fvd(fake_videos, real_videos).eval(session=sess)
        print(result)
        scores.append(result)

    scores = np.asarray(scores)
    print(np.max(scores), np.mean(scores), np.std(scores))


def get_parser() -> argparse.ArgumentParser:
    """
    Get program parser.

    :return: parser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--location")
    parser.add_argument("--mode", default="")
    parser.add_argument("--dataset_loc")
    parser.add_argument("--video_length", type=int)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--cuda", action="store_false")
    parser.add_argument("--calc_iter", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--launches", type=int, default=4)
    parser.add_argument("--every_nth", type=int, default=2)
    parser.add_argument("--artifact", default="")
    parser.add_argument("--non_artifact_length", type=int, default=8)
    parser.add_argument("--add_noise_artifacts", action="store_true")
    parser.add_argument("--noise_artifacts_std", type=float, default=0.01)
    parser.add_argument("--map_loc", type=str, default="cpu")

    return parser


if __name__ == "__main__":
    # NOTE: number of videos must be divisible by 16.
    PARSER = get_parser()

    ARGS = PARSER.parse_args()
    print(ARGS)

    start(ARGS)
