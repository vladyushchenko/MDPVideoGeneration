import argparse
import random

import numpy as np
import torch


from mdp_video.trainers import Trainer


def get_parser() -> argparse.ArgumentParser:
    """
    Get program parser.
    """
    parser = argparse.ArgumentParser("Trainer for MDP")
    parser.add_argument("--dataset", type=str, required=True, help="dataset to train on")
    parser.add_argument("--log_folder", type=str, required=True, help="logging folder")
    parser.add_argument("--checkpoint", type=str, default="", help="checkpoint for models")
    parser.add_argument("--generator", type=str, required=True, help="generator name")
    parser.add_argument("--discriminators", type=str, nargs="+", required=True, help="discriminator names")

    parser.add_argument(
        "--trainer_type",
        type=str,
        default="vanilla_gan",
        choices=["vanilla_gan", "temporal_gan"],
        help=" type of trainer",
    )
    parser.add_argument("--optimizer", type=str, default="adam", help="optimizer to use")

    parser.add_argument("--image_batch", type=int, default=32, help="batch size for images")
    parser.add_argument("--video_batch", type=int, default=32, help="batch size for videos")
    parser.add_argument("--video_length", type=int, default=16, help="video length for training")
    parser.add_argument("--image_size", type=int, default=64, help="target image size")

    parser.add_argument("--n_iterations", type=int, default=100000, help="# of training iterations")
    parser.add_argument("--n_channels", type=int, default=3, help="# of channels")
    parser.add_argument("--every_nth", type=int, default=2, help="# of subsampling rate for video")
    parser.add_argument("--seed", type=int, default=0, help="sets the seed across the application ")
    parser.add_argument("--num_workers", type=int, default=4, help="dataloader workers")
    parser.add_argument("--print_every", type=int, default=100, help="printing period")
    parser.add_argument("--save_every", type=int, default=1000, help="save model period")
    parser.add_argument("--show_data_interval", type=int, default=500, help="show random samples period")

    parser.add_argument("--dim_z_content", type=int, default=50, help="content dimension")
    parser.add_argument("--dim_z_motion", type=int, default=10, help="motion dimension")
    parser.add_argument("--dim_z_category", type=int, default=0, help="category dimension")
    parser.add_argument("--n_temp_blocks", type=int, default=3, help="number of temporal blocks")
    parser.add_argument("--temporal_kernel_size", type=int, default=3, help="kernel size for temporal block")

    parser.add_argument("--use_infogan", action="store_true", help="flag for infogan loss")
    parser.add_argument("--use_noise", action="store_true", help="when specified instance noise is used")
    parser.add_argument("--deterministic", action="store_true", help="sets full determinism on CuDNN operations")
    parser.add_argument("--snapshot_init", action="store_true", help="saves initial model snapshot")
    parser.add_argument(
        "--save_graph", action="store_true", help="saves the computational graph for discriminator and generator"
    )

    parser.add_argument("--gen_learning_rate", type=float, default=0.0002, help="learning rate for generator")
    parser.add_argument("--disc_learning_rate", type=float, default=0.0002, help="learning rate for discriminators")
    parser.add_argument("--noise_sigma", type=float, default=0.0, help="magnitude of the instance noise")
    parser.add_argument("--temporal_sigma", type=float, default=1.0, help="weighting parameter for temporal disc")
    parser.add_argument("--temporal_beta", type=float, default=1.0, help="weighting parameter for temporal gen loss")

    return parser


if __name__ == "__main__":
    PARSER = get_parser()
    ARGS = PARSER.parse_args()

    # Determinism slows cudnn drastically
    if ARGS.deterministic:
        torch.backends.cudnn.enabled = False

    # Seeding whole application
    SEED = ARGS.seed
    if SEED >= 0:
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)

    TRAINER = Trainer(ARGS)
    TRAINER.train()
