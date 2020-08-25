"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.

Licensed under the CC BY-NC-ND 4.0 license
(https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode).

Sergey Tulyakov, Ming-Yu Liu, Xiaodong Yang, Jan Kautz, MoCoGAN: Decomposing Motion and Content for Video Generation
https://arxiv.org/abs/1707.04993

Usage:
    train.py [options] <dataset> <log_folder> <generator> <discriminators>...

Options:
    --checkpoint=<path>             specify the checkpoint for models [default: ]
    --image_dataset=<path>          specifies a separate dataset to train for images [default: ]
    --image_batch=<count>           number of images in image batch [default: 32]
    --video_batch=<count>           number of videos in video batch [default: 32]
    --video_length=<len>            length of the video [default: 16]
    --image_size=<int>              resize all frames to this size [default: 64]
    --batches=<count>               specify number of batches to train [default: 100000]
    --n_channels=<count>            number of channels in the input data [default: 3]
    --every_nth=<count>             sample training videos using every nth frame [default: 2]

    --use_infogan                   when specified infogan loss is used
    --use_categories                when specified ground truth categories are used to
                                    train CategoricalVideoDiscriminator

    --use_noise                     when specified instance noise is used
    --noise_sigma=<float>           when use_noise is specified, noise_sigma controls
                                    the magnitude of the noise [default: 0]

    --trainer_type=<type>           type of trainer (exclusively 'vanilla_gan' or 'wasserstein_gan')
                                    [default: vanilla_gan]
    --optimizer=<type>              optimizer to choose [default: adam]

    --gen_learning_rate=<float>     learning rate for generator [default: 0.0002]
    --disc_learning_rate=<float>    learning rate for discriminators [default: 0.0002]
    --step_size=<int>               step size for merged discriminator [default: 2]
    --n_temp_blocks=<count>         number of temporal blocks [default: 3]
    --temporal_kernel_size=<count>  kernel size for temporal block [default: 3]
    --wgan_disc_iter=<count>        number of iterations to train discriminator for 1 generator iteration
                                    [default: 5]
    --clamp_lower=<float>           lower clamp edge [default: -0.01]
    --clamp_upper=<float>           upper clamp edge [default: 0.01]

    --temporal_sigma=<float>        Weighting parameter for temporal disc [default: 1.0]
    --temporal_beta=<float>         Weighting parameter for temporal gen loss [default: 1.0]

    --print_every=<count>           print every iterations [default: 100]
    --save_every=<count>            save model period [default: 1000]


    --num_workers=<count>           num workers to prefetch data [default: 4]
    --show_data_interval=<count>    show random sampled batches and learned images [default: 500]
    --save_graph                    saves the computational graph for discriminator and generator
    --seed=<count>                  sets the seed across the application [default: 0]

    --dim_z_content=<count>         dimensionality of the content input, ie hidden space [default: 50]
    --dim_z_motion=<count>          dimensionality of the motion input [default: 10]
    --dim_z_category=<count>        dimensionality of categorical input [default: 0]
    --deterministic                 sets full determinism on CuDNN operations
    --snapshot_init                 saves initial model snapshot
"""
import random

import numpy as np
import torch
import docopt

from mdp_video.trainers import Trainer

if __name__ == "__main__":
    ARGS = docopt.docopt(__doc__)

    # Determinism slows cudnn drastically
    if ARGS["--deterministic"]:
        torch.backends.cudnn.enabled = False

    # Seeding whole application
    SEED = int(ARGS["--seed"])
    if SEED >= 0:
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)

    TRAINER = Trainer(ARGS)
    TRAINER.train()
