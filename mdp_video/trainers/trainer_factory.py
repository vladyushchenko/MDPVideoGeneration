"""
Factory for creating Generator/Discriminator trainer objects.
"""
import logging
import os
from typing import Any, Optional

import torch
import torchvision.transforms
from torch.utils.data import DataLoader
import docopt

import mdp_video.models as models
import mdp_video.util as util
from mdp_video.loaders.ucf_loader import SplittedVideoDataset
from mdp_video.logger import Logger
from mdp_video.trainers.temporal_trainer import TemporalGeneratorTrainer, TemporalDiscriminatorTrainer
from mdp_video.trainers.vanilla_trainer import VanillaGeneratorTrainer, VanillaDiscriminatorTrainer


class RealBatchSampler:
    """
    Wrapper for endless batch sampling.
    """

    def __init__(self, sampler: Any, cuda: bool = True) -> None:
        self._batch_size = sampler.batch_size
        self._sampler = sampler
        self._enumerator: Optional[Any] = None
        self._cuda = cuda

    def __call__(self, *args, **kwargs) -> Any:  # type: ignore
        """
        Sample provider call.
        """
        if self._enumerator is None:
            self._enumerator = enumerate(self._sampler)

        batch_idx, (batch, batch_cat) = next(self._enumerator)

        if self._cuda:
            batch = batch.cuda()
            batch_cat = batch_cat.cuda()

        if batch_idx == len(self._sampler) - 1:
            self._enumerator = enumerate(self._sampler)

        return batch, batch_cat

    @property
    def batch_size(self) -> int:
        """
        Get batch size.
        """
        return int(self._batch_size)


class DiscriminatorTrainerFactory:
    """
    Factory for DiscriminatorTrainers.
    """

    def __init__(self, args: docopt.docopt, generator: torch.nn.Module) -> None:
        self._args = args
        self._generator = generator

        self.image_size = int(self._args["--image_size"])
        self.logger = Logger(self._args["<log_folder>"])
        self.use_cuda = torch.cuda.is_available()

        self.transform = torchvision.transforms.Compose(
            [
                # torchvision.transforms.Resize(self.image_size),
                torchvision.transforms.CenterCrop(self.image_size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def load_disc(self, checkpoint: str, name: str) -> torch.nn.Module:
        """
        Load discriminator from checkpoint directory.
        """
        checkpoint = os.path.normpath(checkpoint)
        disc_path = os.path.join(checkpoint, "{}.pytorch".format(name))
        disc = torch.load(disc_path)
        self.logger.log("Restored checkpoint: {}".format(disc_path))
        return disc

    @staticmethod
    def build_discriminator(class_type: str, **kwargs) -> torch.nn.Module:  # type: ignore
        """
        Build discriminator.
        """
        discriminator_type = getattr(models, class_type)

        if "Categorical" not in class_type and "dim_categorical" in kwargs:
            kwargs.pop("dim_categorical")

        return discriminator_type(**kwargs)

    def select_trainer(self, name: str) -> Any:
        """
        Select trainer based on name.

        :param name: name of the trainer
        :return: current trainer class
        :raises RuntimeError: in case trainer type not supported
        """
        trainer_type = self._args["--trainer_type"]

        if trainer_type == "vanilla_gan":
            trainer: Any = VanillaDiscriminatorTrainer
        elif trainer_type == "temporal_gan":
            if "Temporal" in name:
                trainer = TemporalDiscriminatorTrainer
            else:
                trainer = VanillaDiscriminatorTrainer
        else:
            raise RuntimeError("Unvalid trainer type: {}".format(trainer_type))

        return trainer

    def create_trainer(self, name: str) -> Any:
        """
        Create trainer wrapper for discriminator.
        """
        if "Temporal" in name:
            trainer = self.create_temporal_discriminator(name)
        elif "Merged" in name:
            trainer = self.create_merged_discriminator(name)
        elif "Image" in name:
            trainer = self.create_image_discriminator(name)
        elif "Video" in name:
            trainer = self.create_video_discriminator(name)
        else:
            raise NotImplementedError("No suitable discriminator found")

        return trainer

    # pylint: disable=R0914
    def create_image_discriminator(self, name: str) -> torch.nn.Module:
        """
        Build image discriminator.
        """
        dataset_path = self._args["<dataset>"]
        num_workers = int(self._args["--num_workers"])
        use_noise = self._args["--use_noise"]
        noise_sigma = float(self._args["--noise_sigma"])
        every_nth = int(self._args["--every_nth"])
        n_channels = int(self._args["--n_channels"])
        image_batch_size = int(self._args["--image_batch"])
        learning_rate = float(self._args["--disc_learning_rate"])
        checkpoint = self._args["--checkpoint"]

        if checkpoint:
            image_disc = self.load_disc(checkpoint, name)
        else:
            image_disc = self.build_discriminator(
                class_type=name, n_channels=n_channels, use_noise=use_noise, noise_sigma=noise_sigma
            )
        image_flag_args = util.Flags(self._args, learning_rate)
        image_flag_args.use_infogan, image_flag_args.use_categories = False, False

        if self.use_cuda:
            image_disc = image_disc.cuda()

        dataset = SplittedVideoDataset(
            dataset_path, sample_duration=1, n_samples=every_nth, spatial_transform=self.transform, squeezed=1
        )

        image_loader = DataLoader(
            dataset, batch_size=image_batch_size, num_workers=num_workers, drop_last=True, shuffle=True
        )
        self.logger.log("Image Dataset length: {}".format(len(dataset)))

        image_loader = RealBatchSampler(image_loader, cuda=self.use_cuda)
        disc_trainer_class = self.select_trainer(name)
        return disc_trainer_class(image_disc, image_loader, self._generator.sample_images, image_flag_args)

    # pylint: disable=R0914
    def create_video_discriminator(self, name: str) -> torch.nn.Module:
        """
        Build video discriminator.
        """
        dataset_path = self._args["<dataset>"]
        num_workers = int(self._args["--num_workers"])
        use_noise = self._args["--use_noise"]
        noise_sigma = float(self._args["--noise_sigma"])
        n_channels = int(self._args["--n_channels"])
        dim_category = int(self._args["--dim_z_category"])
        every_nth = int(self._args["--every_nth"])
        video_length = int(self._args["--video_length"])
        video_batch_size = int(self._args["--video_batch"])
        learning_rate = float(self._args["--disc_learning_rate"])
        checkpoint = self._args["--checkpoint"]

        if checkpoint:
            video_disc = self.load_disc(checkpoint, name)
        else:
            kwargs = {
                "dim_categorical": dim_category,
                "n_channels": n_channels,
                "use_noise": use_noise,
                "noise_sigma": noise_sigma,
            }
            video_disc = self.build_discriminator(name, **kwargs)

        if self.use_cuda:
            video_disc = video_disc.cuda()

        dataset = SplittedVideoDataset(
            dataset_path, sample_duration=video_length, n_samples=every_nth, spatial_transform=self.transform
        )

        video_loader = DataLoader(
            dataset, batch_size=video_batch_size, num_workers=num_workers, drop_last=True, shuffle=True
        )
        self.logger.log("Video Dataset length: {}".format(len(dataset)))

        video_loader = RealBatchSampler(video_loader, cuda=self.use_cuda)
        video_flag_args = util.Flags(self._args, learning_rate)
        disc_trainer_class = self.select_trainer(name)
        return disc_trainer_class(video_disc, video_loader, self._generator.sample_videos, video_flag_args)

    # pylint: disable=R0914
    def create_temporal_discriminator(self, name: str) -> torch.nn.Module:
        """
        Build video discriminator.
        """
        dataset_path = self._args["<dataset>"]
        num_workers = int(self._args["--num_workers"])
        use_noise = self._args["--use_noise"]
        noise_sigma = float(self._args["--noise_sigma"])
        n_channels = int(self._args["--n_channels"])
        dim_category = int(self._args["--dim_z_category"])
        every_nth = int(self._args["--every_nth"])
        video_length = int(self._args["--video_length"])
        video_batch_size = int(self._args["--video_batch"])
        image_size = int(self._args["--image_size"])
        learning_rate = float(self._args["--disc_learning_rate"])
        n_blocks = int(self._args["--n_temp_blocks"])
        tempo_kernel_size = int(self._args["--temporal_kernel_size"])
        checkpoint = self._args["--checkpoint"]
        full_score = self._args["--trainer_type"] == "temporal_gan"

        if checkpoint:
            temporal_disc = self.load_disc(checkpoint, name)
        else:
            kwargs = {
                "dim_categorical": dim_category,
                "n_channels": n_channels,
                "use_noise": use_noise,
                "noise_sigma": noise_sigma,
                "n_blocks": n_blocks,
                "batch_size": video_batch_size,
                "image_size": image_size,
                "video_length": video_length,
                "tempo_kernel_size": tempo_kernel_size,
                "full_score": full_score,
            }
            temporal_disc = self.build_discriminator(name, **kwargs)

        if self.use_cuda:
            temporal_disc = temporal_disc.cuda()

        dataset = SplittedVideoDataset(
            dataset_path, sample_duration=video_length, n_samples=every_nth, spatial_transform=self.transform
        )
        video_loader = DataLoader(
            dataset, batch_size=video_batch_size, num_workers=num_workers, drop_last=True, shuffle=True
        )
        self.logger.log("Video Dataset length: {}".format(len(dataset)))

        video_loader = RealBatchSampler(video_loader, cuda=self.use_cuda)
        video_flag_args = util.Flags(self._args, learning_rate)
        disc_trainer_class = self.select_trainer(name)
        return disc_trainer_class(temporal_disc, video_loader, self._generator.sample_videos, video_flag_args)

    # pylint: disable=R0914
    def create_merged_discriminator(self, name: str) -> torch.nn.Module:
        """
        Build merged discriminator.
        """
        dataset_path = self._args["<dataset>"]
        num_workers = int(self._args["--num_workers"])
        use_noise = self._args["--use_noise"]
        noise_sigma = float(self._args["--noise_sigma"])
        n_channels = int(self._args["--n_channels"])
        dim_category = int(self._args["--dim_z_category"])
        every_nth = int(self._args["--every_nth"])
        video_length = int(self._args["--video_length"])
        video_batch_size = int(self._args["--video_batch"])
        merged_step_size = int(self._args["--step_size"])
        learning_rate = float(self._args["--disc_learning_rate"])
        checkpoint = self._args["--checkpoint"]

        if checkpoint:
            merged_disc = self.load_disc(checkpoint, name)
        else:
            merged_disc = self.build_discriminator(
                class_type=name,
                step_size=merged_step_size,
                dim_categorical=dim_category,
                n_channels=n_channels,
                use_noise=use_noise,
                noise_sigma=noise_sigma,
            )

        if self.use_cuda:
            merged_disc = merged_disc.cuda()

        def merged_transform(video: Any) -> Any:
            """
            Merge transform.
            """
            video_list = torch.split(video, 1, dim=1)
            chunked_video = []
            for item in util.chunks(video_list, merged_step_size):
                new_item = torch.cat(item).squeeze()
                chunked_video.append(new_item)
            chunked_video = torch.stack(chunked_video)
            return chunked_video

        dataset = SplittedVideoDataset(
            dataset_path,
            sample_duration=video_length,
            n_samples=every_nth,
            spatial_transform=self.transform,
            temporal_transform=merged_transform,
        )
        merged_loader = DataLoader(
            dataset=dataset, batch_size=video_batch_size, num_workers=num_workers, drop_last=True, shuffle=True
        )
        self.logger.log("Merged Dataset length: {}".format(len(dataset)))

        merged_loader = RealBatchSampler(merged_loader)
        merged_flag_args = util.Flags(self._args, learning_rate)
        disc_trainer_class = self.select_trainer(name)
        return disc_trainer_class(merged_disc, merged_loader, self._generator.sample_merged, merged_flag_args)


class GeneratorTrainerFactory:
    """
    Trainer factory for generators.
    """

    def __init__(self, args: docopt.docopt) -> None:
        """
        Init call.
        """
        self._args = args
        self.logger = Logger(self._args["<log_folder>"])
        self.use_cuda = torch.cuda.is_available()

    # pylint: disable=R0914
    def create_trainer(self, name: str) -> Any:
        """
        Build generator trainer wrapper.
        """
        dim_z_content = int(self._args["--dim_z_content"])
        dim_z_motion = int(self._args["--dim_z_motion"])
        dim_z_category = int(self._args["--dim_z_category"])
        n_channels = int(self._args["--n_channels"])
        video_length = int(self._args["--video_length"])
        merged_step_size = int(self._args["--step_size"])
        trainer_type = self._args["--trainer_type"]
        learning_rate = float(self._args["--gen_learning_rate"])
        gen_named_args = util.Flags(self._args, learning_rate)
        checkpoint = self._args["--checkpoint"]
        gen_iterations = 0

        try:
            generator_class = getattr(models, name)
        except AttributeError:
            self.logger.log("{} not found. Falling back to VideoGenerator".format(name), level=logging.WARN)
            generator_class = models.VideoGenerator

        if checkpoint:
            checkpoint = os.path.normpath(checkpoint)
            checkpoint_folder = os.path.basename(checkpoint)
            gen_iterations = int(checkpoint_folder.split("_")[-1])
            generator_path = os.path.join(checkpoint, "{}.pytorch".format(name))
            generator = torch.load(generator_path)
            self.logger.log("Restored generator checkpoint: {}".format(generator_path))
        else:
            generator = generator_class(
                n_channels, dim_z_content, dim_z_category, dim_z_motion, video_length, merged_step_size
            )
        if self.use_cuda:
            generator = generator.cuda()

        if trainer_type == "vanilla_gan":
            gen_trainer_class: Any = VanillaGeneratorTrainer
        elif trainer_type == "temporal_gan":
            gen_trainer_class = TemporalGeneratorTrainer
        else:
            raise RuntimeError("Unvalid trainer type: {}".format(trainer_type))

        return gen_trainer_class(generator, gen_named_args, gen_iterations)
