"""
Trainer module for training GANs.
"""
import logging
import os
import time
from typing import Any, List

import torch
from tqdm import tqdm

import mdp_video.util as util
from mdp_video.logger import Logger
from mdp_video.trainers.trainer_factory import DiscriminatorTrainerFactory, GeneratorTrainerFactory


class Trainer:
    """
    Class for model training loop.
    """

    def __init__(self, args: Any) -> None:
        """
        Init call.
        """
        self._args = args
        self.logger = Logger(self._args.log_folder)
        self.logger.log(args)
        self.generator_trainer = self._create_generator_trainer()
        self.discriminator_trainers = self._create_discriminator_trainers()

    def _create_generator_trainer(self) -> Any:
        """
        Create generator.
        """
        gen_type = self._args.generator
        generator_factory = GeneratorTrainerFactory(self._args)
        return generator_factory.create_trainer(gen_type)

    def _create_discriminator_trainers(self) -> List[Any]:
        """
        Create discriminators.
        """

        dim_category = self._args.dim_z_category
        disc_types = self._args.discriminators

        if dim_category > 0:
            self.logger.log(
                "Setting infogan to True, as dim_category is {}".format(dim_category), level=logging.WARNING
            )
            self._args.use_infogan = True
        else:
            self.logger.log(
                "Setting infogan to False, as dim_category is {}".format(dim_category), level=logging.WARNING
            )
            self._args.use_infogan = False

        disc_factory = DiscriminatorTrainerFactory(self._args, self.generator_trainer.generator)
        discriminator_trainers = [disc_factory.create_trainer(disc_type) for disc_type in disc_types]
        return discriminator_trainers

    def save_models(self, log_folder: str, batch_number: int) -> None:
        """
        Make checkpoint of all trained models.
        """

        def save_model(model: Any, folder: str) -> None:
            """
            Save model.
            """
            if not os.path.exists(folder):
                os.makedirs(folder)
            torch.save(model, os.path.join(folder, "{}.pytorch".format(model.__class__.__name__)))

        save_model(self.generator_trainer.generator, os.path.join(log_folder, "Checkpoint_%05d" % batch_number))
        for trained_model in self.discriminator_trainers:
            save_model(trained_model.discriminator, os.path.join(log_folder, "Checkpoint_%05d" % batch_number))

    def save_examples(self, batch_number: int) -> None:
        """
        Generate and save samples.
        """
        self.generator_trainer.generator.eval()

        images, _ = self.generator_trainer.generator.sample_images(10)
        self.logger.image_summary("Images", util.to_numpy(images).transpose(0, 2, 3, 1), batch_number)

    def _get_disc_iter(self) -> int:
        """
        Return the number of discriminator train iterations.
        """
        trainer_type = self._args.trainer_type

        if trainer_type == "vanilla_gan":
            iter_num = 1
        elif trainer_type == "temporal_gan":
            iter_num = 1
        else:
            raise RuntimeError("Unvalid trainer type: {}".format(trainer_type))
        return iter_num

    # pylint: disable=R0912, R0914, R0915
    def train(self) -> None:
        """
        Train overall model.
        """
        self.logger.log(self.generator_trainer)
        for trainer in self.discriminator_trainers:
            self.logger.log(trainer)

        save_init_model = self._args.snapshot_init
        log_interval = self._args.print_every
        save_interval = self._args.save_every
        train_iterations = self._args.n_iterations
        show_interval = self._args.show_data_interval
        show_data = show_interval > 0

        start_time = time.time()
        logs = util.LoggerDict()
        safeguard = util.SafeGuardKiller()

        self.logger.log("Starting/restored iteration: {}".format(self.generator_trainer.gen_iterations))
        self.logger.log(
            "{} is cuda: {}".format(
                self.generator_trainer.generator.__class__.__name__,
                next(self.generator_trainer.generator.parameters()).is_cuda,
            ),
            level=logging.INFO,
        )

        for discriminator in self.discriminator_trainers:
            self.logger.log(
                'Number of {} parameters" {}'.format(
                    discriminator.discriminator.__class__.__name__,
                    sum(p.numel() for p in discriminator.discriminator.parameters()),
                )
            )
            self.logger.log(
                "{} is cuda: {}".format(
                    discriminator.discriminator.__class__.__name__,
                    next(discriminator.discriminator.parameters()).is_cuda,
                ),
                level=logging.INFO,
            )

        self.logger.log("Use CuDNN benchmarking: {}".format(torch.backends.cudnn.benchmark))
        self.logger.log("Use CuDNN deterministic: {}".format(torch.backends.cudnn.deterministic))
        self.logger.log("Use CuDNN: {}".format(torch.backends.cudnn.enabled))

        if save_init_model:
            self.logger.log("Saving initinal snapshot ...")
            self.save_models(self._args.log_folder, 0)

        for batch_num in range(self.generator_trainer.gen_iterations + 1, train_iterations + 1):
            # Set all to train mode
            self.generator_trainer.generator.train()
            for discriminator in self.discriminator_trainers:
                discriminator.discriminator.train()

            # train discriminators
            disc_iterations = self._get_disc_iter()
            self.logger.log("Pretraining discriminators: {} iterations".format(disc_iterations))

            for _ in tqdm(range(disc_iterations), total=disc_iterations):
                for discriminator in self.discriminator_trainers:
                    result_dict = discriminator.train_disc()
                    logs.update(result_dict)

            # train generator
            gen_losses = [disc.calculate_generator_criterion() for disc in self.discriminator_trainers]
            result_dict = self.generator_trainer.train_generator(gen_losses)
            logs.update(result_dict)

            if batch_num % log_interval == 0:
                add_string = "\n"
                log_string = "Batch %d" % batch_num
                for k, v in logs.items():
                    if not isinstance(v, dict):
                        log_string += " [%s] %5.3f" % (k, v / log_interval)
                    else:
                        for hist_tag, hist_val in v.items():
                            add_string += "Min, max, mean {} :: {} {} {} \n".format(
                                hist_tag, torch.min(hist_val), torch.max(hist_val), torch.mean(hist_val)
                            )
                log_string += ". Took %5.2f" % (time.time() - start_time)
                self.logger.log(log_string)
                self.logger.log(add_string)

                for tag, value in list(logs.items()):
                    if isinstance(value, dict):
                        for hist_tag, hist_val in value.items():
                            self.logger.histogram_summary(hist_tag, hist_val, batch_num)
                    else:
                        self.logger.scalar_summary(tag, value / log_interval, batch_num)

                start_time = time.time()
                logs = util.LoggerDict()

            if show_data and (batch_num % show_interval == 0):
                self.save_examples(batch_num)

            if (batch_num % save_interval == 0) or (batch_num == train_iterations):
                self.save_models(self._args.log_folder, batch_num)

            if safeguard.kill_now:
                self.save_models(self._args.log_folder, batch_num)
                break
