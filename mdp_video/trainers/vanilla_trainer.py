import os
from typing import Any, Dict

import torch
import torch.optim as optim
from torch import nn
from torch.autograd import Variable
import torchviz

import mdp_video.util as util

if torch.cuda.is_available():
    T = torch.cuda
else:
    T = torch


class VanillaDiscriminatorTrainer:
    """
    Combines DataLoader and Discriminator model for training.
    """

    def __init__(self, discriminator: Any, data_sampler: Any, generator_sampler: Any, named_args: Any) -> None:
        self.discriminator = discriminator
        self._real_sampler = data_sampler
        self._fake_sampler = generator_sampler
        self._opt = optim.Adam(
            self.discriminator.parameters(), lr=named_args.learning_rate, betas=(0.5, 0.999), weight_decay=0.00001
        )
        self._batch_size = self._real_sampler.batch_size
        self._named_args = named_args
        self._gan_criterion = nn.BCEWithLogitsLoss()
        self._category_criterion = nn.CrossEntropyLoss()

    def __repr__(self) -> str:
        """
        Get repr str.
        """
        return "{} \n {}".format(self.discriminator, self._opt)

    # pylint: disable=R0914
    def train_disc(self) -> Dict:
        """
        Train discriminator.
        """
        for param in self.discriminator.parameters():
            param.requires_grad = True

        self.discriminator.train()
        self._opt.zero_grad()

        real_batch, real_categories = self._real_sampler()
        real_batch = Variable(real_batch, requires_grad=False)
        fake_batch, _ = self._fake_sampler(self._batch_size)

        real_labels, real_categorical = self.discriminator(real_batch)
        fake_labels, _ = self.discriminator(fake_batch.detach())

        ones = torch.ones_like(real_labels)
        zeros = torch.zeros_like(fake_labels)

        l_discriminator = self._gan_criterion(real_labels, ones) + self._gan_criterion(fake_labels, zeros)

        if self._named_args.use_categories:
            # The video discriminator includes loss of real categories vs discriminator categories of real images
            categories_gt = Variable(torch.squeeze(real_categories.long()), requires_grad=False)
            l_discriminator += self._category_criterion(real_categorical.squeeze(), categories_gt)

        l_discriminator.backward()
        self._opt.step()
        loss_val = l_discriminator.item()

        if self._named_args.save_graph:
            comp_graph_folder = os.path.join(self._named_args.log_folder, "ComputationalGraph")
            torchviz.make_dot(l_discriminator, dict(self.discriminator.named_parameters())).view(
                self.discriminator.__class__.__name__, comp_graph_folder
            )
            self._named_args.save_graph = False

        return {self.discriminator.__class__.__name__: loss_val}

    def calculate_generator_criterion(self) -> Dict:
        """
        Calculate criterion for generator trainer.
        """
        for param in self.discriminator.parameters():
            param.requires_grad = False

        generated_batch, generated_categories = self._fake_sampler(self._batch_size)
        fake_labels, fake_categorical = self.discriminator(generated_batch)
        all_ones = torch.ones_like(fake_labels)
        val_criterion = self._gan_criterion(fake_labels, all_ones)
        kl_loss = T.FloatTensor([0])

        if isinstance(generated_categories, tuple):
            kl_loss = generated_categories[0].item()
            generated_categories = generated_categories[1]

        if self._named_args.use_infogan:
            val_criterion += self._category_criterion(fake_categorical.squeeze(), generated_categories.long())

        return {
            "{}_orig_gen".format(self.discriminator.__class__.__name__): val_criterion,
            "{}_kl_gen".format(self.discriminator.__class__.__name__): kl_loss,
        }


class VanillaGeneratorTrainer:
    """
    Combine DataLoader and Generator model for training.
    """

    def __init__(self, generator: Any, named_args: Any, gen_iterations: int = 0) -> None:
        """
        Init call.

        :param generator:
        :param named_args:
        :param gen_iterations:
        """
        self.gen_named_args = named_args
        self.generator = generator
        self._opt_generator = optim.Adam(
            self.generator.parameters(), lr=named_args.learning_rate, betas=(0.5, 0.999), weight_decay=0.00001
        )
        self.gen_iterations = gen_iterations

    def __repr__(self) -> str:
        """
        Get repr str.
        """
        return "{} \n {}".format(self.generator, self._opt_generator)

    def train_generator(self, gen_losses: Any) -> Dict:
        """
        Train generator.
        """
        self.generator.train()
        self._opt_generator.zero_grad()
        l_generator = T.FloatTensor([0])
        dist_dict = {}

        for item in gen_losses:
            for key, value in item.items():
                l_generator += value
                dist_dict[key] = value

        l_generator.backward(retain_graph=self.gen_named_args.save_graph)
        self._opt_generator.step()
        loss_val = l_generator.item()

        if self.gen_named_args.save_graph:
            comp_graph_folder = os.path.join(self.gen_named_args.log_folder, "ComputationalGraph")
            torchviz.make_dot(l_generator).view(self.generator.__class__.__name__, comp_graph_folder)
            self.gen_named_args.save_graph = False

        return {self.generator.__class__.__name__: loss_val, **dist_dict}

    def save_sample_video(self, batch_counter: int) -> None:
        """
        Save training samples to disk.
        """
        max_videos = 3
        fake_batch, _ = self.generator.sample_videos(max_videos)
        filename = "Iteration_{}.jpg".format(batch_counter)
        util.show_batch(fake_batch.data, title=filename, max_videos=max_videos)
