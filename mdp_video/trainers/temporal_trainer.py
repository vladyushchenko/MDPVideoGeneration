import os
from typing import Any, Dict, List

import torch
import torch.optim as optim
from torch import nn
from torch.autograd import Variable

import torchviz

from mdp_video.util import show_batch

if torch.cuda.is_available():
    T = torch.cuda
else:
    T = torch


class TemporalDiscriminatorTrainer:
    """
    Get DataLoader and Discriminator model for TCN.
    """

    def __init__(self, discriminator: Any, data_sampler: Any, generator_sampler: Any, named_args: Any) -> None:
        """
        Init call.
        """
        self.discriminator = discriminator
        self._real_sampler = data_sampler
        self._fake_sampler = generator_sampler
        self._opt = optim.Adam(
            self.discriminator.parameters(), lr=named_args.learning_rate, betas=(0.5, 0.999), weight_decay=0.00001
        )
        self._batch_size = self._real_sampler.batch_size
        self._named_args = named_args
        self._gan_criterion = nn.BCEWithLogitsLoss()

        # BUG ALERT: elementwise reduction pytorch
        # https://github.com/pytorch/pytorch/issues/12901
        self._temporal_disc_criterion = lambda x, y: torch.pow(x - y, 2)

        self._category_criterion = nn.CrossEntropyLoss()
        self._use_cuda = torch.cuda.is_available()

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

        score_real_disc, score_real_gen, real_categorical = self.discriminator(real_batch)
        score_fake_disc, score_fake_gen, _ = self.discriminator(fake_batch.detach())

        ones = torch.ones_like(score_real_disc)
        zeros = torch.zeros_like(score_real_disc)

        # Calculate standard GAN loss
        l_disc_orig = self._gan_criterion(score_real_disc, ones) + self._gan_criterion(score_fake_disc, zeros)

        if self._named_args.use_categories:
            # The video discriminator includes loss of real categories vs discriminator categories of real images
            categories_gt = Variable(torch.squeeze(real_categories.long()), requires_grad=False)
            l_disc_orig += self._category_criterion(real_categorical.squeeze(), categories_gt)

        # Calculate temporal loss for g_t
        score_real_weighted_disc = self.calculate_weighted_disc_score(score_real_disc)
        l_disc_real_temp = torch.mean(self._temporal_disc_criterion(score_real_gen, score_real_weighted_disc))

        score_fake_weighted_disc = self.calculate_weighted_disc_score(score_fake_disc)
        l_disc_fake_temp = torch.mean(self._temporal_disc_criterion(score_fake_gen, score_fake_weighted_disc))

        l_disc_temp = l_disc_real_temp + l_disc_fake_temp
        l_discriminator = l_disc_orig + l_disc_temp
        l_discriminator.backward()
        self._opt.step()
        loss_val = l_discriminator.item()
        loss_orig = l_disc_orig.item()
        loss_real_temp = l_disc_real_temp.item()
        loss_fake_temp = l_disc_fake_temp.item()
        loss_temp = l_disc_temp.item()

        if self._named_args.save_graph:
            comp_graph_folder = os.path.join(self._named_args.log_folder, "ComputationalGraph")
            torchviz.make_dot(l_discriminator, dict(self.discriminator.named_parameters())).view(
                self.discriminator.__class__.__name__, comp_graph_folder
            )
            self._named_args.save_graph = False

        hist_dict = {
            "score_fake_disc": score_fake_disc,
            "score_real_disc": score_real_disc,
            "score_fake_weighted_disc": score_fake_weighted_disc,
            "score_real_weighted_disc": score_real_weighted_disc,
            "score_fake_gen": score_fake_gen,
            "score_real_gen": score_real_gen,
        }

        return {
            self.discriminator.__class__.__name__: loss_val,
            "T_orig": loss_orig,
            "T_temp": loss_temp,
            "T_real_temp": loss_real_temp,
            "T_fake_temp": loss_fake_temp,
            "hist": hist_dict,
        }

    def calculate_weighted_disc_score(self, score_disc: torch.Tensor) -> torch.Tensor:
        """
        Calculate weighted generator score from discriminator score.
        """
        weights = torch.arange(0, self.discriminator.video_length, dtype=torch.float)
        weights = torch.pow(torch.FloatTensor([self._named_args.temporal_sigma]), weights)

        weights_mat = torch.zeros((self.discriminator.video_length, self.discriminator.video_length), dtype=torch.float)

        for i in range(self.discriminator.video_length):
            n_items = self.discriminator.video_length - i
            weights_mat[i:, i] = weights[:n_items] / n_items

        if self._use_cuda:
            weights_mat = weights_mat.cuda()

        temp_disc = score_disc.permute(0, 1, 3, 4, 2)
        result = torch.matmul(temp_disc, weights_mat)
        result = result.permute(0, 1, 4, 2, 3).contiguous()

        return result

    def calculate_generator_criterion(self) -> Dict:
        """
        Calculate criterion for generator trainer.
        """
        for param in self.discriminator.parameters():
            param.requires_grad = False

        generated_batch, generated_categories = self._fake_sampler(self._batch_size)
        score_fake_disc, score_fake_gen, fake_categorical = self.discriminator(generated_batch)
        kl_loss = T.FloatTensor([0])

        if isinstance(generated_categories, tuple):
            kl_loss = generated_categories[0].item()
            generated_categories = generated_categories[1]

        all_ones = torch.ones_like(score_fake_disc)
        val_criterion = self._gan_criterion(score_fake_disc, all_ones)

        if self._named_args.use_infogan:
            val_criterion += self._category_criterion(fake_categorical.squeeze(), generated_categories.long())

        # Temporal generator criterion
        score_fake_gen = score_fake_gen.permute(2, 0, 1, 3, 4).contiguous()
        weight_score_fake_gen = self.calculate_weighted_gen_loss(score_fake_gen)

        temporal_loss = -torch.sum(weight_score_fake_gen)

        return {
            "{}_orig_gen".format(self.discriminator.__class__.__name__): val_criterion,
            "{}_temp_loss_gen".format(self.discriminator.__class__.__name__): temporal_loss,
            "{}_kl_loss_gen".format(self.discriminator.__class__.__name__): kl_loss,
        }

    def calculate_weighted_gen_loss(self, gen_pred_loss: torch.Tensor) -> torch.Tensor:
        """
        Calculate weighted generator loss.
        """
        weights = torch.arange(1, self.discriminator.video_length + 1, dtype=torch.float)
        weights = torch.pow(torch.FloatTensor([self._named_args.temporal_beta]), weights)

        if self._use_cuda:
            weights = weights.cuda()

        weights = weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        weights = weights.expand_as(gen_pred_loss)
        result = weights * gen_pred_loss

        return result


class TemporalGeneratorTrainer:
    """
    Combine DataLoader and Generator model for training.
    """

    def __init__(self, generator: Any, named_args: Any, gen_iterations: int = 0) -> None:
        """
        Init call.
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

    def train_generator(self, gen_losses: List[Dict]) -> Dict:
        """
        Train generator.
        """
        self.generator.train()
        self._opt_generator.zero_grad()
        l_generator = T.FloatTensor([0])

        dist_dict = {}

        for item in gen_losses:
            for key, value in item.items():
                if not isinstance(value, dict):
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
        show_batch(fake_batch.data, title=filename, max_videos=max_videos)
