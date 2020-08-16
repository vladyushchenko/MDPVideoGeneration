"""
Copyright (C) 2017 NVIDIA Corporation.

All rights reserved.
Licensed under the CC BY-NC-ND 4.0 license (https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode).
"""
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable

from mdp_video.util import chunks

if torch.cuda.is_available():
    T = torch.cuda
else:
    T = torch


class Noise(nn.Module):
    """
    Noise layer.
    """

    def __init__(self, use_noise: bool, sigma: float = 0.2) -> None:
        """
        Init call.

        :param use_noise:
        :param sigma:
        """
        super(Noise, self).__init__()
        self.use_noise = use_noise
        self.sigma = sigma

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        """
        if self.use_noise:
            return data + self.sigma * Variable(torch.empty_like(data).normal_(), requires_grad=False)
        return data


class ImageDiscriminator(nn.Module):
    """
    2D Image discriminator base class.
    """

    # pylint: disable=R0913
    def __init__(
        self, n_channels: int, dim_categorical: int = 0, ndf: int = 64, use_noise: bool = False, noise_sigma: float = 0
    ):
        """
        Init call.

        :param n_channels:
        :param dim_categorical:
        :param ndf:
        :param use_noise:
        :param noise_sigma:
        """
        super(ImageDiscriminator, self).__init__()

        self.use_noise = use_noise

        self.features = nn.Sequential(
            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(n_channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.binary_class = nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)
        self.label_class = None
        if dim_categorical > 0:
            self.label_class = nn.Conv2d(ndf * 8, dim_categorical, 4, 1, 0, bias=False)

    def forward(self, data: torch.Tensor) -> Tuple:
        """
        Forward  pass.
        """
        features = self.features(data)
        binary_decision = self.binary_class(features).squeeze()
        categorical = self.label_class(features).squeeze() if self.label_class is not None else None
        return binary_decision, categorical


class MergedDiscriminator(ImageDiscriminator):
    """
    Merged discriminator Image+Video.
    """

    # pylint: disable=R0913
    def __init__(
        self,
        step_size: int,
        dim_categorical: int,
        n_channels: int = 3,
        ndf: int = 64,
        use_noise: bool = False,
        noise_sigma: float = 0.0,
    ):
        """
        Init call.

        :param step_size:
        :param dim_categorical:
        :param n_channels:
        :param ndf:
        :param use_noise:
        :param noise_sigma:
        """
        super().__init__(
            n_channels=n_channels * step_size,
            dim_categorical=dim_categorical,
            ndf=ndf,
            use_noise=use_noise,
            noise_sigma=noise_sigma,
        )

    def forward(self, data: torch.Tensor) -> Tuple:
        """
        Forward pass.
        """
        disc_h, disc_cat = [], []
        for chunk in data:
            h, cat = super().forward(chunk)
            disc_h.append(h)
            disc_cat.append(cat)

        disc_h = torch.stack(disc_h)
        disc_cat = torch.cat(disc_cat) if any((isinstance(item, torch.Tensor) for item in disc_cat)) else None
        return disc_h, disc_cat


class PatchImageDiscriminator(nn.Module):
    """
    PatchImageDiscriminator for nocat training.
    """

    def __init__(self, n_channels: int, ndf: int = 64, use_noise: bool = False, noise_sigma: float = 0.0):
        """
        Init call.

        :param n_channels:
        :param ndf:
        :param use_noise:
        :param noise_sigma:
        """
        super(PatchImageDiscriminator, self).__init__()

        self.use_noise = use_noise

        self.main = nn.Sequential(
            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(n_channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(ndf * 4, 1, 4, 2, 1, bias=False),
        )

    def forward(self, data: torch.Tensor) -> Tuple:
        """
        Forward pass.
        """
        h = self.main(data).squeeze()
        return h, None


class PatchVideoDiscriminator(nn.Module):
    """
    PatchVideoDiscriminator for nocat training.
    """

    # pylint: disable=R0913
    def __init__(
        self,
        n_channels: int,
        n_output_neurons: int = 1,
        bn_use_gamma: bool = True,
        use_noise: bool = False,
        noise_sigma: float = 0.0,
        ndf: int = 64,
    ) -> None:
        """
        Init call.

        :param n_channels:
        :param n_output_neurons:
        :param bn_use_gamma:
        :param use_noise:
        :param noise_sigma:
        :param ndf:
        """
        super(PatchVideoDiscriminator, self).__init__()

        self.n_channels = n_channels
        self.n_output_neurons = n_output_neurons
        self.use_noise = use_noise
        self.bn_use_gamma = bn_use_gamma

        self.main = nn.Sequential(
            Noise(use_noise, sigma=noise_sigma),
            nn.Conv3d(n_channels, ndf, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            Noise(use_noise, sigma=noise_sigma),
            nn.Conv3d(ndf, ndf * 2, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            Noise(use_noise, sigma=noise_sigma),
            nn.Conv3d(ndf * 2, ndf * 4, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(ndf * 4, 1, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
        )

    def forward(self, data: torch.Tensor) -> Tuple:
        """
        Forward pass.
        """
        h = self.main(data).squeeze()

        return h, None


class VideoDiscriminator(nn.Module):
    """
    VideoDiscriminator for nocat and base class for categories.
    """

    # pylint: disable=R0913
    def __init__(
        self,
        n_channels: int,
        n_output_neurons: int = 1,
        bn_use_gamma: bool = True,
        use_noise: bool = False,
        noise_sigma: float = 0.0,
        ndf: int = 64,
    ) -> None:
        """
        Init call.

        :param n_channels:
        :param n_output_neurons:
        :param bn_use_gamma:
        :param use_noise:
        :param noise_sigma:
        :param ndf:
        """
        super(VideoDiscriminator, self).__init__()

        self.n_channels = n_channels
        self.n_output_neurons = n_output_neurons
        self.use_noise = use_noise
        self.bn_use_gamma = bn_use_gamma

        self.avg_pool = nn.AdaptiveAvgPool1d(output_size=1)

        self.main = nn.Sequential(
            Noise(use_noise, sigma=noise_sigma),
            nn.Conv3d(n_channels, ndf, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            Noise(use_noise, sigma=noise_sigma),
            nn.Conv3d(ndf, ndf * 2, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            Noise(use_noise, sigma=noise_sigma),
            nn.Conv3d(ndf * 2, ndf * 4, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            Noise(use_noise, sigma=noise_sigma),
            nn.Conv3d(ndf * 4, ndf * 8, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(ndf * 8, n_output_neurons, 4, 1, 0, bias=False),
        )

    def forward(self, data: torch.Tensor) -> Tuple:
        """
        Forward pass.
        """
        # batch_size * n_channels * time * x_dim * y_dim
        h = self.main(data).squeeze()

        return h, None


class CategoricalVideoDiscriminator(VideoDiscriminator):
    """
    VideoDiscriminator with categories.
    """

    # pylint: disable=R0913
    def __init__(
        self,
        n_channels: int,
        dim_categorical: int,
        n_output_neurons: int = 1,
        use_noise: bool = False,
        noise_sigma: float = 0.0,
    ) -> None:
        """
        Init call.

        :param n_channels:
        :param dim_categorical:
        :param n_output_neurons:
        :param use_noise:
        :param noise_sigma:
        """
        super(CategoricalVideoDiscriminator, self).__init__(
            n_channels=n_channels,
            n_output_neurons=n_output_neurons + dim_categorical,
            use_noise=use_noise,
            noise_sigma=noise_sigma,
        )

        self.dim_categorical = dim_categorical

    def split(self, data: torch.Tensor) -> Tuple:
        """
        Split data.

        :param data: input tensor
        :return: split tensor
        """
        return data[:, : data.size(1) - self.dim_categorical], data[:, data.size(1) - self.dim_categorical :]

    def forward(self, data: torch.Tensor) -> Tuple:
        """
        Forward pass.
        """
        h, _ = super(CategoricalVideoDiscriminator, self).forward(data)
        labels, categ = self.split(h)
        return labels, categ


class VideoGenerator(nn.Module):
    """
    Mocogan video generator.
    """

    # pylint: disable=R0913
    def __init__(
        self,
        n_channels: int,
        dim_z_content: int,
        dim_z_category: int,
        dim_z_motion: int,
        video_length: int,
        merged_step_size: int,
        ngf: int = 64,
    ) -> None:
        """
        Init call.

        :param n_channels:
        :param dim_z_content:
        :param dim_z_category:
        :param dim_z_motion:
        :param video_length:
        :param merged_step_size:
        :param ngf:
        """
        super(VideoGenerator, self).__init__()

        self.n_channels = n_channels
        self.dim_z_content = dim_z_content
        self.dim_z_category = dim_z_category
        self.dim_z_motion = dim_z_motion
        self.video_length = video_length
        self.merged_step_size = merged_step_size

        dim_z = dim_z_motion + dim_z_category + dim_z_content

        self.recurrent = nn.GRUCell(dim_z_motion, dim_z_motion)

        self.main = nn.Sequential(
            nn.ConvTranspose2d(dim_z, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, self.n_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def sample_z_m(self, num_samples: int, video_len: int = 0) -> torch.Tensor:
        """
        Sample latent video motion.
        """
        video_len = video_len if video_len is not None else self.video_length

        h_t = [self.get_gru_initial_state(num_samples)]

        for _ in range(video_len):
            e_t = self.get_iteration_noise(num_samples)
            h_t.append(self.recurrent(e_t, h_t[-1]))

        z_m_t = [h_k.view(-1, 1, self.dim_z_motion) for h_k in h_t]
        z_m = torch.cat(z_m_t[1:], dim=1).view(-1, self.dim_z_motion)

        return z_m

    def sample_z_categ(self, num_samples: int, video_len: int) -> Tuple:
        """
        Sample latent video category.
        """
        video_len = video_len if video_len is not None else self.video_length

        classes_to_generate = np.random.randint(self.dim_z_category, size=num_samples)
        one_hot = np.zeros((num_samples, self.dim_z_category), dtype=np.float32)
        one_hot[np.arange(num_samples), classes_to_generate] = 1
        one_hot_video = np.repeat(one_hot, video_len, axis=0)

        one_hot_video = torch.from_numpy(one_hot_video)

        if torch.cuda.is_available():
            one_hot_video = one_hot_video.cuda()

        return Variable(one_hot_video), classes_to_generate

    def sample_z_content(self, num_samples: int, video_len: int = 0) -> torch.Tensor:
        """
        Sample latent video content.
        """
        video_len = video_len if video_len is not None else self.video_length

        content = np.random.normal(0, 1, (num_samples, self.dim_z_content)).astype(np.float32)
        content = np.repeat(content, video_len, axis=0)
        content = torch.from_numpy(content)
        if torch.cuda.is_available():
            content = content.cuda()
        return Variable(content)

    def sample_z_video(self, num_samples: int, video_len: int = 0) -> Tuple:
        """
        Sample latent video variable.
        """
        z_content = self.sample_z_content(num_samples, video_len)
        z_motion = self.sample_z_m(num_samples, video_len)

        if self.dim_z_category > 0:
            z_category, z_category_labels = self.sample_z_categ(num_samples, video_len)
            z = torch.cat([z_content, z_category, z_motion], dim=1)
        else:
            z_category_labels = None
            z = torch.cat([z_content, z_motion], dim=1)

        return z, z_category_labels

    def sample_videos(self, num_samples: int, video_len: int = 0, seed: Optional[int] = None) -> Tuple:
        """
        Get samples for video discriminator.
        """
        if seed is not None:
            np.random.seed(seed)
            torch.cuda.manual_seed(seed)
        video_len = video_len if video_len is not None else self.video_length

        z, z_category_labels = self.sample_z_video(num_samples, video_len)

        h = self.main(z.view(z.size(0), z.size(1), 1, 1))
        h = h.view(h.size(0) // video_len, video_len, self.n_channels, h.size(3), h.size(3))
        h = h.permute(0, 2, 1, 3, 4).contiguous()

        if z_category_labels is not None:
            with torch.no_grad():
                z_category_labels = torch.from_numpy(z_category_labels)
                if torch.cuda.is_available():
                    z_category_labels = z_category_labels.cuda()

        return h, z_category_labels

    def sample_merged(self, num_samples: int, chunk_size: int = 0, video_len: int = 0) -> Tuple:
        """
        Get samples for merged discriminator.
        """
        chunk_size = chunk_size if chunk_size is not None else self.merged_step_size
        video_len = video_len if video_len is not None else self.video_length
        samples, category = self.sample_videos(num_samples=num_samples, video_len=video_len)
        samples = samples.squeeze()

        split_tensors = torch.split(samples, 1, dim=2)
        new_samples, new_cat = [], []
        for items in chunks(split_tensors, chunk_size):
            concatenated = torch.cat(items, dim=1).squeeze()
            new_samples.append(concatenated)
            new_cat.append(category)

        new_samples = torch.stack(new_samples).permute(1, 0, 2, 3, 4).contiguous()
        new_cat = torch.cat(new_cat) if any((isinstance(item, torch.Tensor) for item in new_cat)) else None
        return new_samples, new_cat

    def sample_images(self, num_samples: int) -> Tuple:
        """
        Get samples for image discriminator.
        """
        z, _ = self.sample_z_video(num_samples * self.video_length * 2)

        j = np.sort(np.random.choice(z.size(0), num_samples, replace=False)).astype(np.int64)
        z = z[j, ::]
        z = z.view(z.size(0), z.size(1), 1, 1)
        h = self.main(z)

        return h, None

    def get_gru_initial_state(self, num_samples: int) -> torch.Tensor:
        """
        Get initial state.

        :param num_samples: number of samples
        :return: initial state
        """
        return Variable(T.FloatTensor(num_samples, self.dim_z_motion).normal_())

    def get_iteration_noise(self, num_samples: int) -> torch.Tensor:
        """
        Get iteration noise.

        :param num_samples: number of samples
        :return: noise tensor
        """
        return Variable(T.FloatTensor(num_samples, self.dim_z_motion).normal_())


class NormalizedVideoGenerator(nn.Module):
    """
    Mocogan video generator.
    """

    # pylint: disable=R0913
    def __init__(
        self,
        n_channels: int,
        dim_z_content: int,
        dim_z_category: int,
        dim_z_motion: int,
        video_length: int,
        merged_step_size: int,
        ngf: int = 64,
    ) -> None:
        """
        Init call.

        :param n_channels:
        :param dim_z_content:
        :param dim_z_category:
        :param dim_z_motion:
        :param video_length:
        :param merged_step_size:
        :param ngf:
        """
        super(NormalizedVideoGenerator, self).__init__()

        self.n_channels = n_channels
        self.dim_z_content = dim_z_content
        self.dim_z_category = dim_z_category
        self.dim_z_motion = dim_z_motion
        self.video_length = video_length
        self.merged_step_size = merged_step_size

        dim_z = dim_z_motion + dim_z_category + dim_z_content

        self.recurrent = nn.GRUCell(dim_z_motion, dim_z_motion)
        self.mu_layer = nn.Linear(dim_z_motion, dim_z_motion)
        self.log_var_layer = nn.Linear(dim_z_motion, dim_z_motion)

        self.main = nn.Sequential(
            nn.ConvTranspose2d(dim_z, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, self.n_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    # pylint: disable=R0914
    def sample_z_m(self, num_samples: int, video_len: int = 0) -> Tuple:
        """
        Sample latent video motion.
        """
        video_len = video_len if video_len is not None else self.video_length

        h_init = self.get_gru_initial_state(num_samples)
        m = self.mu_layer(h_init)
        log_var = self.log_var_layer(h_init)
        z_init = self.reparameterize(m, log_var)
        h_t = [z_init]
        loss = [self.kl_divergence(m, log_var)]

        for _ in range(video_len):
            e_t = self.get_iteration_noise(num_samples)
            new_h = self.recurrent(e_t, h_t[-1])
            new_m = self.mu_layer(new_h)
            new_log_var = self.log_var_layer(new_h)
            new_z = self.reparameterize(new_m, new_log_var)
            h_t.append(new_z)
            loss.append(self.kl_divergence(new_m, new_log_var))

        z_m_t = [h_k.view(-1, 1, self.dim_z_motion) for h_k in h_t]
        z_m = torch.cat(z_m_t[1:], dim=1).view(-1, self.dim_z_motion)

        return z_m, sum(loss) / len(loss)

    @staticmethod
    def reparameterize(mu_val: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Do reparametrization trick.

        :param mu_val:
        :param logvar:
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu_val)

    @staticmethod
    def kl_divergence(mu_val: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Calculate KL divergence.

        :param mu_val:
        :param logvar:
        :return:
        """
        return -0.5 * torch.mean(1 + logvar - mu_val.pow(2) - logvar.exp())

    def sample_z_categ(self, num_samples: int, video_len: int) -> Tuple:
        """
        Sample latent video category.
        """
        video_len = video_len if video_len is not None else self.video_length

        classes_to_generate = np.random.randint(self.dim_z_category, size=num_samples)
        one_hot = np.zeros((num_samples, self.dim_z_category), dtype=np.float32)
        one_hot[np.arange(num_samples), classes_to_generate] = 1
        one_hot_video = np.repeat(one_hot, video_len, axis=0)

        one_hot_video = torch.from_numpy(one_hot_video)

        if torch.cuda.is_available():
            one_hot_video = one_hot_video.cuda()

        return Variable(one_hot_video), classes_to_generate

    def sample_z_content(self, num_samples: int, video_len: int = 0) -> torch.Tensor:
        """
        Sample latent video content.
        """
        video_len = video_len if video_len is not None else self.video_length

        content = np.random.normal(0, 1, (num_samples, self.dim_z_content)).astype(np.float32)
        content = np.repeat(content, video_len, axis=0)
        content = torch.from_numpy(content)
        if torch.cuda.is_available():
            content = content.cuda()
        return Variable(content)

    def sample_z_video(self, num_samples: int, video_len: int = 0) -> Tuple:
        """
        Sample latent video variable.
        """
        z_content = self.sample_z_content(num_samples, video_len)
        z_motion, loss = self.sample_z_m(num_samples, video_len)

        if self.dim_z_category > 0:
            z_category, z_category_labels = self.sample_z_categ(num_samples, video_len)
            z = torch.cat([z_content, z_category, z_motion], dim=1)
        else:
            z_category_labels = None
            z = torch.cat([z_content, z_motion], dim=1)

        return z, (loss, z_category_labels)

    def sample_videos(self, num_samples: int, video_len: int = 0, seed: Optional[int] = None) -> Tuple:
        """
        Get samples for video discriminator.
        """
        if seed is not None:
            np.random.seed(seed)
            torch.cuda.manual_seed(seed)
        video_len = video_len if video_len is not None else self.video_length

        z, (dl_loss, z_category_labels) = self.sample_z_video(num_samples, video_len)

        h = self.main(z.view(z.size(0), z.size(1), 1, 1))
        h = h.view(h.size(0) // video_len, video_len, self.n_channels, h.size(3), h.size(3))
        h = h.permute(0, 2, 1, 3, 4).contiguous()

        if z_category_labels is not None:
            with torch.no_grad():
                z_category_labels = torch.from_numpy(z_category_labels)
                if torch.cuda.is_available():
                    z_category_labels = z_category_labels.cuda()

        return h, (dl_loss, z_category_labels)

    def sample_merged(self, num_samples: int, chunk_size: Optional[int] = None, video_len: int = 0) -> Tuple:
        """
        Get samples for merged discriminator.
        """
        chunk_size = chunk_size if chunk_size is not None else self.merged_step_size
        video_len = video_len if video_len is not None else self.video_length
        samples, category = self.sample_videos(num_samples=num_samples, video_len=video_len)
        samples = samples.squeeze()

        split_tensors = torch.split(samples, 1, dim=2)
        new_samples, new_cat = [], []
        for items in chunks(split_tensors, chunk_size):
            concatenated = torch.cat(items, dim=1).squeeze()
            new_samples.append(concatenated)
            new_cat.append(category)

        new_samples = torch.stack(new_samples).permute(1, 0, 2, 3, 4).contiguous()
        new_cat = torch.cat(new_cat) if any((isinstance(item, torch.Tensor) for item in new_cat)) else None
        return new_samples, new_cat

    def sample_images(self, num_samples: int) -> Tuple:
        """
        Get samples for image discriminator.
        """
        z, z_category_labels = self.sample_z_video(num_samples * self.video_length * 2)

        j = np.sort(np.random.choice(z.size(0), num_samples, replace=False)).astype(np.int64)
        z = z[j, ::]
        z = z.view(z.size(0), z.size(1), 1, 1)
        h = self.main(z)

        return h, z_category_labels

    def get_gru_initial_state(self, num_samples: int) -> torch.Tensor:
        """
        Get initial state.

        :param num_samples: number of samples
        :return: initial state
        """
        return Variable(T.FloatTensor(num_samples, self.dim_z_motion).normal_())

    def get_iteration_noise(self, num_samples: int) -> torch.Tensor:
        """
        Get iteration noise.

        :param num_samples: number of samples
        :return: noise tensor
        """
        return Variable(T.FloatTensor(num_samples, self.dim_z_motion).normal_())
