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


class NoiseLayer(nn.Module):
    """
    Noise layer.
    """

    def __init__(self, use_noise: bool, sigma: float = 0.2) -> None:
        """
        Init call.

        :param use_noise:
        :param sigma:
        """
        super().__init__()
        self.use_noise = use_noise
        self.sigma = sigma

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        """
        if self.use_noise:
            return data + self.sigma * Variable(torch.empty_like(data).normal_(), requires_grad=False)
        return data


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
        super().__init__()

        self.use_noise = use_noise

        self.main = nn.Sequential(
            NoiseLayer(use_noise, sigma=noise_sigma),
            nn.Conv2d(n_channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            NoiseLayer(use_noise, sigma=noise_sigma),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            NoiseLayer(use_noise, sigma=noise_sigma),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            NoiseLayer(use_noise, sigma=noise_sigma),
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
        super().__init__()

        self.n_channels = n_channels
        self.n_output_neurons = n_output_neurons
        self.use_noise = use_noise
        self.bn_use_gamma = bn_use_gamma

        self.main = nn.Sequential(
            NoiseLayer(use_noise, sigma=noise_sigma),
            nn.Conv3d(n_channels, ndf, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            NoiseLayer(use_noise, sigma=noise_sigma),
            nn.Conv3d(ndf, ndf * 2, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            NoiseLayer(use_noise, sigma=noise_sigma),
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
        merged_step_size: int = 0,
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
        super().__init__()

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
        video_len = video_len if video_len else self.video_length

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
        video_len = video_len if video_len else self.video_length

        classes = np.random.randint(self.dim_z_category, size=num_samples)
        one_hot = np.zeros((num_samples, self.dim_z_category), dtype=np.float32)
        one_hot[np.arange(num_samples), classes] = 1
        one_hot_video = np.repeat(one_hot, video_len, axis=0)

        one_hot_video = torch.from_numpy(one_hot_video)

        if torch.cuda.is_available():
            one_hot_video = one_hot_video.cuda()

        return Variable(one_hot_video), classes

    def sample_z_content(self, num_samples: int, video_len: int = 0) -> torch.Tensor:
        """
        Sample latent video content.
        """
        video_len = video_len if video_len else self.video_length

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
        video_len = video_len if video_len else self.video_length

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
        chunk_size = chunk_size if chunk_size else self.merged_step_size
        video_len = video_len if video_len else self.video_length
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
