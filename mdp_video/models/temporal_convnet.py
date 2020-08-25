# MIT License
#
# Copyright (c) 2018 CMU Locus Lab
# https://github.com/locuslab/TCN
#
from typing import Union, Tuple, List

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from mdp_video.models.mocogan_models import NoiseLayer


class Chomp3d(nn.Module):
    """
    Layer that truncates unneded output after convolution.
    """

    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size: int = chomp_size

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward call.

        :param data: input data
        :return: output tensor
        """
        return data[:, :, : -self.chomp_size, :, :].contiguous()


class TemporalBlockOriginal(nn.Module):
    """
    Residual block architecture as in TCN paper.
    """

    # pylint: disable=R0913
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: Union[int, Tuple],
        stride: Union[int, Tuple],
        dilation: Union[int, Tuple],
        padding: Union[int, Tuple],
        dropout: float = 0.2,
    ) -> None:
        """
        Init call.

        :param n_inputs: input channels
        :param n_outputs: output channels
        :param kernel_size: kernel size
        :param stride: stride
        :param dilation: dilation
        :param padding: padding
        :param dropout: dropout
        """
        super().__init__()

        self.conv1 = weight_norm(
            nn.Conv3d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp1 = Chomp3d(padding)  # type: ignore
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv3d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp2 = Chomp3d(padding)  # type: ignore
        self.relu2 = nn.ReLU()

        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv3d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of TemporalBlock.
        """
        out = data
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        res = data if self.downsample is None else self.downsample(data)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """
    TemporalConvNet with residual connection.
    """

    def __init__(self, num_inputs: int, num_channels: List[int], kernel_size: int = 4, dropout: float = 0.2) -> None:
        """
        Init call.

        :param num_inputs: input channels
        :param num_channels: output channels
        :param kernel_size: kernel size
        :param dropout: dropout
        """
        super().__init__()

        layers: List[torch.nn.Module] = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            time_padding = (kernel_size - 1) * dilation_size
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                TemporalBlockOriginal(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=(1, 2, 2),
                    dilation=(dilation_size, 1, 1),
                    padding=(time_padding, 1, 1),
                    dropout=dropout,
                )
            ]

        self.network = nn.Sequential(*layers)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward call.

        :param data: input data
        :return: output tensor
        """
        return self.network(data)


class DilatedConvBlock(nn.Module):
    """
    DilatedConvBlock from MOCOGAN.
    """

    # pylint: disable=R0913
    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        kernel_size: int,
        padding: int,
        dilation: int,
        use_noise: bool = False,
        noise_sigma: float = 0.0,
    ):
        super().__init__()
        self.block = nn.Sequential(
            NoiseLayer(use_noise, sigma=noise_sigma),
            nn.Conv3d(
                ch_in,
                ch_out,
                (kernel_size, 4, 4),
                bias=False,
                stride=(1, 2, 2),
                padding=(padding, 1, 1),
                dilation=(dilation, 1, 1),
            ),
            Chomp3d(padding),
            nn.BatchNorm3d(ch_out),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of DilatedConvBlock.
        """
        return self.block(data)


class TemporalVideoDiscriminator(nn.Module):
    """
    Temporal Discriminator network from PatchVideoDiscriminator.
    """

    # pylint: disable=R0913, R0914
    def __init__(
        self,
        n_channels: int,
        n_blocks: int,
        video_length: int,
        use_noise: bool = False,
        noise_sigma: float = 0.0,
        ndf: int = 64,
        tempo_kernel_size: int = 3,
        full_score: bool = False,
    ) -> None:
        super().__init__()
        self.full_score = full_score
        self.video_length = video_length

        layers = []
        in_channels = n_channels
        out_channels = ndf
        for dilation_rate in range(n_blocks):
            dilation = 2 ** dilation_rate
            padding = (tempo_kernel_size - 1) * dilation
            block = DilatedConvBlock(
                in_channels, out_channels, tempo_kernel_size, padding, dilation, use_noise, noise_sigma
            )
            layers.append(block)

            in_channels = ndf * dilation
            out_channels = in_channels * 2

        self.tcn_net = nn.Sequential(*layers)
        self.disc_net = nn.Conv3d(out_channels // 2, 1, (1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1), bias=False)
        self.gen_net = nn.Conv3d(out_channels // 2, 1, (1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1), bias=False)

    def forward(self, data: torch.Tensor) -> Tuple:
        """
        Forward pass of TemporalDiscriminator.
        """
        out = self.tcn_net(data)
        if self.full_score:
            return self.disc_net(out), self.gen_net(out), None
        return self.disc_net(out), None


class TemporalVideoDiscriminatorWeighted(nn.Module):
    """
    Temporal Discriminator network from PatchVideoDiscriminator.
    """

    # pylint: disable=R0913, R0914
    def __init__(
        self,
        n_channels: int,
        n_blocks: int,
        batch_size: int,
        video_length: int,
        image_size: int,
        use_noise: bool = False,
        noise_sigma: float = 0.0,
        ndf: int = 64,
        tempo_kernel_size: int = 3,
        full_score: bool = False,
    ):
        super().__init__()
        self.full_score = full_score
        self.video_length = video_length

        layers = []
        in_channels = n_channels
        out_channels = ndf
        for dilation_rate in range(n_blocks):
            dilation = 2 ** dilation_rate
            padding = (tempo_kernel_size - 1) * dilation
            block = DilatedConvBlock(
                in_channels, out_channels, tempo_kernel_size, padding, dilation, use_noise, noise_sigma
            )
            layers.append(block)

            in_channels = ndf * dilation
            out_channels = in_channels * 2

        self.tcn_net = nn.Sequential(*layers,)

        self.disc_net = nn.Conv3d(out_channels // 2, 1, (1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1), bias=False)
        self.gen_net = nn.Conv3d(out_channels // 2, 1, (1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1), bias=False)

        test_out_size = torch.zeros(batch_size, n_channels, video_length, image_size, image_size)
        self._linear_size = self.disc_net(self.tcn_net(test_out_size)).numel() // batch_size
        self.linear = nn.Linear(self._linear_size, 1)

    def forward(self, data: torch.Tensor) -> Tuple:
        """
        Forward pass of TemporalDiscriminator.
        """
        out = self.tcn_net(data)
        gen_out = self.gen_net(out)
        disc_out = self.disc_net(out)
        disc_out = self.linear(disc_out.view(-1, self._linear_size))

        if self.full_score:
            return disc_out, gen_out, None
        return disc_out, None
