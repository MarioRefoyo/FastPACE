from typing import cast
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def calculate_next_power_of_two(number):
    if number < 4:
        return 4
    else:
        pow2 = 4
        while True:
            if number < pow2:
                break
            else:
                pow2 = pow2 * 2
        return pow2


def maxpool1d_same_padding(input, kernel_size, stride, dilation):
    # stride and dilation are expected to be tuples.
    l_out = l_in = input.size(2)
    padding = (((l_out - 1) * stride) - l_in + (dilation * (kernel_size - 1)) + 1)
    if padding % 2 != 0:
        input = F.pad(input, [0, 1])

    return F.max_pool1d(input=input, kernel_size=kernel_size, stride=stride, padding=padding // 2, dilation=dilation)


class MaxPool1dSamePadding(nn.Module):
    def __init__(self, kernel_size: int, stride: int = 1, dilation: int = 1):
        super().__init__()
        self.k = int(kernel_size)
        self.s = int(stride)
        self.d = int(dilation)

    def forward(self, x):
        L_in = x.size(-1)
        L_out = (L_in + self.s - 1) // self.s
        pad_needed = max(0, (L_out - 1) * self.s + (self.k - 1) * self.d + 1 - L_in)
        pad_left = pad_needed // 2
        pad_right = pad_needed - pad_left
        if pad_needed > 0:
            x = F.pad(x, (int(pad_left), int(pad_right)))
        return F.max_pool1d(x, kernel_size=self.k, stride=self.s, padding=0, dilation=self.d)


class Conv1dSamePadding(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, dilation: int = 1, bias: bool = True) -> None:
        super().__init__()
        self.k = int(kernel_size)
        self.s = int(stride)
        self.d = int(dilation)
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self.k,
            stride=self.s,
            dilation=self.d,
            bias=bias,
            padding=0,
        )

    def forward(self, x):
        L_in = x.size(-1)
        L_out = (L_in + self.s - 1) // self.s
        pad_needed = max(0, (L_out - 1) * self.s + (self.k - 1) * self.d + 1 - L_in)
        pad_left = pad_needed // 2
        pad_right = pad_needed - pad_left
        if pad_needed > 0:
            x = F.pad(x, (int(pad_left), int(pad_right)))
        return self.conv(x)


class ConvBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int) -> None:
        super().__init__()
        """Conv1dSamePadding(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=kernel_size,
                          stride=stride),"""
        self.layers = nn.Sequential(
            Conv1dSamePadding(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride),
            nn.BatchNorm1d(num_features=out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        return self.layers(x)
