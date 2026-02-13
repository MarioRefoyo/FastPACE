import torch
from torch import nn
from experiments.models.pytorch_model_utils import ConvBlock, MaxPool1dSamePadding
from typing import cast, Union, List


class InceptionModel(nn.Module):
    """A PyTorch implementation of the InceptionTime model.
    From https://arxiv.org/abs/1909.04939"""

    def __init__(self, depth: int, in_channels: int, out_channels: Union[list, int],
                 bottleneck_channels: Union[list, int], kernel_sizes: Union[list, int],
                 use_residuals: bool = True,
                 num_classes: int = 1
                 ) -> None:
        super().__init__()

        # for easier saving and loading
        self.input_args = {
            'depth': depth,
            'in_channels': in_channels,
            'out_channels': out_channels,
            'bottleneck_channels': bottleneck_channels,
            'kernel_sizes': kernel_sizes,
            'use_residuals': use_residuals,
            'num_classes': num_classes
        }

        # channels = [in_channels] + cast(List[int], self._expand_to_blocks(out_channels, num_blocks))
        # bottleneck_channels = cast(List[int], self._expand_to_blocks(bottleneck_channels, num_blocks))
        # kernel_sizes = cast(List[int], self._expand_to_blocks(kernel_sizes, num_blocks))
        self.back_bone = InceptionBackBone(
            in_channels, out_channels, bottleneck_channels,
            kernel_sizes, depth, use_residuals)

        # Global average pooling (i.e. mean of the time dimension) is why
        # in_features=channels[-1]
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(in_features=self.back_bone.output_channels, out_features=num_classes)

    @staticmethod
    def _expand_to_blocks(value: Union[int, bool, List[int], List[bool]],
                          num_blocks: int) -> Union[List[int], List[bool]]:
        if isinstance(value, list):
            assert len(value) == num_blocks, \
                f'Length of inputs lists must be the same as num blocks, ' \
                f'expected length {num_blocks}, got {len(value)}'
        else:
            value = [value] * num_blocks
        return value

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        x = self.back_bone(x)
        x = self.gap(x)
        x = x.squeeze(dim=-1)
        x = self.linear(x)
        return x

    def predict(self, x):
        return self.forward(x)


class InceptionBackBone(nn.Module):
    def __init__(self, in_channels: int, out_channels: Union[list, int], bottleneck_channels: Union[list, int],
                 kernel_sizes: int, depth: int, use_residuals: bool) -> None:
        super().__init__()
        self.depth = depth
        self.use_residuals = use_residuals
        self.inception_blocks, self.shortcut_blocks = nn.ModuleList(), nn.ModuleList()
        for d in range(depth):
            inception_in_channels = in_channels if d == 0 else out_channels * 4
            self.inception_blocks.append(
                InceptionBlock(in_channels=inception_in_channels, out_channels=out_channels,
                               bottleneck_channels=bottleneck_channels,
                               kernel_size=kernel_sizes)
            )
            if use_residuals and d % 3 == 2:
                shortcut_in_channels = in_channels if d == 2 else out_channels * 4
                shortcut_out_channels = out_channels * 4
                self.shortcut_blocks.append(
                    nn.Sequential(
                        nn.Conv1d(
                            in_channels=shortcut_in_channels,
                            out_channels=shortcut_out_channels,
                            kernel_size=1,
                            stride=1,
                            padding='same'),
                        nn.BatchNorm1d(num_features=shortcut_out_channels)  # No activation
                    )

                )
        self.relu = nn.ReLU()
        self.output_channels = out_channels*4

    def forward(self, x):
        res = x
        for d, l in enumerate(range(self.depth)):
            x = self.inception_blocks[d](x)
            if self.use_residuals and d % 3 == 2:
                res = self.shortcut_blocks[d // 3](res)
                res = self.relu(x + res)
                x = res
        return x


class InceptionBlock(nn.Module):
    """An inception block consists of an (optional) bottleneck, followed
    by 3 conv1d layers. Optionally residual
    """

    def __init__(self, in_channels: int, out_channels: int,
                 stride: int = 1, bottleneck_channels: int = 32,
                 kernel_size: int = 41) -> None:
        assert kernel_size > 3, "Kernel size must be strictly greater than 3"
        super().__init__()

        self.use_bottleneck = bottleneck_channels > 0 and int(in_channels) > 1
        if self.use_bottleneck:
            self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False, padding="same")
            # ToDo: activation of bottleneck
            start_channels = bottleneck_channels
        else:
            start_channels = in_channels

        kernel_size_s = [kernel_size // (2 ** i) for i in range(3)]
        channels = [start_channels] + [out_channels] * 3
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(in_channels=start_channels, out_channels=channels[i + 1],
                      kernel_size=kernel_size_s[i], stride=stride, bias=False, padding="same")
            for i in range(len(kernel_size_s))
        ])

        self.max_pool_conv = nn.Sequential(*[
            MaxPool1dSamePadding(kernel_size=3, stride=stride),
            nn.Conv1d(in_channels=start_channels, out_channels=out_channels,
                      kernel_size=1, stride=stride, bias=False, padding="same"),
        ])

        self.batch_norm = nn.BatchNorm1d(num_features=channels[-1]*4)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        if self.use_bottleneck:
            x = self.bottleneck(x)
        x = torch.cat([l(x) for l in self.conv_layers] + [self.max_pool_conv(x)], dim=1)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x
