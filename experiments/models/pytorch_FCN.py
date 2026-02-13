from torch import nn
from experiments.models.pytorch_model_utils import ConvBlock


class FCNBackbone(nn.Module):
    def __init__(self, in_channels: int, channels: list, kernel_sizes: list) -> None:
        super().__init__()
        if len(channels) != len(kernel_sizes):
            raise ValueError("Length of channels and kernel sizes must be the same.")

        self.layer_output_channels = channels
        channels = [in_channels] + [channel for channel in channels]
        self.layers = nn.Sequential(*[
            ConvBlock(in_channels=channels[i], out_channels=channels[i+1],
                      kernel_size=kernel_sizes[i], stride=1) for i in range(len(kernel_sizes))
        ])

        self.kernel_sizes = kernel_sizes
        self.strides = [1 for _ in range(len(kernel_sizes))]
        self.paddings = ["SAME" for _ in range(len(kernel_sizes))]
        self.output_channels = channels[-1]

    def conv_info(self, layer=None):
        if layer is None:
            return self.kernel_sizes, self.strides, self.paddings
        else:
            return self.kernel_sizes[:layer], self.strides[:layer], self.paddings[:layer]

    def forward(self, x):
        x = self.layers(x)
        return x


class FCN(nn.Module):
    def __init__(self, in_channels: int, channels: list = [128, 256, 128], kernel_sizes: list = [8, 5, 3],
                 num_classes: int = 1) -> None:
        super().__init__()

        self.back_bone = FCNBackbone(in_channels, channels, kernel_sizes)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels[-1], num_classes)

    def forward(self, x):
        x = self.back_bone(x)
        x = self.gap(x)
        x = x.squeeze(dim=-1)
        x = self.fc(x)
        return x

    def predict(self, x):
        return self.forward(x)

