import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from experiments.models.pytorch_model_utils import Conv1dSamePadding, ConvBlock, calculate_next_power_of_two


class ConvTransposeBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        output_padding = stride - 1
        self.layers = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
            ),
            nn.BatchNorm1d(num_features=out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ConvAutoencoder1d(nn.Module):
    def __init__(self, input_channels: int, input_length: int, compression_rate: float,
                 encoder_filters_kernels: list, stride: int) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.input_length = input_length
        self.stride = stride

        new_ts_length = calculate_next_power_of_two(input_length)
        padding_count = new_ts_length - input_length
        self.pad_left = padding_count // 2
        self.pad_right = padding_count - self.pad_left

        latent_length = new_ts_length
        for _ in encoder_filters_kernels:
            latent_length = math.ceil(latent_length / stride)
        total_input_size = new_ts_length * input_channels
        new_latent_channels = max(1, int(math.floor(compression_rate * total_input_size / latent_length)))

        encoder_layers = []
        in_channels = input_channels
        for filters, kernel_size in encoder_filters_kernels:
            encoder_layers.append(ConvBlock(in_channels, filters, kernel_size, stride))
            in_channels = filters
        self.encoder = nn.Sequential(*encoder_layers)

        self.bottleneck_in = Conv1dSamePadding(
            in_channels=in_channels,
            out_channels=new_latent_channels,
            kernel_size=3,
            stride=1,
        )
        self.bottleneck_out = Conv1dSamePadding(
            in_channels=new_latent_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=1,
        )

        decoder_layers = []
        for filters, kernel_size in encoder_filters_kernels[::-1]:
            decoder_layers.append(ConvTransposeBlock(in_channels, filters, kernel_size, stride))
            in_channels = filters
        self.decoder = nn.Sequential(*decoder_layers)

        self.output_layer = Conv1dSamePadding(
            in_channels=in_channels,
            out_channels=input_channels,
            kernel_size=3,
            stride=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad_left or self.pad_right:
            x = F.pad(x, (self.pad_left, self.pad_right))
        x = self.encoder(x)
        x = self.bottleneck_in(x)
        x = self.bottleneck_out(x)
        x = self.decoder(x)
        x = self.output_layer(x)
        if self.pad_left or self.pad_right:
            end_idx = x.size(-1) - self.pad_right
            x = x[..., self.pad_left:end_idx]
        return x

    def predict(self, x, batch_size: int = 64, **kwargs) -> np.ndarray:
        if isinstance(x, np.ndarray):
            x_tensor = torch.from_numpy(x).float()
        else:
            x_tensor = x.detach().cpu().float()
        if x_tensor.ndim == 2:
            x_tensor = x_tensor.unsqueeze(0)

        batch_size = max(1, min(batch_size, x_tensor.shape[0]))
        loader = DataLoader(TensorDataset(x_tensor), batch_size=batch_size, shuffle=False)

        was_training = self.training
        self.eval()
        preds = []
        device = next(self.parameters()).device
        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(device)
                output = self.forward(batch)
                preds.append(output.cpu())
        if was_training:
            self.train()
        return torch.cat(preds, dim=0).numpy()


class AEModelConstructorV1:
    def __init__(self, input_shape, stride, compression_rate):
        self.input_shape = input_shape
        self.stride = stride
        self.compression_rate = compression_rate
        self.ae_cnn_shallow_arch = {"encoder_filters_kernels": [(16, 7)]}
        self.ae_cnn_simple_arch = {"encoder_filters_kernels": [(16, 7), (32, 5)]}
        self.ae_cnn_intermediate_arch = {"encoder_filters_kernels": [(16, 7), (32, 5), (64, 3)]}
        self.ae_cnn_complex_arch = {"encoder_filters_kernels": [(16, 7), (32, 5), (64, 3), (128, 3)]}

    def get_model(self, model_name):
        if model_name == "ae-cnn-shallow":
            encoder_filters_kernels = self.ae_cnn_shallow_arch["encoder_filters_kernels"]
        elif model_name == "ae-cnn-simple":
            encoder_filters_kernels = self.ae_cnn_simple_arch["encoder_filters_kernels"]
        elif model_name == "ae-cnn-intermediate":
            encoder_filters_kernels = self.ae_cnn_intermediate_arch["encoder_filters_kernels"]
        elif model_name == "ae-cnn-complex":
            encoder_filters_kernels = self.ae_cnn_complex_arch["encoder_filters_kernels"]
        else:
            raise NameError(f"Model name {model_name} is not valid.")

        input_channels, input_length = self.input_shape
        model = ConvAutoencoder1d(
            input_channels=input_channels,
            input_length=input_length,
            compression_rate=self.compression_rate,
            encoder_filters_kernels=encoder_filters_kernels,
            stride=self.stride,
        )
        return model


Autoencoder = AEModelConstructorV1
