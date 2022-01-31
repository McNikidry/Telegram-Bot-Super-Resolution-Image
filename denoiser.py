import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

"""
This is an implementation of architecture mentioned in https://arxiv.org/pdf/1904.07396.pdf
used for image Denoising.
"""


class MRU_block(nn.Module):

    def __init__(
            self,
            output_channels: int = 64,
            first_dilation: int = 1,
            second_dilation: int = 2
    ):
        super(MRU_block, self).__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(
                in_channels = output_channels,
                out_channels = output_channels,
                kernel_size = 3,
                stride = 1,
                padding = first_dilation,
                dilation = first_dilation
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels = output_channels,
                out_channels = output_channels,
                kernel_size = 3,
                stride = 1,
                padding = second_dilation,
                dilation = second_dilation
            ),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.layer(x)
        return output


class FeatureAttention(nn.Module):

    def __init__(
            self,
            input_channels: int = 64,
            reduction_level: int = 4
    ):
        super(FeatureAttention, self).__init__()

        self.layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                in_channels = input_channels,
                out_channels = input_channels // reduction_level,
                kernel_size = 1,
                stride = 1,
                padding = 0
            ),
            nn.Conv2d(
                in_channels = input_channels // reduction_level,
                out_channels = input_channels,
                kernel_size = 1,
                stride = 1,
                padding = 0
            ),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.layer(x)
        output = x + x1
        return output


class EAM(nn.Module):

    def __init__(
            self,
            output_channels: int = 64,
            first_branch_dilations: list = [1, 2],
            second_branch_dilations: list = [3, 4],
            downsampling: int = 16
    ):
        super(EAM, self).__init__()

        self.first_branch = MRU_block(
            output_channels = output_channels,
            first_dilation = first_branch_dilations[0],
            second_dilation = first_branch_dilations[1]
        )

        self.second_branch = MRU_block(
            output_channels = output_channels,
            first_dilation = second_branch_dilations[0],
            second_dilation = second_branch_dilations[1]
        )

        self.conc_conv = nn.Sequential(
            nn.Conv2d(
                in_channels = output_channels * 2,
                out_channels = output_channels,
                kernel_size = 3,
                stride = 1,
                padding = 1
            ),
            nn.ReLU()
        )

        self.second_block = nn.Sequential(
            nn.Conv2d(
                in_channels = output_channels,
                out_channels = output_channels,
                kernel_size = 3,
                stride = 1,
                padding = 1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels = output_channels,
                out_channels = output_channels,
                kernel_size = 3,
                stride = 1,
                padding = 1
            ),
            nn.ReLU()
        )

        self.third_block = nn.Sequential(
            nn.Conv2d(
                in_channels = output_channels,
                out_channels = output_channels,
                kernel_size = 3,
                stride = 1,
                padding = 1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels = output_channels,
                out_channels = output_channels,
                kernel_size = 3,
                stride = 1,
                padding = 1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels = output_channels,
                out_channels = output_channels,
                kernel_size = 1,
                stride = 1,
                padding = 0
            ),
            nn.ReLU()
        )

        self.feature_attention = FeatureAttention(input_channels = output_channels)

        self.fourth_block = nn.Sequential(
            nn.Conv2d(
                in_channels = output_channels,
                out_channels = output_channels // downsampling,
                kernel_size = 1,
                stride = 1,
                padding = 0
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels = output_channels // downsampling,
                out_channels = output_channels,
                kernel_size = 1,
                stride = 1,
                padding = 0
            ),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b_1 = self.first_branch(x)
        b_2 = self.second_branch(x)
        c = torch.cat([b_1, b_2], dim = 1)

        x1 = self.conc_conv(c)
        x1_1 = x1 + x

        x2 = self.second_block(x1)
        x2_1 = x2 + x1

        x3 = self.third_block(x2_1)
        x3_1 = x3 + x2

        x4 = self.feature_attention(x3_1)
        x4 = self.fourth_block(x4)
        x5 = x4 * x3

        output = x + x5

        return output


class RidNet(nn.Module):

    def __init__(self,
                 out_channels: int = 64,
                 n_res_block: int = 4,
                 first_branch_dilations: list = [1, 2],
                 second_branch_dilations: list = [3, 4],
                 downsampling: int = 16
                 ):
        super(RidNet, self).__init__()

        self.feature_extraction = nn.Sequential(
            nn.Conv2d(in_channels = 3,
                      out_channels = out_channels,
                      kernel_size = 3,
                      padding = 1
                      ),
            nn.ReLU()
        )

        residual_block = []
        for i in range(n_res_block):
            residual_block.append(EAM(
                output_channels = out_channels,
                first_branch_dilations = first_branch_dilations,
                second_branch_dilations = second_branch_dilations,
                downsampling = downsampling
            ))
        self.residual_block = nn.Sequential(*residual_block)

        self.last_conv = nn.Sequential(
            nn.Conv2d(
                in_channels = out_channels,
                out_channels = 3,
                kernel_size = 3,
                stride = 1,
                padding = 1
            )
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """
        Weights initialization.
        For convolutional blocks there is "He initialization".
        :return:
            None
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x1 = self.feature_extraction(x)
        x2 = self.residual_block(x1)
        x3 = x2 + x1
        x4 = self.last_conv(x2)
        output = x + x4
        return output


class DnCNN(nn.Module):

    def __init__(
            self,
            in_channels: int = 3,
            num_of_mid_layers: int = 10

    ):
        super(DnCNN, self).__init__()

        self.first_layer = nn.Sequential(
            nn.Conv2d(
                in_channels = in_channels,
                out_channels = 64,
                kernel_size = 3,
                stride = 1,
                padding = 1
            ),
            nn.ReLU()
        )

        mid_block = []
        for i in range(num_of_mid_layers):
            mid_block.append(nn.Sequential(
                nn.Conv2d(
                    in_channels = 64,
                    out_channels = 64,
                    kernel_size = 3,
                    stride = 1,
                    padding = 1
                ),
                nn.BatchNorm2d(64),
                nn.ReLU()
            ))
        self.mid_block = nn.Sequential(*mid_block)

        self.last_layer = nn.Conv2d(
            in_channels = 64,
            out_channels = 3,
            kernel_size = 3,
            stride = 1,
            padding = 1
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """
            Weights initialization.
            For convolutional blocks there is "He initialization".
            :return:
                None
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        noise = self.first_layer(x)
        noise = self.mid_block(noise)
        noise = self.last_layer(noise)
        output = x - noise
        return output

    nn.MSELoss
