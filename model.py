import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

"""
Generator's block of the model
"""
class ConvResBlock(nn.Module):
    """
    This is an implementation of a Residual Convolution Block from the article:
        https://arxiv.org/pdf/1609.04802.pdf
    """

    def __init__(
            self,
            channels: int = 64
    ):
        """
        :param channels: int
            How many channels should be in convolutional blocks
        """
        self.channels = channels
        super(ConvResBlock, self).__init__()

        self.conv_res_block = nn.Sequential(
            nn.Conv2d(
                in_channels = self.channels,
                out_channels = self.channels,
                kernel_size = 3,
                stride = 1,
                padding = 1
            ),
            nn.BatchNorm2d(self.channels),
            nn.PReLU(),
            nn.Conv2d(
                in_channels = self.channels,
                out_channels = self.channels,
                kernel_size = 3,
                stride = 1,
                padding = 1,
                bias = False
            ),
            nn.BatchNorm2d(self.channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Residual Convolution Block
        :param x: torch.Tensor
            Input tensor
        :return: torch.Tensor
            Output tensor
        """
        initial_state = x
        out = self.conv_res_block(x)
        out = torch.add(out, initial_state)
        return out


class Generator(nn.Module):
    """
    Generator implementation of SRGAN from the article:
        https://arxiv.org/pdf/1609.04802.pdf
    """

    def __init__(self,
            input_channels: int = 3,
            out_channels: int = 64,
            input_kernel_size: int = 9,
            input_stride: int = 1,
            num_of_res_layers: int = 5
    ):
        super(Generator, self).__init__()

        self.input_channels = input_channels
        self.out_channels = out_channels
        self.input_kernel = input_kernel_size
        self.input_stride = input_stride
        self.first_conv_padding = int(np.ceil((self.input_kernel - self.input_stride) / 2))

        self.conv_1 = nn.Sequential(
            nn.Conv2d(
                in_channels = self.input_channels,
                out_channels = self.out_channels,
                kernel_size = self.input_kernel,
                stride = self.input_stride,
                padding = self.first_conv_padding
            ),
            nn.PReLU()
        )

        residual_block = []
        for i in range(num_of_res_layers):
            residual_block.append(ConvResBlock(self.out_channels))
        self.residual_block = nn.Sequential(*residual_block)

        self.conv_2 = nn.Sequential(
            nn.Conv2d(
                in_channels = self.out_channels,
                out_channels = self.out_channels,
                kernel_size = 3,
                stride = 1,
                padding = 1
            ),
            nn.BatchNorm2d(self.out_channels)
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(
                in_channels = self.out_channels,
                out_channels = 256,
                kernel_size = 3,
                stride = 1,
                padding = 1
            ),
            nn.PixelShuffle(2),
            nn.PReLU(),
            nn.Conv2d(
                in_channels = self.out_channels,
                out_channels = 256,
                kernel_size = 3,
                stride = 1,
                padding = 1
            ),
            nn.PixelShuffle(2),
            nn.PReLU(),
        )

        self.conv_4 = nn.Conv2d(
            in_channels = self.out_channels,
            out_channels = 3,
            kernel_size = self.input_kernel,
            stride = self.input_stride,
            padding = self.first_conv_padding
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
        """
        Forward pass of Generator
        :param x: torch.tensor
            Input Tensor
        :return: torch.Tensor
            Output tensor
        """
        output_1 = self.conv_1(x)
        output_2 = self.residual_block(output_1)
        output = self.conv_2(output_2)
        output = torch.add(output, output_1)
        output = self.conv_3(output)
        output = self.conv_4(output)

        return output

"""
Discriminator's block of the model
"""

class convBlockDiscriminator(nn.Module):
    # '''
    # Block in the discriminator with different stride and in/out channels.
    # '''
  def __init__(self,
               stride_size: int = 1,
               in_channels_size: int = 64,
               out_channels_size: int = 64
               ):
    super(convBlockDiscriminator, self).__init__()
    
    self.conv_block = nn.Sequential(nn.Conv2d(kernel_size=3, in_channels=in_channels_size, out_channels=out_channels_size, stride=stride_size),
                                    nn.BatchNorm2d(out_channels_size), 
                                    nn.LeakyReLU(inplace=True)
                                    )
    
  def forward(self, x: torch.Tensor) -> torch.Tensor:
      return self.conv_block(x)


class Discriminator(nn.Module):
    # """
    # Discriminator implementation of SRGAN from the article:
    #     https://arxiv.org/pdf/1609.04802.pdf
    # """
  def __init__(self,
               in_channels_size: int = 3,
               out_channels_size: int = 64,
               layer_size: int = 1000
               ):
    

    super(Discriminator, self).__init__()


    self.conv_start_block = nn.Sequential(nn.Conv2d(kernel_size=3,
                                                    in_channels=in_channels_size,
                                                    out_channels=out_channels_size,
                                                    stride=1),
                                          nn.LeakyReLU(inplace=True))
    
    self.conv_blocks = nn.ModuleList()

    self.conv_block_1 = convBlockDiscriminator(2, 64, 128)
    self.conv_block_2 = convBlockDiscriminator(1, 128, 128)
    self.conv_block_3 = convBlockDiscriminator(2, 128, 256)
    self.conv_block_4 = convBlockDiscriminator(1, 256, 256)
    self.conv_block_5 = convBlockDiscriminator(2, 256, 512)
    self.conv_block_6 = convBlockDiscriminator(1, 512, 512)
    self.conv_block_7 = convBlockDiscriminator(2, 512, 512)
    
    self.linear_block = nn.Sequential(nn.Linear(512*64*64, layer_size),
                                      nn.LeakyReLU(inplace=True),
                                      nn.Linear(layer_size, 1))


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
    """
        Forward pass of Discriminator
        :param x: torch.tensor
            Input Tensor
        :return: torch.Tensor
            Output tensor
    """
    out = self.conv_start_block(x)

    out = self.conv_block_1(out)
    out = self.conv_block_2(out)
    out = self.conv_block_3(out)
    out = self.conv_block_4(out)
    out = self.conv_block_5(out)
    out = self.conv_block_6(out)
    out = self.conv_block_7(out)
    
    out = out.flatten(start_dim=1)


    out = self.linear_block(out) 

    return F.sigmoid(out)