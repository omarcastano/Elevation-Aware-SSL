from elevation_aware_ssl.backbone.resnet import conv3x3
import torch
from torch import nn

torch.cuda.empty_cache()
import torch.nn.functional as F


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channel, output_channel, scale_factor) -> None:
        super().__init__()

        self.scale_factor = scale_factor

        self.conv1 = conv3x3(in_channels + skip_channel, output_channel)
        self.bn1 = nn.BatchNorm2d(output_channel)

        self.conv2 = conv3x3(output_channel, output_channel)
        self.relu = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(output_channel)

    def forward(self, x, skip=None):

        x = F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")

        if skip is not None:
            x = torch.cat([x, skip], dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x


class ResNetDecoder(nn.Module):
    """
    ResNet-Based decoder for Unet

    Parameters
    ----------
        encoder_channel: list of integers
            number of output channels of the encoder at each block, this will be used to
            generate the skip connections of Unet architectures
        decoder_channels: list of integers
            specify **in_channels** parameter for convolutions used in decoder.
        scale_factor: list of float
            up-sampling factor to rescale the image.
    """

    def __init__(self, encoder_channels, decoder_channels, scale_factors) -> None:
        super().__init__()

        # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]

        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        blocks = [
            DecoderBlock(in_ch, sk_ch, out_ch, scale)
            for in_ch, sk_ch, out_ch, scale in zip(in_channels, skip_channels, out_channels, scale_factors)
        ]

        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features):

        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        skips = features[1:]
        x = features[0]

        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        return x
