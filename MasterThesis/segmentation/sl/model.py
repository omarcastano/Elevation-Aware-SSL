from torch import nn
import numpy as np
from typing import List
from . import ResNetDecoder, ResNetEncoder, SegmentationHead


class Unet(nn.Module):
    """
    Unet_ is a fully convolution neural network for image semantic segmentation. Consist of *encoder*
    and *decoder* parts connected with *skip connections*. Encoder extract features of different spatial
    resolution (skip connections) which are used by decoder to define accurate segmentation mask. Use *concatenation*
    for fusing decoder blocks with skip connections.ResNet-based Unet Encoder

    Arguments
    ----------
        backbone : str, default=resnet50
            Name of the classification model that will be used as an encoder (a.k.a backbone)
            to extract features of different spatial resolution. Possible options are: resnet18, resnet50
        encoder_depth: int, default=5
            a number of stages used in encoder. Each stage generate features two times smaller in spatial
            dimensions than previous one (e.g. for depth 0 we will have features with shapes [(N, C, H, W),],
            for depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
        decoder_channels: list of integers, default=[256, 128, 64, 32, 16]
            specify **in_channels** parameter for convolutions used in decoder.
        input_size: int, default = 100
            hight and width of the input image
        output_size: int, default = 100
            hight and width of the output mask
        number_classes: int, default=1
            the number of classes for output mask (or you can think as a number of channels of output mask)
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        encoder_depth: int = 5,
        decoder_channels: List = [256, 128, 64, 32, 16],
        input_size: int = 100,
        number_classes: int = 1,
        output_size: int = 100,
    ) -> None:
        super().__init__()

        self.backbone_name = backbone
        self.encoder_depth = encoder_depth
        self.decoder_channels = decoder_channels
        self.decoder_depth = len(decoder_channels)
        self.input_size = input_size
        self.scale_factor, self.decoder_input_size = self.get_upscale_factors()

        self.encoder = ResNetEncoder(self.backbone_name, self.encoder_depth)
        self.decoder = ResNetDecoder(self.encoder.backbone.out_channels, self.decoder_channels, self.scale_factor)

        self.header = SegmentationHead(decoder_channels[-1], number_classes, self.decoder_input_size[-1], output_size)

        self._init_weights()

    def get_upscale_factors(self):

        in_size = self.input_size
        out_size = in_size
        scale_factor = []
        output_size = [in_size]
        for _ in range(self.encoder_depth):
            in_size = out_size
            out_size = np.ceil(in_size / 2)
            output_size.append(out_size)
            scale_factor.append(in_size / out_size)

        scale_factor = scale_factor[::-1]
        output_size = output_size[::-1]

        return scale_factor[: self.decoder_depth], output_size[1 : self.decoder_depth + 1]

    def forward(self, x):

        encoder_output = self.encoder(x)
        decoder_output = self.decoder(*encoder_output)
        mask = self.header(decoder_output)

        return mask

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
