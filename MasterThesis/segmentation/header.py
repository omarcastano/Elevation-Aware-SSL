from torch import nn
from .decoder import DecoderBlock


class SegmentationHead(nn.Module):
    def __init__(self, in_channels, out_channels, input_size, output_size) -> None:
        super().__init__()

        self.upscaling = output_size / input_size

        if self.upscaling > 1:
            self.block1 = DecoderBlock(in_channels, 0, in_channels // 2, self.upscaling)
            self.conv1 = nn.Conv2d(in_channels // 2, out_channels, kernel_size=1)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):

        if self.upscaling > 1:
            x = self.block1(x)
        x = self.conv1(x)

        return x
