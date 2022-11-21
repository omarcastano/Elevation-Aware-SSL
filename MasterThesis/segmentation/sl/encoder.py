from MasterThesis.backbone.resnet import resnet50, resnet18
import torch
from torch import nn

torch.cuda.empty_cache()
import torch.nn.functional as F
import numpy as np

from typing import List, Tuple


BACKBONES = {
    "resnet18": resnet18,
    "resnet50": resnet50,
}

out_channels = {
    "resnet18": [3, 64, 64, 128, 256, 512],
    "resnet50": [3, 64, 256, 512, 1024, 2048],
}


class ResNetEncoder(nn.Module):
    """ResNet-based Unet Encoder

    Arguments
    ----------
        backbone : str, default='resnet50'
            Name of the classification model that will be used as an encoder (a.k.a backbone)
            to extract features of different spatial resolution. Possible options are: resnet18, resnet50
        depth: int, default=5
            A number of stages used in encoder in range [3, 5]. Each stage generate features
            two times smaller in spatial dimensions than previous one (e.g. for depth 0 we will have features
            with shapes [(N, C, H, W),], for depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
            if cifar=True, the first stage will not reduce de spatial dimensions of the input image.
        cifar : bool, default=False
            If True removes the first max pooling layer, and the kernel in the
            first convolutional layer is change from 7x7 to 3x3.
    """

    def __init__(self, backbone: str = "resnet50", depth=5, cifar: bool = False) -> None:
        super().__init__()

        self.backbone = BACKBONES[backbone](cifar=cifar)
        self.backbone.out_channels = out_channels[backbone]
        self.depth = depth

        self.backbone.fc = torch.nn.Identity()

    def get_stages(self):

        stages = [
            nn.Identity(),
            nn.Sequential(self.backbone.conv1, self.backbone.bn1, self.backbone.relu),
            nn.Sequential(self.backbone.maxpool, self.backbone.layer1),
            self.backbone.layer2,
            self.backbone.layer3,
            self.backbone.layer4,
        ]

        return stages

    def forward(self, x):

        stages = self.get_stages()

        features = []
        for i in range(self.depth + 1):
            x = stages[i](x)
            features.append(x)

        return features
