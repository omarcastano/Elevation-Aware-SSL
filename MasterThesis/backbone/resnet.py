import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Optional, List


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """
    3x3 convolution with padding
    """

    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """
    1x1 convolution
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


class BasicBlock(nn.Module):
    """
    Return resnet BasickBLock

    Arguments:
    ----------
        in_channels: int,
            input number of channels
        out_channels: int
            output numbers of channels
        stride: int
            stride of the convolution
        downsample: torch.nn.Module
            module to reduce the number of channels
    """

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample: Optional[nn.Module] = None):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x) -> torch.Tensor:

        skip = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample:
            skip = self.downsample(skip)

        x += skip
        x = self.relu(x)

        return x


class BottleNeck(nn.Module):
    """
    Return resnet BlottleNeck

    Arguments:
    ----------
        in_channels: int,
            input number of channels
        out_channels: int
            output numbers of channels
        stride: int
            stride of the convolution
        downsample: torch.nn.Module
            module to reduce the number of channels
    """

    expansion = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample: Optional[nn.Module] = None) -> None:
        super().__init__()

        self.conv1 = conv1x1(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = conv3x3(out_channels, out_channels, stride=stride)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = conv1x1(out_channels, out_channels * self.expansion)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x) -> torch.Tensor:

        skip = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.downsample:
            skip = self.downsample(skip)

        x += skip
        x = self.relu(x)

        return x


class ResNet(nn.Module):
    def __init__(self, block: BasicBlock, layers: List[int], num_classes: int, in_channels: int, cifar: bool = False) -> None:
        super().__init__()

        self.in_planes = 64

        if cifar:
            self.maxpool = nn.Identity()
            self.conv1 = nn.Conv2d(in_channels, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(in_channels, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU()

        self.layer1 = self._make_layer(block, layers[0], planes=64, stride=1)
        self.layer2 = self._make_layer(block, layers[1], planes=128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], planes=256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], planes=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._init_weights()

    def forward(self, x) -> torch.Tensor:

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def _make_layer(self, block: BasicBlock, blocks: int, planes: int, stride: int = 1) -> nn.Sequential:
        downsample = None

        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.in_planes, planes * block.expansion, stride), nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))

        self.in_planes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def resnet18(num_classes: int = 1000, num_channels: int = 3, cifar: bool = False):
    """
    Returns ResNet18

    Arguments:
    ----------
        num_classes: int, default=1000
            number of classes
        num_channels: int, default=3
            number of input image channels
        cifar: bool, default=3
            if True removes the first max pooling layer, and the kernel in the
            first convolutional layer is change from 7x7 to 3x3.
    """

    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, in_channels=num_channels, cifar=cifar)


def resnet50(num_classes: int = 1000, num_channels: int = 3, cifar: bool = False):
    """
    Returns ResNet50

    Arguments:
    ----------
        num_classes: int, default=1000
            number of classes
        num_channels: int, default=3
            number of input image channels
        cifar: bool, default=3
            if True removes the first max pooling layer, and the kernel in the
            first convolutional layer is change from 7x7 to 3x3.
    """

    return ResNet(BottleNeck, [3, 4, 6, 3], num_classes=num_classes, in_channels=num_channels, cifar=cifar)
