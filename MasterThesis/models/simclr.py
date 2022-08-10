import torch
import torch.nn as nn
import torch.nn.functional as F
from .deeplab_utils import ResNet as ResNet
from .deeplab_utils.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from .deeplab_utils.encoder import Encoder


class DeepLab(nn.Module):
    def __init__(
        self,
        num_classes=2,
        in_channels=3,
        arch="resnet101",
        output_stride=16,
        bn_momentum=0.9,
        freeze_bn=False,
        pretrained=False,
        **kwargs
    ):
        super(DeepLab, self).__init__(**kwargs)
        self.model_name = "deeplabv3plus_" + arch

        # Setup arch
        if arch == "resnet18":
            NotImplementedError("resnet18 backbone is not implemented yet.")
        elif arch == "resnet34":
            NotImplementedError("resnet34 backbone is not implemented yet.")
        elif arch == "resnet50":
            self.backbone = ResNet.resnet50(bn_momentum, pretrained)
            if in_channels != 3:
                self.backbone.conv1 = nn.Conv2d(
                    in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
                )
        elif arch == "resnet101":
            self.backbone = ResNet.resnet101(bn_momentum, pretrained)
            if in_channels != 3:
                self.backbone.conv1 = nn.Conv2d(
                    in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
                )

        self.encoder = Encoder(bn_momentum, output_stride)
        # self.decoder = Decoder(20,100, bn_momentum)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # projection head
        """
        self.proj = nn.Sequential(
            nn.Conv2d(256, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 10, 1, bias=True)
        )
        """
        self.proj = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, num_classes)
        )

    def forward(self, input, input1):
        x, _ = self.backbone(input)
        # print(low_level_features.size()),56
        x = self.encoder(x)
        # print(x.size()),14
        x = self.avgpool(x)
        # print(x.size())
        x = torch.flatten(x, 1)
        # print(x.size())
        q = self.proj(x)
        x, _ = self.backbone(input1)

        x = self.encoder(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        k = self.proj(x)
        # predict,predict1 = self.decoder(x1, low_level_features)

        return q, k

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()


def build_model(num_classes=5, in_channels=3, pretrained=False, arch="resnet101"):
    model = DeepLab(
        num_classes=num_classes,
        in_channels=in_channels,
        pretrained=pretrained,
        arch=arch,
    )
    return model


if __name__ == "__main__":
    model = DeepLab(output_stride=16, class_num=21, pretrained=False, freeze_bn=False)
    model.eval()
    for m in model.modules():
        if isinstance(m, SynchronizedBatchNorm2d):
            print(m)
