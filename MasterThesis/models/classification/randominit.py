import torch
from torch.utils.data import DataLoader
from torch.nn import Module
from torch import nn
from MasterThesis.backbone import resnet50, resnet18


torch.manual_seed(42)

BACKBONES = {
    "resnet18": resnet18,
    # "resnet34": resnet34(weights=None),
    "resnet50": resnet50,
}


class LinearClassifier(nn.Module):

    """
    Linear Classifier model on top a ResNet model

    Arguments:
    ----------
        backbone: string, default = "resnet50"
            backbone to be pre-trained
        num_classes: integer, default=None
            number of different classes
        cifar: boolean, default=False
            If True removes the first max pooling layer, and the kernel in the
            first convolutional layer is change from 7x7 to 3x3.
    """

    def __init__(self, backbone: str = "resnet50", num_classes: int = None, cifar: bool = False) -> None:

        super(LinearClassifier, self).__init__()

        self.backbone = BACKBONES[backbone](cifar=cifar)
        self.backbone.fc = nn.Identity()

        self.fc = nn.Sequential(
            nn.Linear(in_features=self.backbone.in_planes, out_features=num_classes),
        )

        self._init_weight()

    def forward(self, x):

        x = self.backbone(x)
        x = self.fc(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)


class NoneLinearClassifier(nn.Module):

    """
    None Linear Classifier model on top a ResNet model

    Arguments:
    ----------
        backbone: string, default = "resnet50"
            backbone to be pre-trained
        num_classes: integer, default=None
            number of different classes
        proj_hidden_dim: integer, default=512
            number of hidden units
        cifar: boolean, default=False
            If True removes the first max pooling layer, and the kernel in the
            first convolutional layer is change from 7x7 to 3x3.
    """

    def __init__(self, backbone: str = "resnet50", num_classes: int = 3, n_hidden: int = 512, cifar: bool = False) -> None:

        super(NoneLinearClassifier, self).__init__()

        self.backbone = BACKBONES[backbone](cifar=cifar)
        self.backbone.fc = nn.Identity()

        self.fc = nn.Sequential(
            nn.Linear(in_features=self.backbone.in_planes, out_features=n_hidden),
            nn.ReLU(),
            nn.BatchNorm1d(n_hidden),
            nn.Linear(in_features=n_hidden, out_features=n_hidden),
            nn.ReLU(),
            nn.BatchNorm1d(n_hidden),
            nn.Linear(in_features=n_hidden, out_features=num_classes),
        )

        self._init_weight()

    def forward(self, x):

        x = self.backbone(x)
        x = self.fc(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
