import torch
from torch.utils.data import DataLoader
from torch.nn import Module
from torch import nn
from torchvision.models import resnet50, resnet18, resnet34


torch.manual_seed(42)

BACKBONES = {
    "resnet18": resnet18(weights=None),
    "resnet34": resnet34(weights=None),
    "resnet50": resnet50(weights=None),
}


class LinearClassifier(nn.Module):

    """
    Linear Classifier model on top a ResNet model
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        num_classes: int = 3,
    ) -> None:

        super(LinearClassifier, self).__init__()

        self.backbone = BACKBONES[backbone]
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
        self.backbone.maxpool = nn.Identity()
        self.backbone.fc = nn.Identity()

        self.fc = nn.Sequential(
            nn.Linear(in_features=self.backbone.inplanes, out_features=num_classes),
        )

    def forward(self, x):

        x = self.backbone(x)
        x = self.fc(x)

        return x


class NoneLinearClassifier(nn.Module):

    """
    None Linear Classifier model on top a ResNet model
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        num_classes: int = 3,
        n_hidden: int = 512,
    ) -> None:

        super(NoneLinearClassifier, self).__init__()

        self.backbone = BACKBONES[backbone]
        self.backbone.fc = nn.Identity()

        self.fc = nn.Sequential(
            nn.Linear(in_features=self.backbone.inplanes, out_features=n_hidden),
            nn.ReLU(),
            nn.Linear(in_features=n_hidden, out_features=n_hidden),
            nn.ReLU(),
            nn.Linear(in_features=n_hidden, out_features=num_classes),
        )

    def forward(self, x):

        x = self.backbone(x)
        x = self.fc(x)

        return x
