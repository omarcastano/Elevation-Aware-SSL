import torch
from torch import nn
from torchvision.models import resnet18, resnet34, resnet50

BACKBONES = {
    "resnet18": resnet18(weights=None),
    "resnet34": resnet34(weights=None),
    "resnet50": resnet50(weights=None),
}


class SimCLR(nn.Module):

    """
    SimCLR model using ResNet as backbone

    Arguments:
    ----------
        backbone: string, default = "resnet50"
            backbone to be pre-trained
        proj_hidden_dim: integer, default=512
            number of hidden units
        proj_output_dim: integer, default=128
            number of output units
        cifar: boolean, default=False
            If True removes the first max pooling layer, and the kernel in the
            first convolutional layer is change from 7x7 to 3x3.
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        proj_hidden_dim: int = 128,
        proj_output_dim: int = 512,
        cifar: bool = False,
    ) -> None:

        super(SimCLR, self).__init__()

        self.backbone = BACKBONES[backbone]
        if cifar:
            self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
            self.backbone.maxpool = nn.Identity()

        self.backbone.fc = nn.Identity()

        self.projector = nn.Sequential(
            nn.Linear(in_features=self.backbone.inplanes, out_features=proj_hidden_dim, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.Linear(in_features=proj_hidden_dim, out_features=proj_output_dim, bias=False),
        )

    def forward(self, x):

        x = self.backbone(x)
        x = self.projector(x)

        return x
