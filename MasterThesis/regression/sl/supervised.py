import torch
from torch import nn
from MasterThesis.regression.sl.resnet import resnet50, resnet18


torch.manual_seed(42)

BACKBONES = {
    "resnet18": resnet18(),
    "resnet50": resnet50(),
}


class LinearRegressor(nn.Module):

    """
    Linear Regressor model on top a ResNet model

    Arguments:
    ----------
        backbone: string, default = "resnet50"
            backbone to be pre-trained
        output_size: integer, default=None
            number of outputs
    """

    def __init__(self, backbone: str = "resnet50", output_size: int = None) -> None:

        super(LinearRegressor, self).__init__()

        self.backbone = BACKBONES[backbone]
        self.backbone.fc = nn.Identity()

        self.fc = nn.Sequential(
            nn.Linear(in_features=self.backbone.in_planes, out_features=output_size),
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


class NoneLinearRegressor(nn.Module):

    """
    None Linear Regressor model on top a ResNet model

    Arguments:
    ----------
        backbone: string, default = "resnet50"
            backbone to be pre-trained
        output_size: integer, default=None
            number of outputs
        proj_hidden_dim: integer, default=512
            number of hidden units
    """

    def __init__(self, backbone: str = "resnet50", output_size: int = 3, n_hidden: int = 512) -> None:

        super(NoneLinearRegressor, self).__init__()

        self.backbone = BACKBONES[backbone]
        self.backbone.fc = nn.Identity()

        self.fc = nn.Sequential(
            nn.Linear(in_features=self.backbone.in_planes, out_features=n_hidden),
            nn.ReLU(),
            nn.BatchNorm1d(n_hidden),
            nn.Dropout(0.3),
            nn.Linear(in_features=n_hidden, out_features=n_hidden),
            nn.ReLU(),
            nn.BatchNorm1d(n_hidden),
            nn.Dropout(0.3),
            nn.Linear(in_features=n_hidden, out_features=output_size),
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
