import torch
import wandb
from torch import nn
from MasterThesis.backbone import resnet18, resnet50

from typing import List
from torch.utils.data.dataloader import DataLoader
from lightly.loss import NTXentLoss
from tqdm import tqdm


BACKBONES = {
    "resnet18": resnet18,
    # "resnet34": resnet34(weights=None),
    "resnet50": resnet50,
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
        self, backbone: str = "resnet50", proj_hidden_dim: int = 512, proj_output_dim: int = 128, cifar: bool = False, **kwargs
    ) -> None:

        super(SimCLR, self).__init__()

        self.backbone = BACKBONES[backbone](cifar=cifar)

        self.backbone.fc = nn.Identity()

        self.projector = nn.Sequential(
            nn.Linear(in_features=self.backbone.in_planes, out_features=proj_hidden_dim, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.Linear(in_features=proj_hidden_dim, out_features=proj_output_dim, bias=False),
        )

        self._init_weight()

    def forward(self, x):

        x = self.backbone(x)
        x = self.projector(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)

    def configure_model(self, device: str, hypm_kwargs: dict):
        """
        configures models hyperparameters such as training loss, optimizer,
        learning rate scheduler

        Parameters
        ----------
        device : str
            device to train and test model
        wandb_kwargs : dictionary
            dictionary with the following keys
                {'temperature': temperature for the contrastive loss
                 'learning_rate': learning rate for AnadW optimizer
                 'weight_decay': weight decay for AnadW optimizer}
                 'epochs' number of epoch for CosineAnnealingLR lr_scheduler:
        """

        self.to(device)

        # Define loss function
        self.loss = NTXentLoss(temperature=hypm_kwargs["temperature"])

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            weight_decay=hypm_kwargs["weight_decay"],
            lr=hypm_kwargs["learning_rate"],
        )

        # learning scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=hypm_kwargs["epochs"], eta_min=4e-08)

        # device
        self.device = device

    # Train one epoch
    def train_one_epoch(self, train_loader: DataLoader, **kwargs):
        """
        Train SimCLR model for one epoch

        Arguments:
        ----------
            train_dataloader: pytorch dataloader
                pytorch dataloader with training images and labels
        """

        running_loss = 0

        self.train()
        bar = tqdm(train_loader, leave=True, position=1)
        for epoch, (input1, input2) in enumerate(bar, 1):

            # Set zero gradients for every batch
            self.optimizer.zero_grad()

            input1, input2 = input1.to(self.device), input2.to(self.device)

            # Get the keys and Queries
            q = self(input1)
            k = self(input2)

            # Compute the total loss
            loss = self.loss(q, k)

            # compute gradients
            loss.backward()

            # Update weeigths
            self.optimizer.step()

            running_loss += loss.item()

            bar.set_description(f"(Trai_loss:{round(running_loss/epoch, 3)})")

        self.scheduler.step()
        running_loss = running_loss / epoch

        logs = {"train_total_loss": running_loss}

        return logs

    # Test one epoch
    def test_one_epoch(self, test_loader: DataLoader, **kwargs):
        """
        Evaluate SimCLR representation learning in one epoch

        Arguments:
        ----------
            test_dataloader: pytorch dataloader
                pytorch dataloader with training images and labels
        """

        running_loss = 0

        # model.eval()

        bar = tqdm(test_loader, leave=True, position=1)

        with torch.no_grad():
            for epoch, (input1, input2) in enumerate(bar, 1):

                input1, input2 = input1.to(self.device), input2.to(self.device)

                # Get the keys and Queries
                q = self(input1)
                k = self(input2)

                # Compute the total loss
                loss = self.loss(q, k)

                running_loss += loss.item()

                bar.set_description(f"(Test_loss:{round(running_loss/epoch, 3)})")

        running_loss = running_loss / epoch

        logs = {"test_total_loss": running_loss}

        return logs

    @staticmethod
    def log_one_epoch(logs_train: dict, logs_test: dict, **kwargs):
        """
        logs metrics to wandb

        Arguments:
        ----------
            logs_train : dict
                train metrics
            logs_test : dict
                test metrics
        """

        logs_train.update(logs_test)

        wandb.log(logs_train)
