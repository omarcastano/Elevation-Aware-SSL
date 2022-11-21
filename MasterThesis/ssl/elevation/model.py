from MasterThesis.backbone.resnet import resnet50, resnet18
import torch
from torch import nn


from typing import List
from MasterThesis.segmentation.sl.model import Unet
from torch.utils.data.dataloader import DataLoader
from lightly.loss import NTXentLoss
from tqdm import tqdm

import wandb

BACKBONES = {
    "resnet18": resnet18,
    # "resnet34": resnet34(weights=None),
    "resnet50": resnet50,
}


class ElevationSSL(nn.Module):
    """
    Self supervised method that uses elevation maps to boost contrastive learning

    Arguments
    ----------
    backbone : str, default="resnet50"
        Name of backbone model. Possible options are: resnet18, resnet50
    proj_hidden_dim : int, default=512
        Number of hidden unit in the projection header used in
        the contrastive module.
    proj_output_dim : int, optional
        Dimensionality of the projected embedding in the metric used in
        the contrastive module
    decoder_channels: list of integers, default=[256, 128, 64, 32, 16]
        specify **in_channels** parameter for convolutions used in decoder.
        This parameter can be used along with output_size to control the hight
        and width of the output mask
    input_size: int, default = 100
        hight and width of the input image
    output_size: int, default = 100
        hight and width of the output mask
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        proj_hidden_dim: int = 512,
        proj_output_dim: int = 128,
        decoder_channels: List = [256, 128, 64],
        input_size: int = 100,
        output_size: int = 33,
        **kwargs,
    ) -> None:

        super().__init__()

        self.unet = Unet(backbone, decoder_channels=decoder_channels, input_size=input_size, output_size=output_size)

        self.backbone = self.unet.encoder.backbone

        self.projector = nn.Sequential(
            nn.Linear(in_features=self.unet.encoder.backbone.in_planes, out_features=proj_hidden_dim, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.Linear(in_features=proj_hidden_dim, out_features=proj_output_dim, bias=False),
        )

    def forward(self, x, x_i, x_j):

        x_i = self.backbone(x_i)
        x_j = self.backbone(x_j)

        z_i = torch.flatten(x_i, 1)
        z_i = self.projector(z_i)

        z_j = torch.flatten(x_j, 1)
        z_j = self.projector(z_j)

        mask = self.unet(x)

        return z_i, z_j, mask

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

        self.contrastive_loss = NTXentLoss(temperature=hypm_kwargs["temperature"])
        self.elevation_loss = torch.nn.MSELoss()

        # Define loss function
        self.loss = [self.contrastive_loss, self.elevation_loss]

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
    def train_one_epoch(self, train_loader: DataLoader):
        """
        Train model for one epoch

        Arguments:
        ----------
            train_dataloader: pytorch dataloader
                pytorch dataloader with training images and masks
        """

        running_loss = 0
        running_contrastive_loss = 0
        running_regression_loss = 0

        self.train()
        bar = tqdm(train_loader, leave=False, position=1)
        for epoch, (input1, input2, input3, mask) in enumerate(bar, 1):

            # Set zero gradients for every batch
            self.optimizer.zero_grad()

            # Input image and elevation map
            input1 = input1.to(self.device)
            mask = mask.to(self.device)

            # Augmented views of image1
            input2, input3 = input2.to(self.device), input3.to(self.device)

            # Get the keys and Queries
            q, k, pred_mask = self(input1, input2, input3)

            # Compute the total loss
            pred_mask = pred_mask.squeeze()
            contrastive_loss = self.loss[0](q, k)
            regression_loss = self.loss[1](mask, pred_mask)
            loss = contrastive_loss + regression_loss

            # compute gradients
            loss.backward()

            # Update weeigths
            self.optimizer.step()

            running_loss += loss.item()
            running_contrastive_loss += contrastive_loss.item()
            running_regression_loss += regression_loss.item()

            bar.set_description(
                f"(Trai_loss:{round(running_loss/epoch, 3)}) "
                f"(Train_contrastive_loss:{round(running_contrastive_loss/epoch, 3)}) "
                f"(Train_regression_loss:{round(running_regression_loss/epoch ,3)})"
            )

        self.scheduler.step()
        running_loss = running_loss / epoch
        running_contrastive_loss = running_contrastive_loss / epoch
        running_regression_loss = running_regression_loss / epoch

        logs = {
            "train_total_loss": running_loss,
            "train_contrastive_loss": running_contrastive_loss,
            "train_regression_loss": running_regression_loss,
        }

        return logs

    # Test one epoch
    def test_one_epoch(self, test_loader: DataLoader):
        """
        Evaluate model performance using a test set

        Arguments:
            test_dataloader: pytorch dataloader
                pytorch dataloader with training images and masks

        """

        running_loss = 0
        running_contrastive_loss = 0
        running_regression_loss = 0

        # self.eval()

        bar = tqdm(test_loader, leave=False, position=1)

        with torch.no_grad():
            for epoch, (input1, input2, input3, mask) in enumerate(bar, 1):

                # Input image and elevation map
                input1 = input1.to(self.device)
                mask = mask.to(self.device)

                # Augmented views of image1
                input2, input3 = input2.to(self.device), input3.to(self.device)

                # Get the keys and Queries
                q, k, pred_mask = self(input1, input2, input3)

                # Compute the total loss
                pred_mask = pred_mask.squeeze()
                contrastive_loss = self.loss[0](q, k)
                regression_loss = self.loss[1](mask, pred_mask)
                loss = contrastive_loss + regression_loss

                running_loss += loss.item()
                running_contrastive_loss += contrastive_loss.item()
                running_regression_loss += regression_loss.item()

                bar.set_description(
                    f"(Test_loss:{round(running_loss/epoch, 3)}) "
                    f"(Test_contrastive_loss:{round(running_contrastive_loss/epoch, 3)}) "
                    f"(Test_regression_loss:{round(running_regression_loss/epoch ,3)})"
                )

        running_loss = running_loss / epoch
        running_contrastive_loss = running_contrastive_loss / epoch
        running_regression_loss = running_regression_loss / epoch

        logs = {
            "test_total_loss": running_loss,
            "test_contrastive_loss": running_contrastive_loss,
            "test_regression_loss": running_regression_loss,
        }

        return logs

    @staticmethod
    def log_one_epoch(logs_train, logs_test):

        wandb.log(logs_train)
        wandb.log(logs_test)
