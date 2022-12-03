import torch
from torch import nn
import numpy as np
from MasterThesis.backbone import resnet50, resnet18
import wandb


torch.manual_seed(42)

BACKBONES = {
    "resnet18": resnet18,
    "resnet50": resnet50,
}


def linear_block(in_features, out_features):

    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.ReLU(),
        nn.BatchNorm1d(out_features),
    )


class Regressor(nn.Module):
    """
    Multi-class classifier model on top a ResNet backbone

    Arguments:
    ----------
        backbone: string, default = "resnet50"
            backbone to be pre-trained
        output_size: integer, default=None
            number of outputs
        num_hidden_layers: int, default=0
            number of hidden layer to add to the fully connected block
        num_hidden_units: int, default=512
            number of hidden unit to set to the hidden layers
        cifar: boolean, default=False
            If True removes the first max pooling layer, and the kernel in the
            first convolutional layer is change from 7x7 to 3x3.
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        output_size: int = 2,
        num_hidden_layers: int = 0,
        num_hidden_units: int = 512,
        cifar: bool = False,
        **kwargs,
    ) -> None:

        super().__init__()

        self.backbone = BACKBONES[backbone](cifar=cifar)
        self.backbone.fc = nn.Identity()

        if num_hidden_layers > 0:
            self.layer1 = linear_block(self.backbone.in_planes, num_hidden_units)
            layers = [linear_block(num_hidden_units, num_hidden_units) for _ in range(num_hidden_layers - 1)]
            self.layers = nn.ModuleList(layers)
        else:
            num_hidden_units = self.backbone.in_planes

        self.out_layer = nn.Sequential(
            nn.Linear(in_features=num_hidden_units, out_features=output_size),
        )

        self.num_hidden_layers = num_hidden_layers

        self._init_weight()

    def forward(self, x):

        x = self.backbone(x)

        if self.num_hidden_layers > 0:
            x = self.layer1(x)
            for layer in self.layers:
                x = layer(x)

        x = self.out_layer(x)

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
        self.class_names = hypm_kwargs["class_names"]

        # Define loss function
        self.loss = torch.nn.MSELoss()

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            weight_decay=hypm_kwargs["weight_decay"],
            lr=hypm_kwargs["learning_rate"],
        )

        # learning scheduler
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)

        # device
        self.device = device

    # define training one epoch
    def train_one_epoch(self, train_loader: torch.utils.data.DataLoader):

        """
        Arguments:
        ----------
            train_dataloader: pytorch dataloader with training images and labels
        """

        running_loss = 0

        self.train()
        # bar = tqdm(train_loader, leave=True, position=1)
        for epoch, (image, label) in enumerate(train_loader, 1):

            images, labels = image.to(self.device), label.to(self.device)

            # set gradients to zero
            self.optimizer.zero_grad()

            # prediction
            outputs = self(images)

            # compute loss
            loss = self.loss(outputs, labels)

            # compute gradients
            loss.backward()

            # update weigths
            self.optimizer.step()

            running_loss += loss.item()

            # bar.set_description(f"(Train_loss:{round(running_loss/epoch, 3)}) ")

        self.scheduler.step()
        running_loss = round(running_loss / epoch, 4)
        # scores, logs = metrics.model_evaluation(conf_mt, class_name=class_name, dataset_label="Train")
        logs = {"train_total_loss": running_loss, "train_RMSE": np.sqrt(running_loss)}

        return logs

    # test one epoch
    def test_one_epoch(self, test_loader: torch.utils.data.DataLoader, **kwargs):
        """
        Evaluate model performance using a test set with different metrics

        Arguments:
        ----------
            test_loader: pytorch dataloader with training images and labels
        """

        conf_mt = 0
        running_loss = 0
        logs = {}

        self.eval()

        with torch.no_grad():

            # bar = tqdm(test_loader, leave=True, position=2)
            for epoch, (inputs, labels) in enumerate(test_loader, 1):

                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Make prediction
                outputs = self(inputs)

                # Compute de los
                loss = self.loss(outputs, labels)

                # Compute running loss
                running_loss += loss.item()

                # bar.set_description(f"(Test_loss:{round(running_loss/epoch, 3)}) ")

        running_loss = round(running_loss / epoch, 4)
        logs = {"test_total_loss": running_loss, "test_RMSE": np.sqrt(running_loss)}

        return logs

    @staticmethod
    def log_one_epoch(logs_train, logs_test, **kwargs):

        logs_train.update(logs_test)
        wandb.log(logs_train)
