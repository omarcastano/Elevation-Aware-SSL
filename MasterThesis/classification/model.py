import torch
from torch import nn
import numpy as np
from sklearn.metrics import confusion_matrix
from tqdm.notebook import tqdm
import wandb
import os

os.environ["WANDB_SILENT"] = "true"
from MasterThesis.backbone import resnet18, resnet50
from MasterThesis.classification.metrics import model_evaluation, threshold_metric_evaluation, plot_metrics_from_logs


BACKBONES = {
    "resnet18": resnet18,
    # "resnet34": resnet34(weights=None),
    "resnet50": resnet50,
}


def linear_block(in_features, out_features):

    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.ReLU(),
        nn.BatchNorm1d(out_features),
    )


class Classifier(nn.Module):
    """
    Multi-class classifier model on top a ResNet backbone

    Arguments:
    ----------
        backbone: string, default = "resnet50"
            backbone to be pre-trained
        num_classes: integer, default=None
            number of different classes
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
        num_classes: int = 3,
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
            nn.Linear(in_features=num_hidden_units, out_features=num_classes),
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
        self.loss = torch.nn.CrossEntropyLoss()

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
    def train_one_epoch(
        self,
        train_loader: torch.utils.data.DataLoader,
    ):

        """
        Arguments:
        ----------
            train_dataloader: pytorch dataloader with training images and labels
        """

        running_loss = 0
        conf_mt = 0

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

            # compute confusion matrix
            # labels = labels.cpu().detach().numpy()
            # outputs = outputs.argmax(-1).cpu().detach().numpy()
            # conf_mt += confusion_matrix(labels, outputs, labels=list(range(len(class_name))))

            # bar.set_description(f"(Train_loss:{round(running_loss/epoch, 3)}) ")

        self.scheduler.step()
        running_loss = round(running_loss / epoch, 4)
        # scores, logs = metrics.model_evaluation(conf_mt, class_name=class_name, dataset_label="Train")
        logs = {"train_total_loss": running_loss}

        return logs

    # test one epoch
    def test_one_epoch(self, test_loader: torch.utils.data.DataLoader, last_epoch: bool = False, **kwargs):
        """
        Evaluate model performance using a test set with different metrics

        Arguments:
        ----------
            test_loader: pytorch dataloader with training images and labels
        """

        conf_mt = 0
        running_loss = 0
        logs = {}
        metrics_by_threshold = threshold_metric_evaluation(self.class_names)

        y_true = []
        y_pred = []
        y_pred_proba = []
        self.eval()

        with torch.no_grad():

            # bar = tqdm(test_loader, leave=True, position=2)
            for epoch, (inputs, labels) in enumerate(test_loader, 1):

                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Make prediction
                outputs = self(inputs)
                # Compute de los
                loss = self.loss(outputs, labels)
                running_loss += loss.item()

                # Confusion matrix
                labels = labels.cpu().detach().numpy()
                output = outputs.argmax(1).cpu().detach().numpy()
                output_proba = torch.nn.functional.softmax(outputs, dim=-1).cpu().detach().numpy()  # I have to apply soft max

                y_true.append(labels)
                y_pred.append(output)
                y_pred_proba.append(output_proba)  # I have to apply soft max
                # bar.set_description(f"(Test_loss:{round(running_loss/epoch, 3)}) ")

        y_true = np.hstack(y_true)
        y_pred = np.hstack(y_pred)
        y_pred_proba = np.vstack(y_pred_proba)

        conf_mt = confusion_matrix(y_true, y_pred, labels=list(range(len(self.class_names))))
        running_loss = np.round(running_loss / epoch, 4)

        logs = model_evaluation(conf_mt, class_name=self.class_names, dataset_label="Test")
        logs.update({"test_total_loss": running_loss})

        if last_epoch:
            metrics_by_threshold.metric_evaluation(y_true, y_pred_proba)
            logs.update({"metric_evaluation": metrics_by_threshold})

        return logs

    @staticmethod
    def log_one_epoch(logs_train, logs_test, last_epoch):

        if last_epoch:
            metrics_by_threshold = logs_test["metric_evaluation"]
            metrics_table = wandb.Table(dataframe=metrics_by_threshold.get_table())
            wandb.log({"Table_Metrics": metrics_table})

            wandb.log({"Per Class Accuracy": plot_metrics_from_logs(logs_test, metric="Acc_by_Class")})
            wandb.log({"Recall": plot_metrics_from_logs(logs_test, metric="Recall")})
            wandb.log({"F1 Score": plot_metrics_from_logs(logs_test, metric="F1_score")})
            wandb.log({"Precision": plot_metrics_from_logs(logs_test, metric="Precision")})
            wandb.log({"Precision Recall Curve": metrics_by_threshold.plot_PR_curve()})
            wandb.log({"Precision by Threshold": metrics_by_threshold.get_bar_plot(metric="precision")})
            wandb.log({"Recall by Thresholds": metrics_by_threshold.get_bar_plot(metric="recall")})
            wandb.log({"F1_Score by Threshold": metrics_by_threshold.get_bar_plot(metric="f1_score")})  # Run a experiment

            fig = plot_metrics_from_logs(logs_test, metric="F1_score")
            fig.show()

            logs_test.pop("metric_evaluation")

        logs_train.update(logs_test)
        wandb.log(logs_train)
