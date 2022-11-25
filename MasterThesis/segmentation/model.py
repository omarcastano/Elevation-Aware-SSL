import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix
from typing import List
from tqdm.notebook import tqdm
from . import ResNetDecoder, ResNetEncoder, SegmentationHead
from MasterThesis.classification.metrics import model_evaluation, threshold_metric_evaluation, plot_metrics_from_logs
import wandb


class Unet(nn.Module):
    """
    Unet is a fully convolution neural network for image semantic segmentation. Consist of *encoder*
    and *decoder* parts connected with *skip connections*. Encoder extract features of different spatial
    resolution (skip connections) which are used by decoder to define accurate segmentation mask. Use *concatenation*
    for fusing decoder blocks with skip connections.ResNet-based Unet Encoder

    Arguments
    ----------
        backbone : str, default=resnet50
            Name of the classification model that will be used as an encoder (a.k.a backbone)
            to extract features of different spatial resolution. Possible options are: resnet18, resnet50
        encoder_depth: int, default=5
            a number of stages used in encoder. Each stage generate features two times smaller in spatial
            dimensions than previous one (e.g. for depth 0 we will have features with shapes [(N, C, H, W),],
            for depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
        decoder_channels: list of integers, default=[256, 128, 64, 32, 16]
            specify **in_channels** parameter for convolutions used in decoder.
        input_size: int, default = 100
            hight and width of the input image
        output_size: int, default = 100
            hight and width of the output mask
        number_classes: int, default=1
            the number of classes for output mask (or you can think as a number of channels of output mask)
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        encoder_depth: int = 5,
        decoder_channels: List = [256, 128, 64, 32, 16],
        input_size: int = 100,
        num_classes: int = 1,
        output_size: int = 100,
        **kwargs
    ) -> None:
        super().__init__()

        self.backbone_name = backbone
        self.encoder_depth = encoder_depth
        self.decoder_channels = decoder_channels
        self.decoder_depth = len(decoder_channels)
        self.input_size = input_size
        self.scale_factor, self.decoder_input_size = self.get_upscale_factors()

        self.encoder = ResNetEncoder(self.backbone_name, self.encoder_depth)
        self.backbone = self.encoder.backbone
        self.decoder = ResNetDecoder(self.encoder.backbone.out_channels, self.decoder_channels, self.scale_factor)

        self.header = SegmentationHead(decoder_channels[-1], num_classes, self.decoder_input_size[-1], output_size)

        self._init_weights()

    def get_upscale_factors(self):

        in_size = self.input_size
        out_size = in_size
        scale_factor = []
        output_size = [in_size]
        for _ in range(self.encoder_depth):
            in_size = out_size
            out_size = np.ceil(in_size / 2)
            output_size.append(out_size)
            scale_factor.append(in_size / out_size)

        scale_factor = scale_factor[::-1]
        output_size = output_size[::-1]

        return scale_factor[: self.decoder_depth], output_size[1 : self.decoder_depth + 1]

    def forward(self, x):

        encoder_output = self.encoder(x)
        decoder_output = self.decoder(*encoder_output)
        mask = self.header(decoder_output)

        return mask

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

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
    def train_one_epoch(self, train_dataloader: DataLoader):
        """
        Arguments:
        ----------
            train_dataloader: pytorch dataloader
                images and labels to train the model
        """

        running_loss = 0

        # bar = tqdm(train_loader, leave=True, position=1)
        self.train()
        for epoch, (image, label) in enumerate(train_dataloader, 1):

            images, labels = image.to(self.device), label.to(self.device)

            # set gradies to zero
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
            # outputs = outputs.argmax(1).cpu().detach().numpy()
            # conf_mt += metrics.pixel_confusion_matrix(labels, outputs, class_num=len(class_name))

        self.scheduler.step()
        running_loss = np.round(running_loss / epoch, 4)
        # scores, logs = metrics.model_evaluation(conf_mt, class_name=class_name, dataset_label="Train")
        logs = {"train_total_loss": running_loss}

        return logs

    def test_one_epoch(self, test_loader: DataLoader, last_epoch: bool = False):
        """
        Evaluate model performance using a test set with different metrics

        Arguments:
        ----------
            test_dataloader: pytorch dataloader
                data to test model
        """

        running_loss = 0
        logs = {}
        metrics_by_threshold = threshold_metric_evaluation(self.class_names)
        conf_mt = 0

        y_true = []
        y_pred = []
        y_pred_proba = []

        self.eval()

        # bar = tqdm(test_loader, leave=True, position=2)
        with torch.no_grad():
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
                output_proba = torch.nn.functional.softmax(outputs, dim=1).cpu().detach().numpy()  # I have to apply soft max
                output_proba = np.hstack([output_proba[:, i, :, :].reshape(-1, 1) for i in range(len(self.class_names))])

                y_true.append(labels.flatten())
                y_pred.append(output.flatten())
                y_pred_proba.append(output_proba)  # I have to apply soft max
                # bar.set_description(f"(Test_loss:{round(running_loss/epoch)})"),

        y_true = np.hstack(y_true)
        y_pred = np.hstack(y_pred)
        y_pred_proba = np.vstack(y_pred_proba)

        conf_mt = confusion_matrix(y_true, y_pred, labels=list(range(len(self.class_names))))
        running_loss = np.round(running_loss / epoch, 4)

        logs = model_evaluation(conf_mt, class_name=self.class_names, dataset_label="Test")
        logs.update({"test_total_loss": running_loss})

        # if last_epoch:
        #     metrics_by_threshold.metric_evaluation(y_true, y_pred_proba)
        #     logs.update({"metric_evaluation": metrics_by_threshold})

        return logs

    @staticmethod
    def log_one_epoch(logs_train, logs_test, last_epoch):

        # if last_epoch:
        #     metrics_by_threshold = logs_test["metric_evaluation"]
        #     logs_test.pop("metric_evaluation")

        logs_train.update(logs_test)
        wandb.log(logs_train)

        # if last_epoch:
        #    metrics_table = wandb.Table(dataframe=metrics_by_threshold.get_table())
        #    wandb.log({"Table_Metrics": metrics_table})

        #    wandb.log({"Per Class Accuracy": plot_metrics_from_logs(logs_test, metric="Acc_by_Class")})
        #    wandb.log({"Recall": plot_metrics_from_logs(logs_test, metric="Recall")})
        #    wandb.log({"F1 Score": plot_metrics_from_logs(logs_test, metric="F1_score")})
        #    wandb.log({"Precision": plot_metrics_from_logs(logs_test, metric="Precision")})
        #    wandb.log({"MIoU": plot_metrics_from_logs(logs_test, metric="IoU")})
        #    wandb.log({"Precision Recall Curve": metrics_by_threshold.plot_PR_curve()})
        #    wandb.log({"Precision by Threshold": metrics_by_threshold.get_bar_plot(metric="precision")})
        #    wandb.log({"Recall by Thresholds": metrics_by_threshold.get_bar_plot(metric="recall")})
        #    wandb.log({"F1_Score by Threshold": metrics_by_threshold.get_bar_plot(metric="f1_score")})  # Run a experiment
