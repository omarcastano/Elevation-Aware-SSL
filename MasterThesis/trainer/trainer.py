import os
import torch

from torch import nn
from typing import List, Callable
from tqdm.notebook import tqdm
from MasterThesis.ssl.elevation.model import ElevationSSL
from MasterThesis.classification.model import Classifier
from MasterThesis.ssl.simclr.models import SimCLR
from MasterThesis.segmentation.model import Unet
from MasterThesis.classification.utils import EarlyStopping

import wandb

os.environ["WANDB_SILENT"] = "true"
MODELS = {
    "ElevationSSL": ElevationSSL,
    "Classifier": Classifier,
    "SimCLR": SimCLR,
    "Unet": Unet,
}


class Trainer:
    """
    Train a deep learning model

    Arguments
    ----------
    train_loader : DataLoader
        dataloader to train SimCLR
    test_loader : DataLoader
        dataloader to test SimCLR
    model : Module
        SimCLR model
    wandb_kwargs : dict
        kwargs used to load metrics to wandb
    hypm_kwargs: dict
        dictionary with the hyperparameter to configure the model
    metadata_kwargs : dict
        dictionary with the metadata to save and load trained models
    """

    def __init__(
        self,
        custom_dataloader: nn.Module,
        visualizer: Callable,
        wandb_kwargs: dict,
        hypm_kwargs: dict,
        metadata_kwargs: dict,
    ) -> None:

        self.custom_dataloader = custom_dataloader
        self.wandb_kwargs = wandb_kwargs
        self.hypm_kwargs = hypm_kwargs
        self.metadata_kwargs = metadata_kwargs
        self.visualizer = visualizer

        # Definte paths to save and load model
        self.checkpoint_path = f"{self.metadata_kwargs['path_to_save_model']}/checkpoint_{self.wandb_kwargs['name']}.pt"
        self.model_path = f"{self.metadata_kwargs['path_to_save_model']}/model_{self.wandb_kwargs['name']}.pth"
        self.checkpoint_load_path = f"{self.metadata_kwargs['path_to_load_model']}/checkpoint_{self.wandb_kwargs['name']}.pt"

    # train model
    def fit(self):
        """
        Fit deep learning model
        """

        if self.epoch >= self.hypm_kwargs["epochs"]:
            print(f"The model has been trained for {self.epoch}")
            return

        if not self.wandb_kwargs["id"]:
            run_id = wandb.util.generate_id()
            print("--------------------")
            print("run_id", run_id)
            print("--------------------")

        if self.metadata_kwargs["metadata_valid"] is not None:
            early_stopping = EarlyStopping(self.model, 1000, self.hypm_kwargs["patient"])

        # Initialize WandB
        self.wandb_kwargs.update({"config": self.hypm_kwargs})
        with wandb.init(**self.wandb_kwargs):

            # Start Training the model
            bar = tqdm(
                range(self.epoch, self.hypm_kwargs["epochs"] + 1), desc=f"Epoch {self.epoch}/{self.hypm_kwargs['epochs']} ", position=0
            )
            for epoch in bar:

                # Train the model for one epoch
                logs_train = self.model.train_one_epoch(self.train_loader)

                # Test the model for one epoch
                if (epoch % self.hypm_kwargs["eval_epoch"] == 0) | (epoch == 1):
                    last_epoch = epoch == self.hypm_kwargs["epochs"]
                    logs_test = self.model.test_one_epoch(self.test_loader, last_epoch=last_epoch)

                    if self.metadata_kwargs["metadata_valid"] is not None:
                        logs_valid = self.model.test_one_epoch(self.valid_loader, last_epoch=last_epoch)
                        stop_training = early_stopping(self.model, logs_valid["test_total_loss"])
                        if stop_training:
                            logs_test = early_stopping.best_model.test_one_epoch(self.test_loader, last_epoch=True)
                            self.model.log_one_epoch(logs_train, logs_test, last_epoch=True)
                            break

                    # Save the model
                    if self.metadata_kwargs["path_to_save_model"]:
                        torch.save(
                            {
                                "epoch": epoch,
                                "model_state_dict": self.model.state_dict(),
                                "optimizer_state_dict": self.model.optimizer.state_dict(),
                                "train_loss": logs_train["train_total_loss"],
                                "test_loss": logs_test["test_total_loss"],
                            },
                            self.checkpoint_path,
                        )

                        torch.save(self.model, self.model_path)

                self.model.log_one_epoch(logs_train, logs_test, last_epoch=last_epoch)

                bar.set_description(
                    f"Epoch {epoch}/{self.hypm_kwargs['epochs']} "
                    f"Train_loss:{round(logs_train['train_total_loss'], 3)} "
                    f"Test_loss:{round(logs_test['test_total_loss'], 3)} "
                )

                if self.hypm_kwargs["fine_tune"] and (self.hypm_kwargs["ft_epoch"] == epoch):

                    for parameters in self.model.backbone.parameters():
                        parameters.requires_grad = True

                    for g in self.model.optimizer.param_groups:
                        g["lr"] = self.hypm_kwargs["ft_lr"]

    def configure_trainer(self):

        # Define train dataset
        ds_train = self.custom_dataloader(
            metadata=self.metadata_kwargs["metadata_train"],
            normalizing_factor=self.hypm_kwargs["normalizing_factor"],
            augment=self.hypm_kwargs["augment_train"],
            **self.metadata_kwargs,
        )

        # Define test dataset
        ds_test = self.custom_dataloader(
            metadata=self.metadata_kwargs["metadata_test"],
            normalizing_factor=self.hypm_kwargs["normalizing_factor"],
            augment=self.hypm_kwargs["augment_test"],
            **self.metadata_kwargs,
        )

        # Defines validation dataset
        if self.metadata_kwargs["metadata_valid"] is not None:
            ds_valid = self.custom_dataloader(
                metadata=self.metadata_kwargs["metadata_valid"],
                normalizing_factor=self.hypm_kwargs["normalizing_factor"],
                augment=self.hypm_kwargs["augment_test"],
                **self.metadata_kwargs,
            )

        # Defines sample dataset to visualize images
        ds_train_sample = self.custom_dataloader(
            metadata=self.metadata_kwargs["metadata_train"],
            return_original=True,
            normalizing_factor=self.hypm_kwargs["normalizing_factor"],
            augment=self.hypm_kwargs["augment_train"],
            **self.metadata_kwargs,
        )

        # self.visualizer(ds_train_sample, **self.hypm_kwargs)

        # define train dataloader
        self.train_loader = torch.utils.data.DataLoader(
            ds_train,
            batch_size=self.hypm_kwargs["train_batch_size"],
            shuffle=False,
            num_workers=self.metadata_kwargs["num_workers"],
            drop_last=True,
        )

        # Define test dataloader
        self.test_loader = torch.utils.data.DataLoader(
            ds_test,
            batch_size=self.hypm_kwargs["test_batch_size"],
            shuffle=False,
            num_workers=self.metadata_kwargs["num_workers"],
            drop_last=True,
        )

        # Define validation dataloader
        if self.metadata_kwargs["metadata_valid"] is not None:
            self.valid_loader = torch.utils.data.DataLoader(
                ds_valid,
                batch_size=self.hypm_kwargs["test_batch_size"],
                shuffle=True,
                num_workers=self.metadata_kwargs["num_workers"],
                drop_last=True,
            )

        # Set random seed
        torch.manual_seed(0)

        # Create folder to save model
        if self.metadata_kwargs["path_to_save_model"]:
            if not os.path.isdir(self.metadata_kwargs["path_to_save_model"]):
                os.makedirs(self.metadata_kwargs["path_to_save_model"])

        # Instance model
        if self.metadata_kwargs["path_to_load_model"]:
            checkpoint = torch.load(self.checkpoint_load_path)
            self.model = torch.load(self.model_path)
            self.model.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.epoch = checkpoint["epoch"]

            print("----------------------------------")
            print("The model was loaded successfully")
            print("----------------------------------")

        else:
            self.epoch = 1
            self.model = MODELS[self.hypm_kwargs["model_name"]](**self.hypm_kwargs)
            self.model.configure_model(self.metadata_kwargs["device"], self.hypm_kwargs)

        if self.hypm_kwargs["pretrained"]:
            self.model.backbone.load_state_dict(torch.load(self.metadata_kwargs["path_to_load_backbone"]).backbone.state_dict())

            for parameters in self.model.backbone.parameters():
                parameters.requires_grad = False

            self.model.to(self.metadata_kwargs["device"])

            print("\n --------------------------------------")
            print("The backbone was transfered successfully")
            print("----------------------------------------\n")
