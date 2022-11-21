import os
import torch

from torch import nn
from typing import List, Callable
from tqdm.notebook import tqdm
from MasterThesis.ssl.elevation.model import ElevationSSL

import wandb

MODELS = {"ElevationSSL": ElevationSSL}


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

    # train model
    def fit(self):
        """
        Fit deep learning model
        """
        if not self.wandb_kwargs["id"]:
            run_id = wandb.util.generate_id()
            print("--------------------")
            print("run_id", run_id)
            print("--------------------")

        # Definte paths to save and load model
        checkpoint_path = f"{self.metadata_kwargs['path_to_save_model']}/checkpoint_{self.wandb_kwargs['name']}.pt"
        model_path = f"{self.metadata_kwargs['path_to_save_model']}/model_{self.wandb_kwargs['name']}.pth"
        checkpoint_load_path = f"{self.metadata_kwargs['path_to_load_model']}/checkpoint_{self.wandb_kwargs['name']}.pt"

        # Create folder to save model
        if self.metadata_kwargs["path_to_save_model"]:
            if not os.path.isdir(self.metadata_kwargs["path_to_save_model"]):
                os.mkdir(self.metadata_kwargs["path_to_save_model"])

        # Initialize WandB
        self.wandb_kwargs.update({"config": self.hypm_kwargs})
        with wandb.init(**self.wandb_kwargs):

            if self.metadata_kwargs["path_to_load_model"]:
                checkpoint = torch.load(checkpoint_load_path)
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.model.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                epoch = checkpoint["epoch"]
                bar = tqdm(range(epoch, self.hypm_kwargs["epochs"] + 1), desc=f"Epoch {epoch}/{self.hypm_kwargs['epochs']} ")
            else:
                bar = tqdm(range(1, self.hypm_kwargs["epochs"] + 1), desc=f"Epoch 1/{self.hypm_kwargs['epochs']} ")

            # Start Training the model
            for epoch in bar:
                logs_train = self.model.train_one_epoch(self.train_loader)
                logs_test = self.model.test_one_epoch(self.test_loader)

                self.model.log_one_epoch(logs_train, logs_test)

                bar.set_description(
                    f"Epoch {epoch}/{self.hypm_kwargs['epochs']} "
                    f"Train_loss:{round(logs_train['train_total_loss'], 3)} "
                    f"Test_loss:{round(logs_test['test_total_loss'], 3)} "
                )

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
                        checkpoint_path,
                    )

                    torch.save(self.model, model_path)

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

        # Defines sample dataset to visualize images
        ds_train_sample = self.custom_dataloader(
            metadata=self.metadata_kwargs["metadata_train"],
            return_original=True,
            normalizing_factor=self.hypm_kwargs["normalizing_factor"],
            augment=self.hypm_kwargs["augment_train"],
            **self.metadata_kwargs,
        )

        self.visualizer(ds_train_sample)

        # define train dataloader
        self.train_loader = torch.utils.data.DataLoader(
            ds_train,
            batch_size=self.hypm_kwargs["train_batch_size"],
            shuffle=True,
            num_workers=self.metadata_kwargs["num_workers"],
            drop_last=True,
        )

        # Define test dataloader
        self.test_loader = torch.utils.data.DataLoader(
            ds_test,
            batch_size=self.hypm_kwargs["test_batch_size"],
            shuffle=True,
            num_workers=self.metadata_kwargs["num_workers"],
            drop_last=True,
        )

        # Instance model
        self.model = MODELS[self.hypm_kwargs["model_name"]](**self.hypm_kwargs)
        self.model.configure_model(self.metadata_kwargs["device"], self.hypm_kwargs)
