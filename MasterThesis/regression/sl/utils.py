# python moduls
import os
from typing import List, Tuple

# Standard libraries
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


# torch moduls
import torch
from torch.utils.data import DataLoader
from torch.nn import Module

# Other modules
import wandb
from IPython.display import clear_output
from tqdm import tqdm


# Custom Importations
from MasterThesis import EDA
from MasterThesis.regression.sl import CustomDataset
from MasterThesis.regression.sl.supervised import (
    LinearRegressor,
    NoneLinearRegressor,
)

REGRESSORS = {"linear": LinearRegressor, "non_linear": NoneLinearRegressor}


def generate_metadata_train_test_cv(
    metadata: pd.DataFrame, train_size: float, n_split: int = 5
) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:

    """
    Provides train/test indices to split data in train/test sets. Split
    dataset into k consecutive folds train and test dataset via k-fold sets

    Arguments:
    ----------
        metadata: dataframe with the path to images and labels
        train_size: percentage of in each train set
        n_split: numbers of folds. Must be at least 2
    """

    n_total_images = metadata.shape[0]
    n_images_train = int(n_total_images * train_size)

    print(f"Number fo total images : {n_total_images}")

    metadata_train, metadata_test = [], []

    kf = KFold(n_splits=n_split)

    X = np.arange(10)
    for train, test in kf.split(metadata):
        metadata_train.append(metadata.iloc[train].sample(n_images_train, random_state=42).copy().reset_index(drop=True))
        metadata_test.append(metadata.iloc[test].copy().reset_index(drop=True))

    print(f"Number of images to train: {metadata_train[0].shape[0]} ({(metadata_train[0].shape[0]/n_total_images)*100:.3f}%)")
    print(f"Number of images to test: {metadata_test[0].shape[0]} ({(metadata_test[0].shape[0]/n_total_images)*100:.3f}%)")

    return metadata_train, metadata_test


def visualize_augmented_images(
    dataset: torch.utils.data.Dataset,
    n: int = 10,
    col_target: List = None,
    brightness: float = None,
) -> None:

    """
    Plots augmented images used to train a segmentation model

    Arguments:
    ----------
        dataset: pytorch dataset which must return the augmented
                 views of the same image and the image itself
        n: number of images to plot
        brightness: float, default=0.0
            factor to adjust the brightness of the images
    """

    fig, ax = plt.subplots(2, n, figsize=(32, 10))

    for i in range(n):
        original, augmented, label = dataset[i]

        augmented = np.array(augmented).transpose(1, 2, 0)
        original = np.array(original).transpose(1, 2, 0)

        ax[1, i].imshow(augmented + brightness)
        ax[0, i].imshow(original + brightness)

        s = ""
        for i_, j_ in zip(col_target, label.astype("str")):
            s += i_ + ":" + str(j_) + " \n"

        ax[0, i].set_title(f"Original")
        ax[1, i].set_title(f"Augmented \n {s}")

        ax[0, i].axis("off")
        ax[1, i].axis("off")

    plt.show()


# define training one epoch
def train_one_epoch(
    train_dataloader: DataLoader,
    model: Module,
    loss_fn: Module,
    optmizer: torch.optim,
    schedule: torch.optim.lr_scheduler,
    device: torch.device,
):

    """
    Arguments:
        train_dataloader: pytorch dataloader with training images and labels
        model: pytorch model for semantic segmentation
        loss_fn: loss function for semantic segmentation
        opimizer: pytorch optimizer
        schduler: pytorch learning scheduler
        device: device to tran the network sucha as cpu or cuda
    """

    running_loss = 0

    model.train()
    for epoch, (image, label) in enumerate(tqdm(train_dataloader), 1):

        images, labels = image.to(device), label.to(device)

        # set gradies to zero
        optmizer.zero_grad()

        # prediction
        outputs = model(images)

        # compute loss
        loss = loss_fn(outputs, labels)

        # compute gradients
        loss.backward()

        # update weigths
        optmizer.step()

        running_loss += loss.item()

    schedule.step()
    running_loss = np.round(running_loss / epoch, 4)
    logs = {"Train loss": running_loss, "train_RMSE": np.sqrt(running_loss)}

    return logs


# test one epoch
def test_one_epoch(
    test_loader: DataLoader,
    model: Module,
    loss_fn: Module,
    device: torch.device,
):

    """
    Evaluate model performance using a test set with different metrics

    Arguments:
        train_dataloader: pytorch dataloader with training images and labels
        model: pytorch model for semantic segmentation
        loss_fn: loss function for semantic segmentation
        device: device to tran the network sucha as cpu or cuda
    """

    running_loss = 0
    logs = {}

    model.eval()

    with torch.no_grad():
        for epoch, (inputs, labels) in enumerate(tqdm(test_loader), 1):

            inputs, labels = inputs.to(device), labels.to(device)

            # Make prediction
            outputs = model(inputs)

            # Compute de los
            loss = loss_fn(outputs, labels)
            running_loss += loss.item()

    running_loss = np.round(running_loss / epoch, 4)

    logs = {"Test loss": running_loss, "test_RMSE": np.sqrt(running_loss)}

    return logs


# train model
def train_model(
    train_loader: DataLoader,
    test_loader: DataLoader,
    model: Module,
    loss_fn,
    optimizer,
    schedule,
    wandb: wandb,
    wandb_kwargs: dict,
    metadata_kwargs: dict,
):

    """
    Train a semantic segmentation model

    Arguments:
    ---------
        metadata_kwargs: metadata required to train the model. Example:
            metadata_kwargs = {
                "path_to_labels": path_to_label,
                "path_to_images": path_to_images,
                "path_to_save_model": None,  # Path to save the model that is being trained (do not include the extension .pt or .pth)
                "path_to_load_model": None,  # Path to load a model from a checkpoint (useful to handle notebook disconection)
                "path_to_load_backbone": None,  # Path to load a pre-trained backbone using ssl
                "metadata": metadata,
                "metadata_train": metadata_train[0],
                "metadata_test": metadata_test[0],
                "select_classes": select_classes,
                "device": device,
            }

        wandb_kwargs: parameteres to initialize wandb runs. Example:
            wandb_kwargs = {
                "project": "RandomInit",
                "entity": "omar_castano",
                "config": hypm,
                "id": None,
                "name": "RandomInit",
                "resume": False,
            }

            Example of hyperparameteres expected:
            hypm = {
                "version": "version",
                "pretrained": "SimCLR", # SSL methodology. If not None path_to_load_backbone must be provided
                "fine_tune": True, # wheter or not to fine tune the the whole model (including the pre-trained backbone)
                "ft_epoch": 100, # epoch from which the frozen backbone will be unfreez
                "amount_of_ft_data": train_size,
                "bn_momentum": 0.9,
                "input_shape": [100, 100],
                "num_classes": 3,
                "in_channels": 3,
                "weight_decay": 0.0005,
                "learning_rate": 1e-3,
                "batch_size": 16,
                "epochs": 2,
            }
    """

    # Definte paths to save and load model
    checkpoint_path = f"{metadata_kwargs['path_to_save_model']}/checkpoint_{wandb_kwargs['name']}.pt"
    model_path = f"{metadata_kwargs['path_to_save_model']}/model_{wandb_kwargs['name']}.pth"
    checkpoint_load_path = f"{metadata_kwargs['path_to_load_model']}/checkpoint_{wandb_kwargs['name']}.pt"

    # Create folder to save model
    if metadata_kwargs["path_to_save_model"]:
        if not os.path.isdir(metadata_kwargs["path_to_save_model"]):
            os.mkdir(metadata_kwargs["path_to_save_model"])

    # Initialize WandB
    with wandb.init(**wandb_kwargs):

        epoch = 0

        if metadata_kwargs["path_to_load_model"]:
            checkpoint = torch.load(checkpoint_load_path)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            epoch = checkpoint["epoch"]
            train_loss = checkpoint["train_loss"]
            test_loss = checkpoint["test_loss"]

        train_loss = []
        test_loss = []

        # Start Training the model
        while epoch < wandb_kwargs["config"]["epochs"]:
            print("-------------------------------------------------------------")
            print(f"Epoch {epoch+1}/{wandb_kwargs['config']['epochs']}")

            logs_train = train_one_epoch(
                train_loader,
                model,
                loss_fn,
                optimizer,
                schedule,
                metadata_kwargs["device"],
            )

            if ((epoch + 1) % 5 == 0) | (epoch + 1 == 1):
                logs_test = test_one_epoch(
                    test_loader,
                    model,
                    loss_fn,
                    metadata_kwargs["device"],
                )

            print(f'\n    Train Loss: { logs_train["Train loss"] }')
            print(f'\n    Test Loss: { logs_test["Test loss"] }')

            train_loss.append(logs_train["Train loss"])
            test_loss.append(logs_test["Test loss"])

            logs_test.update(logs_train)
            wandb.log(logs_test)
            print("------------------------------------------------------------- \n")

            if wandb_kwargs["config"]["fine_tune"] and (wandb_kwargs["config"]["ft_epoch"] == epoch):

                for parameters in model.backbone.parameters():
                    parameters.requires_grad = True

                for g in optimizer.param_groups:
                    g["lr"] = wandb_kwargs["config"]["lr_ft"]

            # Save the model
            if metadata_kwargs["path_to_save_model"]:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "tarin_loss": train_loss,
                        "test_loss": test_loss,
                    },
                    checkpoint_path,
                )

                torch.save(model, model_path)

            epoch += 1

        # Create table with the losses
        loss = pd.DataFrame(
            {
                "train_loss": train_loss,
                "test_loss": test_loss,
                "epoch": np.array(range(len(train_loss))) + 1,
            }
        )

        # Save tables with metrics in wandB
        wandb.log({"Loss": wandb.Table(dataframe=loss)})
    clear_output(wait=True)


##Run a experiment
def run_train(
    wandb: wandb,
    wandb_kwargs: dict = None,
    metadata_kwargs: dict = None,
):

    """
    Run a experiment where a model is trained using resnet50 as backbone. A pre-trained
    encoder can be loaded and finetune during the training.

    Arguments:
    ---------
        metadata_kwargs: metadata required to train the model. Example:
            metadata_kwargs = {
                "path_to_labels": #Path to labels
                "path_to_images": # Path to images
                "path_to_save_model": None,  # Path to save the model that is being trained
                "path_to_load_model": None,  # Path to load a model from a checkpoint (useful to handle notebook disconnection)
                "path_to_load_backbone": None,  # Path to load a pre-trained backbone using ssl
                "metadata": # Dataframe with metadata to load images and labels
                "metadata_train": # Dataframe with metadata to load images and labels
                "metadata_test": # Dataframe with metadata to load images and labels
                "select_classes": # List with the unique classes
                "device": # device to train  de DeepLearning model
            }

        wandb_kwargs: parameters to initialize wandb runs. Example:
            wandb_kwargs = {
                "project": "RandomInit",
                "entity": "omar_castano",
                "config": hypm,
                "id": None,
                "name": "RandomInit", # This parameter is used to identify the saved model
                "resume": False,
            }

            Example of hyperparameteres expected:
            hypm = {
                "version": "version",
                "pretrained": "SimCLR", # SSL methodology. If not None path_to_load_backbone must be provided
                "fine_tune": True, # whether or not to fine tune the the whole model (including the pre-trained backbone)
                "ft_epoch": 100, # epoch from which the frozen backbone will be unfreez
                "amount_of_ft_data": train_size,
                "bn_momentum": 0.9,
                "input_shape": [100, 100],
                "num_classes": 3, # Hyperparameter of the model
                "in_channels": 3, # Hyperparameter of the model
                "weight_decay": 0.0005, # Hyperparameter of Adam optimizer
                "learning_rate": 1e-3, # Hyperparameter of Adam optimizer
                "batch_size": 16,  # Hyperparameter of the model
                "epochs": 2,  # Hyperparameter of the model
            }
    """

    if not wandb_kwargs["id"]:
        run_id = wandb.util.generate_id()
        print("run_id", run_id)

    # Define dataset
    ds_train = CustomDataset(
        path_to_images=metadata_kwargs["path_to_images"],
        metadata=metadata_kwargs["metadata_train"],
        col_target=wandb_kwargs["config"]["col_target"],
        return_original=False,
        normalizing_factor=wandb_kwargs["config"]["normalizing_factor"],
        augment=wandb_kwargs["config"]["augment"],
    )
    ds_test = CustomDataset(
        path_to_images=metadata_kwargs["path_to_images"],
        metadata=metadata_kwargs["metadata_test"],
        col_target=wandb_kwargs["config"]["col_target"],
        return_original=False,
        normalizing_factor=wandb_kwargs["config"]["normalizing_factor"],
        augment=None,
    )

    ds_train_sample = CustomDataset(
        path_to_images=metadata_kwargs["path_to_images"],
        metadata=metadata_kwargs["metadata_train"],
        col_target=wandb_kwargs["config"]["normalizing_factor"],
        return_original=False,
        normalizing_factor=wandb_kwargs["config"]["normalizing_factor"],
        augment=wandb_kwargs["config"]["augment"],
    )

    # visualize_augmented_images(ds_train_sample, brightness=0.1, col_target=wandb_kwargs["config"]["col_target"], n=7)

    # define dataloader
    train_dataloader = torch.utils.data.DataLoader(
        ds_train,
        batch_size=wandb_kwargs["config"]["train_batch_size"],
        shuffle=True,
        num_workers=2,
        drop_last=True,
    )

    test_dataloader = torch.utils.data.DataLoader(
        ds_test,
        batch_size=wandb_kwargs["config"]["test_batch_size"],
        shuffle=True,
        num_workers=2,
        drop_last=True,
    )

    # Instance model
    torch.manual_seed(42)

    model = REGRESSORS[wandb_kwargs["config"]["regressor"]](
        output_size=wandb_kwargs["config"]["output_size"],
        backbone=wandb_kwargs["config"]["backbone"],
    )

    model.to(metadata_kwargs["device"])

    if wandb_kwargs["config"]["pretrained"]:
        model.backbone.load_state_dict(torch.load(metadata_kwargs["path_to_load_backbone"]).backbone.state_dict())

        for parameters in model.backbone.parameters():
            parameters.requires_grad = False

        model.to(metadata_kwargs["device"])

        print("\n --------------------------------------")
        print("The backbone was transfered successfully")
        print("----------------------------------------\n")

    # loss function
    loss = torch.nn.MSELoss()

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        weight_decay=wandb_kwargs["config"]["weight_decay"],
        lr=wandb_kwargs["config"]["learning_rate"],
    )

    # Learning schedule
    schedule = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    train_model(
        train_dataloader,
        test_dataloader,
        model,
        loss,
        optimizer,
        schedule,
        wandb=wandb,
        wandb_kwargs=wandb_kwargs,
        metadata_kwargs=metadata_kwargs,
    )
