# import libraries
import cv2
import os
import torch
import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import albumentations as album
from MasterThesis import EDA
from tqdm import tqdm
from torch.nn import Module
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
from MasterThesis.models.classification.simclr import SimCLR

# from MasterThesis.utils.classification.losses import NTXentLoss
from lightly.loss import NTXentLoss
from torchvision import transforms


# Data loader
def data_augmentation(img):

    """
    Data augmentation for such as vertical and horizontal flip,
    random rotation and random sized crop.

    Argumetns:
        image: 3D numpy array
            input image with shape (H,W,C)
        label: 1D numpy array
            labels with shape (H,W)
        input_size: list [H,W].
            python list with the width and heigth
            of the output images and labels.
            example: input_size=[100,100]
    """

    augmentation = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomResizedCrop(size=img.shape[1], scale=(0.9, 1.0)),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.5),
            transforms.RandomGrayscale(p=0.5),
            # transforms.GaussianBlur(kernel_size=5),
        ]
    )

    img1 = augmentation(img)

    return img1


class CustomDataset(torch.utils.data.Dataset):

    """
    This class creates a custom dataset for implementing
    SimCLR self-supervised learning methodology

    Arguments:
        path_to_images: str
            path to the folder where images are stored
        path_to_labels: str
            path to the folder where labels are stored
        metadata: data frame
            dataframe with the names of images and labels
        return_original: If True also return the original image,
            so the output will be (original, augmented1, augmented2), else
            the output will ve (augmented1, augmented2)
    """

    def __init__(
        self,
        path_to_images,
        metadata,
        return_original=False,
        normalizing_factor: int = 6000,
    ):

        super().__init__()
        self.metadata = metadata
        self.path_to_images = path_to_images
        self.return_original = return_original
        self.normalizing_factor = normalizing_factor

    def __len__(self):

        return len(self.metadata)

    def __getitem__(self, index):

        # Load and transform input image
        image = EDA.read_numpy_image(self.path_to_images + self.metadata.Image.tolist()[index])

        if len(image.shape) == 4:
            image = EDA.less_cloudy_image(image)

        image = np.clip(image[:3], 0, self.normalizing_factor) / self.normalizing_factor

        original_image = image.copy()

        image = torch.from_numpy(image.astype(np.float32))

        # Data Augmentation
        image_1 = data_augmentation(image)
        image_2 = data_augmentation(image)

        if self.return_original:
            return original_image, image_1, image_2
        else:
            return image_1, image_2


def visualize_augmented_images(dataset: torch.utils.data.Dataset, n: int = 10) -> None:

    """
    Plots augmented images used to pre-train a model using SimCLR methodology

    Argumetns:
        dataset: pytorch dataset which must return the augmneted
                 views of the same image and the image itslef
        n: number of images to plot
    """

    fig, ax = plt.subplots(3, n, figsize=(32, 10))

    for i in range(n):
        image, image_1, image_2 = dataset[i]

        image = np.array(image).transpose(1, 2, 0)
        image_1 = np.array(image_1).transpose(1, 2, 0)
        image_2 = np.array(image_2).transpose(1, 2, 0)

        ax[0, i].imshow(image)
        ax[1, i].imshow(image_1)
        ax[2, i].imshow(image_2)

        ax[0, i].set_title("Originale")
        ax[1, i].set_title("Augmented 1")
        ax[2, i].set_title("Augmented 2")

        ax[0, i].axis("off")
        ax[1, i].axis("off")
        ax[2, i].axis("off")


# Train one epoch
def train_one_epoch(
    train_loader: DataLoader,
    model: Module,
    criterion: Module,
    optimizer: optim,
    scheduler: optim.lr_scheduler,
    device: torch.device,
):

    """
    Evaluate model performance using a test set with different metrics

    Arguments:
        train_dataloader: pytorch dataloader with training images and labels
        model: pytorch model for semantic segmentation
        criterion: loss function to train the self-supervised model
        opimizer: pytorch optimizer
        schduler: pytorch learning scheduler
        device: device to tran the network sucha as cpu or cuda
    """

    running_loss = 0

    model.train()
    for epoch, (input1, input2) in enumerate(tqdm(train_loader), 1):

        # Set zero gradients for every batch
        optimizer.zero_grad()

        input1, input2 = input1.to(device), input2.to(device)

        # Get the keys and Queries
        q = model(input1)
        k = model(input2)

        # Compute the total loss
        loss = criterion(q, k)

        # compute gradients
        loss.backward()

        # Update weeigths
        optimizer.step()

        running_loss += loss

    scheduler.step()
    running_loss = running_loss / epoch

    return running_loss.item()


def test_one_epoch(
    test_loader: DataLoader,
    model: Module,
    criterion: Module,
    device: torch.device,
):

    """
    Evaluate model performance using a test set with different metrics

    Arguments:
        test_dataloader: pytorch dataloader with training images and labels
        model: pytorch model for semantic segmentation
        criterion: loss function to train the self-supervised model
        device: device to tran the network sucha as cpu or cuda
        Lambda: GLCNet loss function hyperparameter
    """

    running_loss = 0

    # model.eval()

    with torch.no_grad():
        for epoch, (input1, input2) in enumerate(tqdm(test_loader), 1):

            input1, input2 = input1.to(device), input2.to(device)

            # Get the keys and Queries
            q = model(input1)
            k = model(input2)

            # Compute the total loss
            loss = criterion(q, k)

            running_loss += loss

    running_loss = running_loss / epoch

    return running_loss.item()


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
                "path_to_images": path_to_images,
                "path_to_save_model": None,  # Path to save the model that is being trained (do not include the extension .pt or .pth)
                "path_to_load_model": None,  # Path to load a model from a checkpoint (useful to handle notebook disconection)
                "metadata": metadata,
                "metadata_train": metadata_train[0],
                "metadata_test": metadata_test[0],
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
                "amount_of_ft_data": train_size,
                "bn_momentum": 0.9,
                "input_shape": [100, 100],
                "patch_size" : 40,
                "patch_num" : 16,
                "in_channels" : 3,
                "temperature" : 0.5, # Temperature hyperparameter used in the NTXenLoss function
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

        epoch = 1

        if metadata_kwargs["path_to_load_model"]:
            checkpoint = torch.load(checkpoint_load_path)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            epoch = checkpoint["epoch"]
            train_loss = checkpoint["train_loss"]
            test_loss = checkpoint["test_loss"]

        # Start Training the model
        while epoch <= wandb_kwargs["config"]["epochs"]:
            print("-------------------------------------------------------------")
            print(f"Epoch {epoch}/{wandb_kwargs['config']['epochs']}")

            train_loss = train_one_epoch(
                train_loader,
                model,
                loss_fn,
                optimizer,
                schedule,
                metadata_kwargs["device"],
            )

            test_loss = test_one_epoch(
                test_loader,
                model,
                loss_fn,
                metadata_kwargs["device"],
            )

            print(f"\n    Train Loss: {train_loss}")
            print(f"\n    Test Loss: {test_loss}")

            wandb.log({"Train Loss": train_loss, "Test Loss": test_loss})
            print("------------------------------------------------------------- \n")

            # Save the model
            if metadata_kwargs["path_to_save_model"]:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_loss": train_loss,
                        "test_loss": test_loss,
                    },
                    checkpoint_path,
                )

                torch.save(model, model_path)

            epoch += 1


##Run a experiment
def run_train(
    wandb: wandb,
    wandb_kwargs: dict = None,
    metadata_kwargs: dict = None,
):

    """
    Pre-train a backbone using SimCLR methodology

    Arguments:
    ---------
        metadata_kwargs: metadata required to train the model. Example:
            metadata_kwargs = {
                "path_to_images": path_to_images,
                "path_to_save_model": None,  # Path to save the model that is being trained (do not include the extension .pt or .pth)
                "path_to_load_model": None,  # Path to load a model from a checkpoint (useful to handle notebook disconection)
                "metadata": metadata,
                "metadata_train": metadata_train[0],
                "metadata_test": metadata_test[0],
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
                "amount_of_ft_data": train_size,
                "bn_momentum": 0.9,
                "input_shape": [100, 100],
                "patch_size" : 40,
                "patch_num" : 16,
                "in_channels" : 3,
                "temperature" : 0.5, # Temperature hyperparameter used in the NTXenLoss function
                "weight_decay": 0.0005,
                "learning_rate": 1e-3,
                "batch_size": 16,
                "epochs": 2,
            }
    """

    if not wandb_kwargs["id"]:
        run_id = wandb.util.generate_id()
        print("--------------------")
        print("run_id", run_id)
        print("--------------------")

    # Define dataset
    ds_train = CustomDataset(
        metadata_kwargs["path_to_images"],
        metadata_kwargs["metadata_train"],
        normalizing_factor=wandb_kwargs["config"]["normalizing_factor"],
    )

    ds_test = CustomDataset(
        metadata_kwargs["path_to_images"],
        metadata_kwargs["metadata_test"],
        normalizing_factor=wandb_kwargs["config"]["normalizing_factor"],
    )

    ds_train_sample = CustomDataset(
        metadata_kwargs["path_to_images"],
        metadata_kwargs["metadata_train"],
        return_original=True,
        normalizing_factor=wandb_kwargs["config"]["normalizing_factor"],
    )

    visualize_augmented_images(ds_train_sample)

    # define dataloader
    train_dataloader = torch.utils.data.DataLoader(
        ds_train,
        batch_size=wandb_kwargs["config"]["batch_size"],
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )
    test_dataloader = torch.utils.data.DataLoader(
        ds_test,
        batch_size=wandb_kwargs["config"]["batch_size"],
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )

    # Instance Deep Lab model
    model = SimCLR(
        proj_hidden_dim=512,
        proj_output_dim=128,
        backbone=wandb_kwargs["config"]["backbone"],
        cifar=wandb_kwargs["config"]["cifar"],
    )

    model.to(metadata_kwargs["device"])

    # loss  functiction
    loss = NTXentLoss(
        temperature=wandb_kwargs["config"]["temperature"],
    )

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=wandb_kwargs["config"]["learning_rate"],
        weight_decay=wandb_kwargs["config"]["weight_decay"],
    )

    # learning scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=wandb_kwargs["config"]["epochs"], eta_min=4e-08)

    train_model(
        train_dataloader,
        test_dataloader,
        model,
        loss,
        optimizer,
        scheduler,
        wandb=wandb,
        wandb_kwargs=wandb_kwargs,
        metadata_kwargs=metadata_kwargs,
    )
