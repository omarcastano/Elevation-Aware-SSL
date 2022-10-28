# import libraries
import torch
import os
import wandb

import numpy as np
import pandas as pd
import plotly.express as px
from MasterThesis import EDA
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.nn import Module
from tqdm import tqdm
from typing import List, Tuple
from sklearn.model_selection import KFold

# from . import metrics
from sklearn.metrics import confusion_matrix
from MasterThesis.utils.classification import metrics
from sklearn.model_selection import StratifiedKFold, train_test_split
from IPython.display import clear_output
from MasterThesis.models.classification.randominit import (
    LinearClassifier,
    NoneLinearClassifier,
)


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
    class_name: list,
    device: torch.device,
):

    """
    Arguments:
        train_dataloader: pytorch dataloader with training images and labels
        model: pytorch model for semantic segmentation
        loss_fn: loss function for semantic segmentation
        opimizer: pytorch optimizer
        class_name: list with the name of each class
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
    logs = {"Train loss": running_loss}

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
        class_name: list with the name of each class
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

    logs = {"Test loss": running_loss}

    return logs
