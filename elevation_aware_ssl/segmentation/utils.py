import torch
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from typing import List
from elevation_aware_ssl.segmentation.dataset import CustomDataset


def visualize_augmented_images(dataset: torch.utils.data.Dataset, n: int = 10, classes_name: List[str] = None) -> None:
    """
    Plots augmented images used to train a segmentation model

    Arguments:
    ----------
        dataset: pytorch dataset which must return the augmented
                 views of the same image and the image itself
        n: number of images to plot
        classes_name: name of each class
    """

    cmap = {i: [np.random.random(), np.random.random(), np.random.random(), 1] for i in range(len(classes_name))}
    labels_map = {i: name for i, name in enumerate(classes_name)}
    patches = [mpatches.Patch(color=cmap[i], label=labels_map[i]) for i in cmap]

    fig, ax = plt.subplots(3, n, figsize=(32, 10))

    for i in range(n):
        original, augmented, label = dataset[i]

        augmented = np.array(augmented).transpose(1, 2, 0)
        original = np.array(original).transpose(1, 2, 0)
        label = np.array(label)

        ax[2, i].imshow(label)
        ax[1, i].imshow(augmented + 0.1)
        ax[0, i].imshow(original + 0.1)

        ax[0, i].set_title("Originale")
        ax[1, i].set_title("Augmented")
        ax[2, i].set_title("Label")

        ax[0, i].axis("off")
        ax[1, i].axis("off")
        ax[2, i].axis("off")

        arrayShow = np.array([[cmap[i] for i in j] for j in label])
        ax[2, i].imshow(arrayShow)

    plt.legend(
        handles=patches,
        bbox_to_anchor=(1.05, 1),
        loc=2,
        borderaxespad=0.0,
        markerscale=30,
        fontsize="large",
    )

    plt.show()


def plot_predictions(
    path_to_images: str,
    path_to_labels: str,
    metadata: pd.DataFrame,
    model1: torch.nn.Module,
    model2: torch.nn.Module,
    class_names: List[str],
    n: int = 5,
    figsize=(15, 8),
):
    """
    Plots segmentation mask predicted by a model

    Arguments:
    ----------
    path_to_images : str
        path where images are stored
    path_to_labels : str
        path where labels are stored
    metadata : pd.DataFrame
        dataframe with the names of the images and labels
    model1 : torch.nn.Module
        model to make predictions
    model2 : torch.nn.Module
        model to make predictions
    class_names : List[str]
        name of each class
    n : int, optional, default=5
        number of images to plot
    figszie: Tuple
        size of the figure
    """

    t = 1  ## alpha value
    cmap = {i: [np.random.random(), np.random.random(), np.random.random(), 1] for i in range(len(class_names))}
    labels_map = {i: name for i, name in enumerate(class_names)}

    patches = [mpatches.Patch(color=cmap[i], label=labels_map[i]) for i in cmap]

    _, ax = plt.subplots(4, n, figsize=figsize)

    ds = CustomDataset(path_to_images, path_to_labels, metadata.head(n))

    for i, (image, label) in enumerate(ds):

        label = np.array([[cmap[i] for i in j] for j in label.numpy()])

        pred1 = model1(image.reshape(1, 3, 100, 100)).argmax(dim=1).squeeze()
        pred2 = model2(image.reshape(1, 3, 100, 100)).argmax(dim=1).squeeze()

        pred1 = np.array([[cmap[i] for i in j] for j in pred1.numpy()])
        pred2 = np.array([[cmap[i] for i in j] for j in pred2.numpy()])

        ax[0, i].imshow(np.array(image).transpose(1, 2, 0) + 0.1)
        ax[1, i].imshow(label)
        ax[2, i].imshow(pred1)
        ax[3, i].imshow(pred2)

        ax[0, i].axis("off")
        ax[1, i].axis("off")
        ax[2, i].axis("off")
        ax[3, i].axis("off")

        ax[0, i].set_title("Image")
        ax[1, i].set_title("Label")
        ax[2, i].set_title("RandomInit")
        ax[3, i].set_title("SimCLR")

        plt.legend(
            handles=patches,
            bbox_to_anchor=(1.05, 1),
            loc=2,
            borderaxespad=0.0,
            markerscale=30,
            fontsize="large",
        )
