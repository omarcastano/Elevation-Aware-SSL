# python moduls
from typing import List, Tuple

# Standard libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split


# torch moduls
import torch


def visualize_augmented_images(
    dataset: torch.utils.data.Dataset, n: int = 10, col_target: List = None, brightness: float = 0.0, figsize: Tuple = (32, 10)
) -> None:

    """
    Plots augmented images used to train a segmentation model

    Arguments:
    ----------
        dataset: pytorch dataset which must return the augmented
                 views of the same image and the image itself
        n: int, default=10
            number of images to plot
        brightness: float, default=0.0
            factor to adjust the brightness of the images
    """

    fig, ax = plt.subplots(2, n, figsize=figsize)

    for i in range(n):
        original, augmented, label = dataset[i]

        augmented = np.array(augmented).transpose(1, 2, 0)
        original = np.array(original).transpose(1, 2, 0)
        label = np.array(label)

        ax[0, i].imshow(original + brightness)
        ax[1, i].imshow(augmented + brightness)

        s = ""
        for i_, j_ in zip(col_target, label.astype("str")):
            s += i_ + ":" + str(j_) + " \n"

        ax[0, i].set_title(f"Original")
        ax[1, i].set_title(f"Augmented \n {s}")

        ax[0, i].axis("off")
        ax[1, i].axis("off")

    plt.show()


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

    metadata_train = []
    metadata_test = []
    metadata_valid = []

    kf = KFold(n_splits=n_split)

    X = np.arange(10)
    for train, test in kf.split(metadata):

        train_dataset, train = train_test_split(
            metadata.iloc[train], train_size=train_size, stratify=metadata.iloc[train].Labels.tolist(), random_state=42
        )
        metadata_train.append(train_dataset)
        metadata_test.append(metadata.iloc[test].copy().reset_index(drop=True))
        metadata_valid.append(train.sample(len(test)).copy().reset_index(drop=True))

    print(f"Number fo total images : {n_total_images}")
    print(f"Number of images to train: {metadata_train[1].shape[0]}, {round(metadata_train[1].shape[0]/n_total_images, 5)*100}%")
    print(f"Number of images to test: {metadata_test[1].shape[0]}, {round(metadata_test[1].shape[0]/n_total_images, 5)*100}%")
    print(f"Number of images to valid: {metadata_valid[1].shape[0]}, {round(metadata_valid[1].shape[0]/n_total_images, 5)*100}%")

    return metadata_train, metadata_test, metadata_valid
