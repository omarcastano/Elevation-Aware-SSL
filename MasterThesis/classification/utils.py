"""
This module provides utilities
"""

import torch
import numpy as np
import pandas as pd
from typing import List, Tuple
import matplotlib.pylab as plt
from sklearn.model_selection import StratifiedKFold, train_test_split


def visualize_augmented_images(
    dataset: torch.utils.data.Dataset, class_names: List[str], n: int = 10, brightness: float = 0.0, **kwargs
) -> None:
    """
    Plots augmented images used to train a segmentation model

    Arguments:
    ----------
        dataset: pytorch dataset
            data with the augmented images, original images and labels
        classes_name: str, default
            name of each class in the dataset
        n: int, default=10
            number of images to plot
        brightness: float, default=0.0
    """

    fig, ax = plt.subplots(2, n, figsize=(32, 7))

    for i in range(n):
        original, augmented, label = dataset[i]

        augmented = np.array(augmented).transpose(1, 2, 0)
        original = np.array(original).transpose(1, 2, 0)

        ax[1, i].imshow(augmented + brightness)
        ax[0, i].imshow(original + brightness)

        ax[0, i].set_title(f"Originale (Label: {label}) \n {class_names[label]}")
        ax[1, i].set_title(f"Augmented (Label: {label}) \n {[label]}")

        ax[0, i].axis("off")
        ax[1, i].axis("off")

    plt.show()


def generate_metadata_train_test_stratified_cv(
    metadata: pd.DataFrame, train_size: float, n_split: int = 5
) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    """
    Provides train/test indices to split data in train/test sets. Split
    dataset into k consecutive folds to train and test dataset via k-fold sets

    Arguments:
    ----------
        metadata: dataframe with the path to images and labels
        train_size: percentage of in each train set
        n_split: numbers of folds. Must be at least 2
    """

    n_total_images = metadata.shape[0]

    metadata_train, metadata_test = [], []

    kf = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=42)

    for train, test in kf.split(metadata, metadata["Labels"].tolist()):
        train_dataset, _ = train_test_split(
            metadata.iloc[train], train_size=train_size, stratify=metadata.iloc[train].Labels.tolist(), random_state=42
        )
        metadata_train.append(train_dataset)
        metadata_test.append(metadata.iloc[test].copy().reset_index(drop=True))

    print(f"Number fo total images : {n_total_images}")
    print(f"Number of images to train: {metadata_train[1].shape[0]}, {round(metadata_train[1].shape[0]/n_total_images, 5)*100}%")
    print(f"Number of images to test: {metadata_test[1].shape[0]}, {round(metadata_test[1].shape[0]/n_total_images, 5)*100}%")

    return metadata_train, metadata_test
