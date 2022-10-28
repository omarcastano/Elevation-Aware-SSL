# import libraries
import torch
import numpy as np
import pandas as pd
import plotly.express as px
from MasterThesis import EDA
from typing import List, Union, Tuple
from MasterThesis.utils.classification import metrics
from torchvision import transforms


AUGMENTATIONS = {
    "horizontal_flip_prob": 0.5,
    "vertical_flip_prob": 0.5,
    "resize_scale": (0.7, 1.0),
    "brightness": 0.2,
    "contrast": 0.2,
    "saturation": 0.2,
    "hue": 0.2,
    "color_jitter_prob": 0.8,
    "gray_scale_prob": 0.2,
}

# Data loader
def data_augmentation(img, augment: dict = None):

    """
    Data augmentation for such as vertical and horizontal flip,
    random rotation and random sized crop.

    Arguments:
        img: 3D numpy array
            input image with shape (C,H,W)
        augment: dictionary
            kwargs to transform images
    """

    augment = augment or AUGMENTATIONS

    # Defines Color Jitter
    color_jitter = transforms.ColorJitter(
        brightness=augment["brightness"],
        contrast=augment["contrast"],
        saturation=augment["saturation"],
        hue=augment["hue"],
    )

    # Defines data augmentation
    augmentation = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=augment["vertical_flip_prob"]),
            transforms.RandomVerticalFlip(p=augment["horizontal_flip_prob"]),
            transforms.RandomResizedCrop(size=img.shape[1], scale=augment["resize_scale"]),
            transforms.RandomApply([color_jitter], p=augment["color_jitter_prob"]),
            transforms.RandomGrayscale(p=augment["gray_scale_prob"]),
        ]
    )

    img1 = augmentation(img)

    return img1


class CustomDataset(torch.utils.data.Dataset):
    """
    This class creates a custom pytorch dataset

    Arguments:
    ----------
        path_to_images: str
            path to the folder where images are stored
        path_to_labels: str
            path to the folder where labels are stored
        metadata: pandas dataframe
            dataframe with the names of images and labels
        return_original: bool, default=False
            If True also return the original image,
            so the output will be (original, augmented, label), else
            the output will ve (augmented label)
        normalizing_factor: int, default=6000
            value to clip and normalizer images
        augment: dictionary
            kwargs to transform images
    """

    def __init__(
        self,
        path_to_images: str,
        metadata: pd.DataFrame,
        col_target: List,
        return_original: bool = False,
        normalizing_factor: int = 6000,
        augment: bool = False,
    ):

        super().__init__()
        self.metadata = metadata
        self.path_to_images = path_to_images
        self.return_original = return_original
        self.normalizing_factor = normalizing_factor
        self.augmentation = augment
        self.col_target = col_target

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

        # Load label
        label = self.metadata[self.col_target].iloc[index].to_numpy().astype(np.float32)

        # Data Augmentation
        if self.augmentation:
            image = data_augmentation(image, self.augmentation)

        if self.return_original:
            return original_image, image, label

        return image, label
