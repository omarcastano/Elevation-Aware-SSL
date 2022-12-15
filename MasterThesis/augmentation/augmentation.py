"""
This Module provides functions to apply data augmentation 
"""

import torch
from torchvision import transforms
import albumentations as album
import numpy as np

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

# Data Augmentation
def data_augmentation(img, augment: dict = None):
    """
    Data augmentation such as vertical and horizontal flip,
    random rotation and random sized crop.

    Arguments:
    ----------
        img: 3D numpy array
            input image with shape (C,H,W)
        transforms: dictionary
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

    img = torch.from_numpy(img.astype(np.float32))
    img = augmentation(img)

    return img


AUGMENTATIONS_V2 = {
    "horizontal_flip_prob": 0.5,
    "vertical_flip_prob": 0.5,
    "resize_scale": (0.85, 1.0),
    "resize_prob": 0.5,
    "brightness": 0.2,
    "contrast": 0.2,
    "saturation": 0.2,
    "hue": 0.1,
    "color_jitter_prob": 0.5,
    "gray_scale_prob": 0.2,
}


def data_augmentation_v2(img, label=None, augment: dict = None):

    augment = augment or AUGMENTATIONS_V2

    # Defines transformation
    augmentation = album.Compose(
        [
            album.VerticalFlip(p=augment["vertical_flip_prob"]),
            album.HorizontalFlip(p=augment["horizontal_flip_prob"]),
            album.RandomResizedCrop(height=img.shape[1], width=img.shape[2], scale=augment["resize_scale"], p=augment["resize_prob"]),
            album.ColorJitter(
                brightness=augment["brightness"],
                contrast=augment["contrast"],
                saturation=augment["saturation"],
                hue=augment["hue"],
                p=augment["color_jitter_prob"],
            ),
            album.ToGray(p=augment["gray_scale_prob"]),
        ]
    )

    img = img.transpose(1, 2, 0).astype(np.float32)

    if label is not None:
        augmented = augmentation(image=img, mask=label)
        img, label = augmented["image"].transpose(2, 0, 1), augmented["mask"]
        img = torch.from_numpy(img.astype(np.float32))

        return img, label

    else:
        augmented = augmentation(image=img)
        img = augmented["image"].transpose(2, 0, 1)
        img = torch.from_numpy(img.astype(np.float32))

        return img


# Data loader
def data_augmentation_segmentation(img, label, augment: dict = None):
    """
    Data augmentation vertical and horizontal flip, random rotation and random sized crop.

    Arguments:
        img: 3D numpy array
            input image with shape (C,H,W)
        label: 1D numpy array
            labels with shape (H,W)
        transforms: dictionary
            kwargs to transform images
    """

    augment = augment or AUGMENTATIONS_V2

    # Defines spatial transformation
    spatial_augment = album.Compose(
        [
            album.VerticalFlip(p=augment["vertical_flip_prob"]),
            album.HorizontalFlip(p=augment["horizontal_flip_prob"]),
            album.RandomResizedCrop(height=img.shape[1], width=img.shape[2], scale=augment["resize_scale"], p=augment["resize_prob"]),
        ]
    )

    # Defines Color Jitter
    color_jitter = transforms.ColorJitter(
        brightness=augment["brightness"],
        contrast=augment["contrast"],
        saturation=augment["saturation"],
        hue=augment["hue"],
    )

    # Defines color augmentation
    color_augment = transforms.Compose(
        [
            transforms.RandomApply([color_jitter], p=augment["color_jitter_prob"]),
            transforms.RandomGrayscale(p=augment["gray_scale_prob"]),
        ]
    )

    img = img.transpose(1, 2, 0)
    augmented = spatial_augment(image=img, mask=label)
    img, label = augmented["image"].transpose(2, 0, 1), augmented["mask"]

    img = torch.from_numpy(img.astype(np.float32))
    img = color_augment(img)

    return img, label
