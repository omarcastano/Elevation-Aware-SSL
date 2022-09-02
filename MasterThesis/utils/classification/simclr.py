# import libraries
import numpy as np
import pandas as pd
import cv2
import albumentations as album
from MasterThesis import EDA
import torch
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import confusion_matrix


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

    img = np.float32(img)

    input_size = img.shape

    transform = album.Compose(
        [
            album.RandomSizedCrop(
                min_max_height=(70, 75),
                height=input_size[0],
                width=input_size[1],
                p=0.5,
            ),
            album.VerticalFlip(p=0.5),
            album.HorizontalFlip(p=0.5),
            album.RandomRotate90(),
            # album.HueSaturationValue(
            #    hue_shift_limit=0.5,
            #    sat_shift_limit=(-0.05, 0.2),
            #    val_shift_limit=0.08,
            #    p=0.8,
            # ),
            album.ToGray(always_apply=False, p=0.1),
            album.GaussianBlur(blur_limit=(1, 5), p=0.5),
        ]
    )

    transformed_img = transform(image=img)
    img1 = transformed_img["image"]

    transformed_img = transform(image=img)
    img2 = transformed_img["image"]

    return img1, img2


class CustomDaset(torch.utils.data.Dataset):

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

    def __init__(self, path_to_images, metadata, return_original):

        super().__init__()
        self.metadata = metadata
        self.path_to_images = path_to_images
        self.return_original = return_original

    def __len__(self):

        return len(self.metadata)

    def __getitem__(self, index):

        # Load and transform input image
        image = EDA.read_numpy_image(
            self.path_to_images + self.metadata.Image.tolist()[index]
        )

        if len(image.shape) == 4:
            image = EDA.less_cloudy_image(image)

        image = np.clip(image[:3], 0, 6000) / 6000
        original_image = image.copy()
        image = image.transpose(1, 2, 0).astype(np.float32)

        # Data Augmentation
        image_1, image_2 = data_augmentation(image)

        # Set data types compatible with pytorch
        image = image.astype(np.float32).transpose(2, 0, 1)
        image_1 = image_1.astype(np.float32).transpose(2, 0, 1)
        image_2 = image_2.astype(np.float32).transpose(2, 0, 1)

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

        image = image.transpose(1, 2, 0)
        image_1 = image_1.transpose(1, 2, 0)
        image_2 = image_2.transpose(1, 2, 0)

        ax[0, i].imshow(image + 0.1)
        ax[1, i].imshow(image_1 + 0.1)
        ax[2, i].imshow(image_2 + 0.1)

        ax[0, i].set_title("Originale")
        ax[1, i].set_title("Augmented 1")
        ax[2, i].set_title("Augmented 2")

        ax[0, i].axis("off")
        ax[1, i].axis("off")
        ax[2, i].axis("off")
