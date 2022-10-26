import geopandas as gpd
import matplotlib.pyplot as plt
import pyproj
import plotly.express as px
import pandas as pd
import h5py
import os
from shapely.geometry import Point, Polygon, MultiPolygon
import matplotlib.patches as mpatches
import numpy as np
import MasterThesis.preprocessing as DP
from shapely.geometry import Polygon
import seaborn as sns
from osgeo import gdal
import earthpy as et
import earthpy.spatial as es
import earthpy.plot as ep
from tqdm import tqdm
import multiprocessing as mp
from typing import List, Tuple

# Function that reads geotiff
def read_geotiff_image(path):
    """
    Read geotiff

    Arguments:
        path: string
            path to image
    """

    image = gdal.Open(path).ReadAsArray()
    return image


# Function that reads numpy image
def read_numpy_image(path):
    """
    Read image stored in .npz format

    Arguments:
        path: string
            path to image
    """

    image = np.load(path)
    return image["arr_0"]


def load_image_and_labels(img_path, label_path):

    """
    Load Image and label given the path

    Arguments:
        img_path: path to image
        label_path:  path to label
    """

    img = read_numpy_image(img_path)
    lbl = read_geotiff_image(label_path)


def image_label_sanity_check(metadata, path_to_images, path_to_labels):

    """
    load all images and labels to verify they exist on the folder

    Arguments:
       metadata: dataframe with the path to label and images
       path_to_images: path to the folder where all images are stored
       path_to_labels: path to the folder where all labels are stored

    """

    with mp.Pool(mp.cpu_count()) as p:
        p.starmap(
            load_image_and_labels,
            zip(path_to_images + metadata.Image, path_to_labels + metadata.Mask),
        )


# Function that plots all bands of a image
def visualize_image_bands(path_to_images: str, metadata: pd.DataFrame):

    """
    Visualize all bands of a given image. The image is expected
    to be in .npz format.

    Arguments:
        path_to_images: string
            path to image
        metadata: DataFrame
            dataframe with the name of each image
    """

    image = read_numpy_image(path_to_images + metadata.loc[2].Image)

    if len(image.shape) == 4:
        print("#########################################")
        print("Image Shape (T,C,W,H):", image.shape)
        print("########################################")
        ep.plot_bands(image[0], cols=6, scale=True, cbar=False, figsize=(25, 10))
    else:
        print("#########################################")
        print("Image Shape (C,W,H):", image.shape)
        print("########################################")
        ep.plot_bands(image, cols=6, scale=True, cbar=False, figsize=(25, 10))


# This function plot pixel histogram for each band
def pixel_histogram(path_to_images, metadata, sample=20, clip=None):

    """
    This function plots histogram and basic statistics for each band.
    This function assums that images are stored in .npz format

    Arguments:
        path_to_images: string
            path to the root folder where images are stroed
        metadata: data frame
            dataframe with the name of each image
        sample: int
            number of images to load and plot
        clip: int, default=None
            value to clip pixel images
    """

    images = np.array([read_numpy_image(img) for img in path_to_images + metadata.head(sample).Image])

    if clip:
        images = np.clip(images, 0, clip)

    print("------Statistics for all Bands------")
    print(f"Pixel Mean={images.mean().round(3)}")
    print(f"Pixel Min={images.min()}")
    print(f"Pixel Max={images.max()}")
    print("------------------------------------")

    i = 1
    fig, axs = plt.subplots(3, 4, figsize=(30, 10))
    for img, ax in zip(images.transpose(2, 0, 1, 3, 4), axs.flatten()):
        sns.histplot(x=img.ravel(), ax=ax)
        ax.set_xlabel("Pixel Value")
        ax.legend(
            [f"Band {i} \n Mean={img.mean().round(3)} \n Min={img.min()}, \n Max={img.max()}"],
            fontsize=15,
        )
        i += 1

    sns.histplot(x=images.ravel(), ax=axs[2, 3])
    axs[2, 3].set_xlabel("Pixel Value")
    axs[2, 3].legend(["All Bands"], fontsize=20)


# Select the less cloudy image from the time series
def less_cloudy_image(images):

    """
    This function select the less cloudy image from the time series

    Arguments :
        images (T,C,W,H):numpy array
            Time series images
    """

    if len(images.shape) == 4:
        means = images[:, 0:3, :, :].mean(axis=(1, 2, 3))
        idx = means[means > 0].argmin()
        return images[idx]
    else:
        return images


# Plot less cloudy image from the time seires
def pixel_histogram_with_image(path_to_images, metadata, sample=5, scale_factor=13000, cloud_mask=False, clip=None):

    """
    This function plots the histogram and the image for the RGB bands. You
    can choose if select the less cloudy image from the time series or
    plot a random image from the it. Images are expected to have the
    shape (T,C,W,H), and RBG bands must be the firs three bands.

    Arguments:
        path_to_images: string
            path to the root folder where images are stroed
        metadata: data frame
            dataframe with the name of each image
        sample: int
            number of images to load and plot
        scale_factor: flaot
            Factor to sacles images
        clip: int, default=None
            value to clip pixel images
        cloud_mask: bool, default=True
            wheter or not to select the less cloudy image
        clip: int
            upper bount value to clip images
    """

    if cloud_mask:
        masked_images = []
        images = np.array([read_numpy_image(img) for img in path_to_images + metadata.head(sample).Image])
        for img in images:
            masked_images.append(less_cloudy_image(img))
        images = np.array(masked_images)

    else:
        idx = 2  # np.random.randint(0,24)
        images = np.array([read_numpy_image(img)[idx] for img in path_to_images + metadata.head(sample).Image])

    if clip:
        images = np.clip(images, 0, clip)

    print("------Statistics for the RGB bands------")
    print(f"Pixel Mean={images[:,0:3,:,:].mean().round(3)}")
    print(f"Pixel Min={images[:,0:3,:,:].min()}")
    print(f"Pixel Max={images[:,0:3,:,:].max()}")
    print("------------------------------------")

    i = 1
    fig, ax = plt.subplots(2, sample, figsize=(30, 10))

    if scale_factor:
        images = images[:, 0:3, :, :] / scale_factor
    else:
        images = images[:, 0:3, :, :]

    for i, img in enumerate(images):

        ax[0, i].imshow(img.transpose(1, 2, 0))
        sns.histplot(x=img.ravel(), ax=ax[1, i])
        ax[1, i].set_xlabel("Pixel Value")
        ax[1, i].legend(
            [f"Mean={img.mean().round(3)} \n Min={img.min().round(3)}, \n Max={img.max().round(3)}"],
            fontsize=10,
        )


def simple_cloud_mask_filter(image, scale_factor, clip, threshold=0.5):

    """
    This function filter images based on a given threshold

    Argumetns:
        image (T,C,W,H):numpy array
            Time series images
        scale_factor: flaot
            Factor to sacles images
        clip: int
            upper bount value to clip images
        threshold: float
            value to filter out cloudy images.
    """

    if clip:
        image = np.clip(image, 0, clip)

    if np.mean(image[:3] / scale_factor) < threshold:
        return ("Not Remove", image)
    else:
        return ("Remove", image)


def pixel_histogram_and_filtered_image(path_to_images, metadata, sample=5, scale_factor=13000, clip=None, threshold=0.5):

    """
    This function plots the histogram and the image for the RGB bands.
    It also shows a label suggesting if an image from the time series
    should be remove due to its high cloudiness. Images are expected
    to have the shape (T,C,W,H), and RBG bands must be the first
    three bands.

    Arguments:
        path_to_images: string
            path to the root folder where images are stroed
        metadata: data frame
            dataframe with the name of each image
        sample: int
            number of images from the time series to plot
        scale_factor: flaot
            Factor to sacles images
        clip: int, default=None
            value to clip pixel images
        threshold: float
            threshold to remove the image
    """

    masked_images = []
    images = read_numpy_image(path_to_images + metadata.loc[2].Image)
    for img in images:
        masked_images.append(simple_cloud_mask_filter(img, scale_factor, clip, threshold))

    images = []
    labels = []
    for label, img in masked_images:
        images.append(img)
        labels.append(label)

    images = np.array(images)

    if clip:
        images = np.clip(images, 0, clip)

    print("------Statistics of RGB Bands------")
    print(f"Pixel Mean={images[:,0:3,:,:].mean().round(3)}")
    print(f"Pixel Min={images[:,0:3,:,:].min()}")
    print(f"Pixel Max={images[:,0:3,:,:].max()}")
    print("------------------------------------")

    i = 1
    fig, ax = plt.subplots(2, sample, figsize=(60, 10))
    images = images[:, 0:3, :, :] / scale_factor

    for i, img in enumerate(images[0:sample]):
        ax[0, i].imshow(img.transpose(1, 2, 0))
        if labels[i] == "Remove":
            ax[0, i].set_title(f"{labels[i]}", fontsize=20, color="r")
        else:
            ax[0, i].set_title(f"{labels[i]}", fontsize=20)

        sns.histplot(x=img.ravel(), ax=ax[1, i])
        ax[1, i].set_xlabel("Pixel Value")
        ax[1, i].legend(
            [f"Mean={img.mean().round(3)} \n Min={img.min().round(3)}, \n Max={img.max().round(3)}"],
            fontsize=10,
        )


# helper function for data visualization#
def visualize_images_and_masks(
    path_to_label,
    path_to_images,
    metadata,
    temporal_dim=True,
    n=5,
    figsize=(10, 5),
    brightness=0.0,
):

    """
    Plots RGB images and labels. If images come with extra temporal
    information, they must have the sahpe (T,C,W,H), otherwise
    they should have the shape (C,W,H).

    Argument:
        path_to_label: string
            path to labels
        path_to_images: string
            path to imags
        metadata: data frame
            dataframe with the name of each image and label
        temporal_dim: bool. default True
            wether images have temporal dimension or not.
            if images have temporal dimension, the must have the
            shape (T,C,W,H)
        n: int
            number of images to plot
        figsize: tuple
            matplotlib figure size
    """
    t = 1  ## alpha value
    cmap = {0: [1.0, 0.5, 0.5, t], 1: [0.5, 0.5, 0.1, t], 2: [0.2, 0.8, 0.2, t]}
    labels_pam = {
        0: "non_agricultural_area",
        1: "legal_exclusions",
        2: "agricultural_frontier",
    }
    patches = [mpatches.Patch(color=cmap[i], label=labels_pam[i]) for i in cmap]

    fig, ax = plt.subplots(2, n, figsize=figsize)

    seed = np.random.randint(low=0, high=100)

    if temporal_dim:
        for i in range(n):
            img = read_numpy_image(path_to_images + metadata["Image"].sample(n, random_state=seed).values[i])
            label = read_geotiff_image(path_to_label + metadata["Mask"].sample(n, random_state=seed).values[i])
            ax[0, i].imshow(np.clip(img[0][[0, 1, 2]].transpose(1, 2, 0), 0, 6000) / 6000 + brightness)
            ax[1, i].imshow(label)
            arrayShow = np.array([[cmap[i] for i in j] for j in label])
            ax[1, i].imshow(arrayShow)
            plt.legend(
                handles=patches,
                bbox_to_anchor=(1.05, 1),
                loc=2,
                borderaxespad=0.0,
                markerscale=30,
                fontsize="large",
            )

    else:
        for i in range(n):
            img = read_numpy_image(path_to_images + metadata["Image"].values[i])
            label = read_geotiff_image(path_to_label + metadata["Mask"].values[i])
            ax[0, i].imshow(np.clip(img[0:3].transpose(1, 2, 0), 0, 6000) / 6000 + brightness)
            ax[1, i].imshow(label)

            arrayShow = np.array([[cmap[i] for i in j] for j in label])
            ax[1, i].imshow(arrayShow)
            plt.legend(
                handles=patches,
                bbox_to_anchor=(1.05, 1),
                loc=2,
                borderaxespad=0.0,
                markerscale=30,
                fontsize="xx-large",
            )
            ax[0, i].axis("off")
            ax[1, i].axis("off")

    return fig


# helper function for data visualization#
def visualize_images_and_labels(
    path_to_images: str,
    metadata: pd.DataFrame,
    n: int = 5,
    temporal_dim: bool = False,
    rgb_bands: List[int] = [0, 1, 2],
    scale_factor: int = 6000,
    figsize: Tuple = (10, 5),
):

    """
    Plots RGB images and labels. If images come with extra temporal
    information, they must have the sahpe (T,C,W,H), otherwise
    they should have the shape (C,W,H).

    Argument:
        path_to_label: string
            path to labels
        path_to_images: string
            path to imags
        metadata: data frame
            dataframe with the name of each image and label
        temporal_dim: bool. default=True
            wether images have temporal dimension or not.
            if images have temporal dimension, the must have the
            shape (T,C,W,H)
        rgb_bands: List, default=[0,1,2]
            list with the indices of the rgb bands
        scale_factor: int, default=6000
            factor to scale images
        n: int
            number of images to plot
        figsize: tuple
            matplotlib figure size
    """

    fig, ax = plt.subplots(1, n, figsize=figsize)

    images = metadata.Image.tolist()
    labels = metadata.Labels.tolist()
    classes = metadata.Classes.tolist()

    if temporal_dim:
        for i in range(n):

            img = read_numpy_image(path_to_images + images[i])
            ax[i].imshow(img[0][rgb_bands].transpose(1, 2, 0))
            ax[i].axis("off")
            ax[i].set_title(f"{classes[i]} ({labels[i]})")

    else:
        for i in range(n):

            img = read_numpy_image(path_to_images + images[i])
            ax[i].imshow(np.clip(img[rgb_bands].transpose(1, 2, 0), 0, scale_factor) / scale_factor)
            ax[i].axis("off")
            ax[i].set_title(f"{classes[i]} ({labels[i]})")


def label_pixel_distributio(path_to_label: str, metadata: pd.DataFrame, select_classes: list):

    """
    Plots label distribution

    Arguments:
    ----------
        path_to_label: path to labels
        metadata: dataframe with the names of images and labels
        path_to_labels: path to the folder where labels are stored
        select_classes: list with the name of each class

    """

    unique = np.zeros(len(select_classes))

    for label in tqdm(metadata.Mask):
        lbl = read_geotiff_image(path_to_label + label)
        lbl[lbl == 2] = 1
        n_unique, counts = np.unique(lbl, return_counts=True)
        unique[n_unique] += counts

    unique = (unique * 100 / unique.sum()).round(3)
    fig = px.bar(x=select_classes, y=unique)
    fig.update_layout(xaxis_title="Labels", yaxis_title="Percentage")

    return fig
