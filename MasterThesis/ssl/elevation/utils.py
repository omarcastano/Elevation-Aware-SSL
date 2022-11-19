import numpy as np
import matplotlib.pyplot as plt
import torch


def visualize_augmented_images(dataset: torch.utils.data.Dataset, n: int = 10, brightness: float = 0.0) -> None:
    """
    Plots augmented images used to pre-train a elevations maps used to pre-train backbone

    Arguments:
    ----------
        dataset: pytorch dataset
            pytorch dataset which must return the augmented
            views of the same image and the image itself
        n: number of images to plot
    """

    fig, ax = plt.subplots(4, n, figsize=(32, 10))

    for i in range(n):
        image, image_1, image_2, elevation = dataset[i]

        image = np.array(image).transpose(1, 2, 0)
        image_1 = np.array(image_1).transpose(1, 2, 0)
        image_2 = np.array(image_2).transpose(1, 2, 0)

        ax[0, i].imshow(image + brightness)
        ax[1, i].imshow(image_1 + brightness)
        ax[2, i].imshow(image_2 + brightness)
        ax[3, i].imshow(elevation)

        ax[0, i].set_title("Original")
        ax[1, i].set_title("Augmented 1")
        ax[2, i].set_title("Augmented 2")

        ax[0, i].axis("off")
        ax[1, i].axis("off")
        ax[2, i].axis("off")
