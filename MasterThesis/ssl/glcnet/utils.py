import numpy as np
import matplotlib.pyplot as plt
import torch
import random


def visualize_augmented_images(dataset: torch.utils.data.Dataset, n: int = 10, brightness: float = 0.0, **kwargs) -> None:
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


def get_index(label1, label2, patch_size=(16, 16), patch_num=4):

    rois = np.zeros((patch_num, 10))

    index_i = 0
    range_x = patch_size[0] // 2
    range_x1 = label1.shape[0] - patch_size[0] // 2
    range_y = patch_size[1] // 2
    range_y1 = label1.shape[1] - patch_size[1] // 2

    list_for_select = label1[range_x:range_x1, range_y:range_y1].reshape(-1).tolist()
    list2 = label2[range_x:range_x1, range_y:range_y1].reshape(-1).tolist()
    list_for_select = list(set(list_for_select).intersection(list2))

    for i in range(patch_num):

        a = random.sample(list_for_select, 1)
        target1_index = np.argwhere(label1 == a)
        if len(target1_index.shape) == 2:
            if (
                target1_index[0][0] - range_x < 0
                or target1_index[0][0] + range_x > label1.shape[0]
                or target1_index[0][1] - range_y < 0
                or target1_index[0][1] + range_y > label1.shape[1]
            ):
                for i1 in range(1, target1_index.shape[0]):
                    if (
                        target1_index[i1][0] - range_x < 0
                        or target1_index[i1][0] + range_x > label1.shape[0]
                        or target1_index[i1][1] - range_y < 0
                        or target1_index[i1][1] + range_y > label1.shape[1]
                    ):
                        continue
                    else:
                        target1_index = target1_index[i1, :]
                        break
            else:
                target1_index = target1_index[0, :]
        target2_index = np.argwhere(label2 == a)
        if len(target2_index.shape) == 2:
            if (
                target2_index[0][0] - range_x < 0
                or target2_index[0][0] + range_x > label2.shape[0]
                or target2_index[0][1] - range_y < 0
                or target2_index[0][1] + range_y > label2.shape[1]
            ):
                for i1 in range(1, target2_index.shape[0]):
                    if (
                        target2_index[i1, 0] - range_x < 0
                        or target2_index[i1][0] + range_x > label2.shape[0]
                        or target2_index[i1][1] - range_y < 0
                        or target2_index[i1][1] + range_y > label2.shape[1]
                    ):
                        continue
                    else:
                        target2_index = target2_index[i1, :]
                        break
            else:
                target2_index = target2_index[0, :]

        rois[index_i, :] = [
            index_i,
            target1_index[1] - range_y,
            target1_index[0] - range_x,
            target1_index[1] + range_y - 1,
            target1_index[0] + range_x - 1,
            index_i,
            target2_index[1] - range_y,
            target2_index[0] - range_x,
            target2_index[1] + range_y - 1,
            target2_index[0] + range_x - 1,
        ]
        index_i += 1
        t_list = (
            label1[
                target1_index[0] - range_x : target1_index[0] + range_x,
                target1_index[1] - range_y : target1_index[1] + range_y,
            ]
            .reshape(-1)
            .tolist()
        )

        list_for_select1 = set(list_for_select).difference(t_list)
        if len(list_for_select1) > 1:
            list_for_select = list_for_select1

    return rois
