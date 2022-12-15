import torch
import numpy as np
from MasterThesis import EDA
import pandas as pd
from MasterThesis.augmentation import data_augmentation_segmentation, data_augmentation_v2


class CustomDataset(torch.utils.data.Dataset):
    """
    This class creates a custom dataset for semantic segmentation

    Arguments:
    ----------
        path_to_images: str
            path to the folder where images are stored
        path_to_labels: str
            path to the folder where labels are stored
        metadata: data frame
            dataframe with the names of images and labels
        return_original: bool
            If True also return the original image,
            so the output will be (original, augmented1, augmented2), else
            the output will ve (augmented1, augmented2)
        normalizing_factor: float, default=6000
            factor to clip and normalize images
        augment: dict, default=none
            dictionary with the augmentations to perform
    """

    def __init__(
        self,
        path_to_images: str,
        path_to_labels: str,
        metadata: pd.DataFrame,
        return_original: bool = False,
        normalizing_factor: int = 6000,
        augment: dict = None,
        **kwargs
    ):

        super().__init__()
        self.augment = augment
        self.metadata = metadata
        self.path_to_images = path_to_images
        self.path_to_labels = path_to_labels
        self.return_original = return_original
        self.normalizing_factor = normalizing_factor

    def __len__(self):

        return len(self.metadata)

    def __getitem__(self, index):

        # Load input image
        image = EDA.read_numpy_image(self.path_to_images + self.metadata.Image.tolist()[index])

        # Load labels
        label = EDA.read_geotiff_image(self.path_to_labels + self.metadata.Mask.tolist()[index])

        if len(image.shape) == 4:
            image = EDA.less_cloudy_image(image)

        image = np.clip(image[:3], 0, self.normalizing_factor) / self.normalizing_factor

        image = image.astype(np.float32)
        original_image = image.copy()

        # Data Augmentation
        if self.augment:
            image, label = data_augmentation_v2(image, label, self.augment)

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image.astype(np.float32))

        # Set data types compatible with pytorch
        label = torch.from_numpy(label).long()

        if self.return_original:
            return original_image, image, label

        return image, label
