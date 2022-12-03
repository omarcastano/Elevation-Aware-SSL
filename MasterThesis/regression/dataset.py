# import libraries
import torch
import numpy as np
import pandas as pd
from MasterThesis import EDA
from typing import List, Union, Tuple
from torchvision import transforms
from MasterThesis.augmentation import data_augmentation


class CustomDataset(torch.utils.data.Dataset):
    """
    This class creates a custom pytorch dataset

    Arguments:
    ----------
        path_to_images: str
            path to the folder where images are stored
        metadata: pandas dataframe
            dataframe with the names of images and labels
        target_column: Union[List[str], str]
            name of the target column
        image_column: str
            name of the column having the name of images
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
        target_column: Union[List[str], str],
        image_column: str,
        return_original: bool = False,
        normalizing_factor: int = 6000,
        augment: bool = False,
        **kwargs,
    ):

        super().__init__()
        self.metadata = metadata
        self.path_to_images = path_to_images
        self.return_original = return_original
        self.normalizing_factor = normalizing_factor
        self.augmentation = augment
        self.target_column = target_column
        self.image_column = image_column

    def __len__(self):

        return len(self.metadata)

    def __getitem__(self, index):

        # Load and transform input image
        image = EDA.read_numpy_image(f"{self.path_to_images}{self.metadata[self.image_column].tolist()[index]}")

        if len(image.shape) == 4:
            image = EDA.less_cloudy_image(image)

        image = np.clip(image[:3], 0, self.normalizing_factor) / self.normalizing_factor

        original_image = image.copy()

        # Load label
        label = self.metadata[self.target_column].to_numpy()[index]
        label = torch.from_numpy(label.astype(np.float32))

        # Data Augmentation
        if self.augmentation:
            image = data_augmentation(image, augment=self.augmentation)

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image.astype(np.float32))

        if self.return_original:
            return original_image, image, label

        return image, label
