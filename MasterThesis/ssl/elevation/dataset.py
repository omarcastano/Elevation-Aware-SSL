import torch

import numpy as np

from MasterThesis import EDA
from MasterThesis.augmentation import data_augmentation, data_augmentation_v2


class CustomDataset(torch.utils.data.Dataset):

    """
    This class creates a custom dataset for implementing
    ElevationSSL self-supervised learning methodology

    Arguments:
    ----------
        path_to_images: str
            path to the folder where images are stored
        path_to_elevations: str
            path to the folder where elevations are stored
        metadata: data frame
            dataframe with the names of images and elevations
        return_original: If True also return the original image,
            so the output will be (original, augmented1, augmented2, Elevation), else
            the output will ve (augmented1, augmented2, Elevation)
    """

    def __init__(
        self,
        path_to_images,
        path_to_elevations,
        metadata,
        return_original=False,
        normalizing_factor: int = 6000,
        augment: dict = None,
        **kwargs
    ):

        super().__init__()
        self.augment = augment
        self.metadata = metadata
        self.path_to_images = path_to_images
        self.path_to_elevations = path_to_elevations
        self.return_original = return_original
        self.normalizing_factor = normalizing_factor

    def __len__(self):

        return len(self.metadata)

    def __getitem__(self, index):

        # Load and transform input image
        image = EDA.read_numpy_image(self.path_to_images + self.metadata.Image.tolist()[index])
        elevation = EDA.read_geotiff_image(self.path_to_elevations + self.metadata.Elevation.tolist()[index])

        image = np.clip(image[:3], 0, self.normalizing_factor) / self.normalizing_factor
        elevation = elevation / 500

        original_image = image.copy()
        original_image = torch.from_numpy(original_image.astype(np.float32))
        elevation = torch.from_numpy(elevation.astype(np.float32))

        # Data Augmentation
        image_1 = data_augmentation(image, augment=self.augment)
        image_2 = data_augmentation(image, augment=self.augment)

        return original_image, image_1, image_2, elevation
