import torch

import numpy as np

from elevation_aware_ssl import EDA
from elevation_aware_ssl.augmentation import data_augmentation, data_augmentation_v2
from elevation_aware_ssl.ssl.glcnet.utils import get_index


class CustomDataset(torch.utils.data.Dataset):
    """
    This class creates a custom pytorch dataset
    """

    def __init__(
        self,
        path_to_images,
        path_to_elevations,
        metadata,
        return_original=False,
        normalizing_factor: int = 6000,
        augment: dict = None,
        augment_original: dict = None,
        patch_size: int = 16,
        patch_num: int = 4,
        **kwargs
    ):
        """
        Arguments:
        ----------
            path_to_images: path to the folder where images are stored
            metadata: dataframe with the names of images and labels
            patch_size: size of the patches to use in the local matching contrastive module
            patch_num: number of patches to use in the local matching contrastive module
            return_original: If True also return the original image,
                so the output will be (original, augmented1, augmented2), else
                the output will ve (augmented1, augmented2)
        """

        super().__init__()
        self.metadata = metadata
        self.path_to_images = path_to_images
        self.return_original = return_original
        self.path_to_elevations = path_to_elevations
        self.patch_size = patch_size
        self.patch_num = patch_num
        self.augment = augment
        self.augment_original = augment_original
        self.normalizing_factor = normalizing_factor

    def __len__(self):

        return len(self.metadata)

    def __getitem__(self, index):

        # Load and transform input image
        image = EDA.read_numpy_image(self.path_to_images + self.metadata.Image.tolist()[index])
        elevation = EDA.read_geotiff_image(self.path_to_elevations + self.metadata.Elevation.tolist()[index])

        image = np.clip(image[:3], 0, self.normalizing_factor) / self.normalizing_factor
        elevation = elevation / 500

        image = image.astype(np.float32)
        original_image = image.copy()

        w, h = image.shape[-1], image.shape[-1]

        # Generate label
        label = np.expand_dims(np.array(range(w * h)).reshape(w, h), axis=2)

        # Data Augmentation
        original_image, elevation = data_augmentation_v2(image, elevation, augment=self.augment_original)
        image1, label1 = data_augmentation_v2(image, label, augment=self.augment)
        image2, label2 = data_augmentation_v2(image, label, augment=self.augment)

        elevation = elevation.astype(np.float32)

        label1 = np.squeeze(label1.astype(np.float32))
        label2 = np.squeeze(label2.astype(np.float32))

        rois = get_index(label1, label2, (self.patch_size, self.patch_size), self.patch_num)
        rois = rois.astype(np.float32)

        if self.return_original:

            return image1, image2, original_image

        return original_image, image1, image2, elevation, rois
