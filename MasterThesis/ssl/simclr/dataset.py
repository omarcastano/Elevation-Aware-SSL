# import libraries
import torch
import numpy as np
from MasterThesis import EDA
from MasterThesis.augmentation import data_augmentation, data_augmentation_v2


class CustomDataset(torch.utils.data.Dataset):
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
        return_original: bool, default=False
            If True also return the original image,
            so the output will be (original, augmented1, augmented2), else
            the output will ve (augmented1, augmented2)
        normalizing_factor: float, default=6000
            value to clip and normalize input images
        augment: dict
            dictionary with the specific transformation to apply
            to input images. default values:
            augment = {
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
    """

    def __init__(self, path_to_images, metadata, return_original=False, normalizing_factor: int = 6000, augment: dict = None, **kwargs):

        super().__init__()
        self.augment = augment
        self.metadata = metadata
        self.path_to_images = path_to_images
        self.return_original = return_original
        self.normalizing_factor = normalizing_factor

    def __len__(self):

        return len(self.metadata)

    def __getitem__(self, index):

        # Load and transform input image
        image = EDA.read_numpy_image(self.path_to_images + self.metadata.Image.tolist()[index])

        if len(image.shape) == 4:
            image = EDA.less_cloudy_image(image)

        image = np.clip(image[:3], 0, self.normalizing_factor) / self.normalizing_factor

        original_image = image.copy()

        # Data Augmentation
        image_1 = data_augmentation_v2(image, augment=self.augment)
        image_2 = data_augmentation_v2(image, augment=self.augment)

        if isinstance(image_1, np.ndarray):

            image_1 = torch.from_numpy(image_1.astype(np.float32))
            image_2 = torch.from_numpy(image_2.astype(np.float32))

        if self.return_original:
            return original_image, image_1, image_2
        else:
            return image_1, image_2
