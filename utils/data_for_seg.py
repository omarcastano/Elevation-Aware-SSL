#import libraries
import numpy as np
import pandas as pd
from MasterThesis import EDA
import torch
from sklearn.metrics import confusion_matrix

class CustomDaset(torch.utils.data.Dataset):

    """
    This class creates a custom dataset

    Arguments:
        path_to_images: str
            path to the folder where images are stored
        path_to_labels: str
            path to the folder where labels are stored
        metadata: data frame
            dataframe with the names of images and labels
    """

    def __init__(self, path_to_images, path_to_labels, metadata):

        super().__init__()
        self.metadata = metadata
        self.path_to_images = path_to_images
        self.path_to_labels = path_to_labels

    def __len__(self):

        return len(self.metadata)

    def __getitem__(self, index):

        #Load and transform input image
        image = EDA.read_numpy_image(self.path_to_images + self.metadata.Image[index])
        image = EDA.less_cloudy_image(image)
        image = np.clip(image[:3], 0, 6000)/6000
        image = image.astype(np.float32)

        #Load label
        label = EDA.read_geotiff_image(self.path_to_labels + self.metadata.Mask[index])
        label = torch.from_numpy(label).long()

        return image, label
