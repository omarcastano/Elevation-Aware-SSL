#import libraries
import numpy as np
import pandas as pd
import cv2
import albumentations as album
from MasterThesis import EDA
import torch
from sklearn.metrics import confusion_matrix

def data_augmentation(image, label, input_size):
    
    """
    Data augmentation such as vertical and horizontal flip, 
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

    aug = album.Compose(transforms=[
                                    album.VerticalFlip(p=0.5),
                                    album.HorizontalFlip(p=0.5),
                                    album.RandomRotate90(p=0.5),
                                    album.RandomSizedCrop(min_max_height=(80, 85), height=input_size[0], width=input_size[1], p=1.0),
                                    #album.HueSaturationValue(hue_shift_limit=1, sat_shift_limit=1, val_shift_limit=1, p=0.9)
                                    ])
    
    augmented = aug(image=image, mask=label)

    image , label = augmented['image'], augmented['mask']

    return image, label



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
        image = image.transpose(1, 2, 0).astype(np.float32)
        
        #Load label
        label = EDA.read_geotiff_image(self.path_to_labels + self.metadata.Mask[index])


        #Data Augmentation
        image, label = data_augmentation(image, label, input_size=[100,100])
        

        #Set data types compatible with pytorch
        label = torch.from_numpy(label).long()
        image = image.astype(np.float32).transpose(2,0,1)

        return image, label