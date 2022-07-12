#import libraries
import numpy as np
import pandas as pd
import cv2
import albumentations as album
from MasterThesis import EDA
import torch
from sklearn.metrics import confusion_matrix
from MasterThesis.utils import metrics, simclr
from torch.utils.data import DataLoader
from torch.nn import Module
import torch

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
        
        if len(image.shape) == 4: 
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

#define training one epoch
def train_one_epoch(train_dataloader:DataLoader, model:Module, loss_fn:Module, optmizer:torch.optim, schedule:torch.optim.lr_scheduler, class_name:list, device:torch.device):

    """

    Arguments:
        train_dataloader: pytorch dataloader with training images and labels
        model: pytorch model for semantic segmentation
        loss_fn: loss function for semantic segmentation
        opimizer: pytorch optimizer
        class_name: list with the name of each class
        schduler: pytorch learning scheduler
        device: device to tran the network sucha as cpu or cuda

    """

    running_loss = 0
    conf_mt = 0

    model.train()
    for epoch, (image, label) in enumerate(train_dataloader, 1):

        images, labels = image.to(device), label.to(device)

        #set gradies to zero
        optmizer.zero_grad()

        #prediction
        outputs = model(images)

        #compute loss
        loss = loss_fn(outputs, labels)

        #compute gradients
        loss.backward()

        #update weigths
        optmizer.step()

        running_loss += loss.item()

        #compute confusion matrix
        labels = labels.cpu().detach().numpy()
        outputs = outputs.argmax(1).cpu().detach().numpy()
        conf_mt += metrics.pixel_confusion_matrix(labels, outputs, class_num = 3)

    schedule.step()
    running_loss = running_loss/epoch
    scores, logs = metrics.model_evaluation(conf_mt, class_name=class_name, dataset_label='Train')
    logs.update({"Train loss":running_loss})

    return scores, logs


def test_one_epoch(test_loader:DataLoader, model:Module, loss_fn:Module, class_name:list, device:torch.device, last_epoch:bool=False):

    """

    Arguments:
        train_dataloader: pytorch dataloader with training images and labels
        model: pytorch model for semantic segmentation
        loss_fn: loss function for semantic segmentation
        class_name: list with the name of each class
        device: device to tran the network sucha as cpu or cuda
        last_epoch: If true precision recall curve is computed 

    """

    running_loss = 0
    logs = {}
    metrics_by_threshold = metrics.threshold_metric_evaluation(class_name)
    conf_mt = 0

    model.eval()

    with torch.no_grad():
        for epoch , (inputs, labels) in enumerate(test_loader, 1):

            inputs, labels = inputs.to(device), labels.to(device)

            #Make prediction
            outputs = model(inputs)
            #Compute de los 
            loss = loss_fn(outputs, labels)
            running_loss += loss.item()

            #Confusion matrix
            labels = labels.cpu().detach().numpy()
            output = outputs.argmax(1).cpu().detach().numpy()
            output_proba = torch.nn.functional.softmax(outputs, dim=1).cpu().detach().numpy()

            ###I have to apply soft max 
            conf_mt += metrics.pixel_confusion_matrix(labels, output, class_num=3)

            if last_epoch:
                metrics_by_threshold.metric_evaluation(labels, output_proba)

    running_loss = running_loss/epoch

    if last_epoch:
        scores, logs = metrics.model_evaluation(conf_mt, class_name=class_name, dataset_label='Test')
        logs.update({"Test loss":running_loss})
        return scores, logs, metrics_by_threshold 
    else:
        scores, logs = metrics.model_evaluation(conf_mt, class_name=class_name, dataset_label='Test')
        logs.update({"Test loss":running_loss})
        return scores, logs, None