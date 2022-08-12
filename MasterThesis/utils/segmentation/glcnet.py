# Load libraries
from typing import List, Callable, Union, Tuple
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from tqdm.autonotebook import tqdm
import albumentations as album
import MasterThesis.preprocessing as DP
from MasterThesis import EDA
import random
from torch.utils.data import DataLoader
from torch.nn import Module
import torch
from torch import optim
import wandb
from MasterThesis.models import glcnet
from MasterThesis.utils.segmentation import contrast_loss


def data_augmentation(image: np.array, label: np.array, input_size: List[int]):

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

    aug = album.Compose(
        transforms=[
            album.RandomSizedCrop(
                min_max_height=(80, 85),
                height=input_size[0],
                width=input_size[1],
                p=1.0,
            ),
            album.VerticalFlip(p=0.5),
            album.HorizontalFlip(p=0.5),
            album.RandomRotate90(p=0.5),
            album.HueSaturationValue(
                hue_shift_limit=0.6,
                sat_shift_limit=(-0.05, 0.3),
                val_shift_limit=0.1,
                p=0.5,
            ),
            album.ToGray(always_apply=False, p=0.2),
            album.GaussianBlur(blur_limit=(1, 7), p=0.5),
        ]
    )

    augmented = aug(image=image, mask=label)
    image1, label1 = augmented["image"], augmented["mask"]

    augmented = aug(image=image, mask=label)
    image2, label2 = augmented["image"], augmented["mask"]

    return image1, label1, image2, label2


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


class CustomDaset(torch.utils.data.Dataset):

    """
    This class creates a custom pytorch dataset
    """

    def __init__(
        self,
        path_to_images: str,
        metadata: pd.DataFrame,
        patch_size: int = 16,
        patch_num: int = 4,
        return_original: bool = False,
    ):

        """
        Arguments:
        ----------
            path_to_images: path to the folder where images are stored
            metadata: dataframe with the names of images and labels
            patch_size: size of the patches to use in the local matching contrastive moduel
            patch_num: number of patches to use in the local matching contrastive moduel
            return_original: If True also return the original image,
                so the output will be (original, augmented, label), else
                the output will ve (augmented label)
        """

        super().__init__()
        self.metadata = metadata
        self.path_to_images = path_to_images
        self.return_original = return_original
        self.patch_size = patch_size
        self.patch_num = patch_num

    def __len__(self):

        return len(self.metadata)

    def __getitem__(self, index):

        # Load and transform input image
        image = EDA.read_numpy_image(
            self.path_to_images + self.metadata.Image.tolist()[index]
        )

        if len(image.shape) == 4:
            image = EDA.less_cloudy_image(image)

        image = np.clip(image[:3], 0, 6000) / 6000
        original_image = image.copy()
        image = image.transpose(1, 2, 0).astype(np.float32)

        w, h = image.shape[:2]

        # Generate Image
        label = np.expand_dims(
            np.array(range(w * h)).reshape(w, h),
            axis=2,
        )

        # Data Augmentation
        image1, label1, image2, label2 = data_augmentation(
            image, label, input_size=[w, h]
        )

        label1 = np.squeeze(label1.astype(np.float32))
        label2 = np.squeeze(label2.astype(np.float32))

        rois = get_index(
            label1, label2, (self.patch_size, self.patch_size), self.patch_num
        )

        rois = rois.astype(np.float32)

        # Set data types compatible with pytorch
        image1 = image1.astype(np.float32).transpose(2, 0, 1)
        image2 = image2.astype(np.float32).transpose(2, 0, 1)

        if self.return_original:

            return image1, image2, original_image

        return image1, image2, rois


def visualize_augmented_images(dataset: torch.utils.data.Dataset, n: int = 10) -> None:

    """
    Plots augmented images used to pre-train a model using SimCLR methodology

    Argumetns:
        dataset: pytorch dataset which must return the augmneted
                 views of the same image and the image itslef
        n: number of images to plot
    """

    fig, ax = plt.subplots(3, n, figsize=(32, 10))

    for i in range(n):
        image_1, image_2, image = dataset[i]

        image = image.transpose(1, 2, 0)
        image_1 = image_1.transpose(1, 2, 0)
        image_2 = image_2.transpose(1, 2, 0)

        ax[0, i].imshow(image + 0.1)
        ax[1, i].imshow(image_1 + 0.1)
        ax[2, i].imshow(image_2 + 0.1)

        ax[0, i].set_title("Originale")
        ax[1, i].set_title("Augmented 1")
        ax[2, i].set_title("Augmented 2")

        ax[0, i].axis("off")
        ax[1, i].axis("off")
        ax[2, i].axis("off")


# Train one epoch
def train_one_epoch(
    train_loader: DataLoader,
    model: Module,
    criterion: Module,
    optimizer: optim,
    scheduler: optim.lr_scheduler,
    device: torch.device,
    Lambda: float = 0.5,
):

    """
    Evaluate model performance using a test set with different metrics

    Arguments:
        train_dataloader: pytorch dataloader with training images and labels
        model: pytorch model for semantic segmentation
        criterion: loss function to train the self-supervised model
        opimizer: pytorch optimizer
        schduler: pytorch learning scheduler
        device: device to tran the network sucha as cpu or cuda
        Lambda: GLCNet loss function hyperparameter
    """

    running_loss = 0

    model.train()
    for epoch, (input1, input2, rois) in enumerate(tqdm(train_loader), 1):

        # Set zero gradients for every batch
        optimizer.zero_grad()

        input1, input2, rois = input1.to(device), input2.to(device), rois.to(device)

        # Get the keys and Queries
        q, k, q1, k1 = model(input1, input2, rois)

        # Compute the total loss
        loss = Lambda * criterion[0](q, k) + (1 - Lambda) * criterion[1](q1, k1)

        # compute gradients
        loss.backward()

        # Update weeigths
        optimizer.step()

        running_loss += loss

    scheduler.step()
    running_loss = running_loss / epoch

    return running_loss.item()


# Test the model every epoch
def test_one_epoch(
    test_loader: DataLoader,
    model: Module,
    criterion: Module,
    device: torch.device,
    Lambda: float = 0.5,
):

    """
    Evaluate model performance using a test set with different metrics

    Arguments:
        test_dataloader: pytorch dataloader with training images and labels
        model: pytorch model for semantic segmentation
        criterion: loss function to train the self-supervised model
        device: device to tran the network sucha as cpu or cuda
        Lambda: GLCNet loss function hyperparameter
    """

    running_loss = 0

    model.eval()

    with torch.no_grad():
        for epoch, (input1, input2, rois) in enumerate(tqdm(test_loader), 1):

            input1, input2, rois = input1.to(device), input2.to(device), rois.to(device)

            # Get the keys and Queries
            q, k, q1, k1 = model(input1, input2, rois)

            # Compute the total loss
            loss = Lambda * criterion[0](q, k) + (1 - Lambda) * criterion[1](q1, k1)

            running_loss += loss

        running_loss = running_loss / epoch

    return running_loss.item()


# train model
def train_model(
    train_loader: DataLoader,
    test_loader: DataLoader,
    model: Module,
    loss_fn,
    optimizer,
    schedule,
    wandb: wandb,
    wandb_kwargs: dict,
    metadata_kwargs: dict,
):

    """
    Train a semantic segmentation model

    Arguments:
    ---------
        metadata_kwargs: metadata required to train the model. Example:
            metadata_kwargs = {
                "path_to_images": path_to_images,
                "path_to_save_model": None,  # Path to save the model that is being trained (do not include the extension .pt or .pth)
                "path_to_load_model": None,  # Path to load a model from a checkpoint (useful to handle notebook disconection)
                "metadata": metadata,
                "metadata_train": metadata_train[0],
                "metadata_test": metadata_test[0],
                "device": device,
            }

        wandb_kwargs: parameteres to initialize wandb runs. Example:
            wandb_kwargs = {
                "project": "RandomInit",
                "entity": "omar_castano",
                "config": hypm,
                "id": None,
                "name": "RandomInit",
                "resume": False,
            }

            Example of hyperparameteres expected:
            hypm = {
                "version": "version",
                "pretrained": "SimCLR", # SSL methodology. If not None path_to_load_backbone must be provided
                "amount_of_ft_data": train_size,
                "bn_momentum": 0.9,
                "input_shape": [100, 100],
                "patch_size" : 40,
                "patch_num" : 16,
                "in_channels" : 3,
                "temperature" : 0.5, # Temperature hyperparameter used in the NTXenLoss function
                "weight_decay": 0.0005,
                "learning_rate": 1e-3,
                "batch_size": 16,
                "epochs": 2,
            }
    """

    # Definte paths to save and load model
    checkpoint_path = (
        f"{metadata_kwargs['path_to_save_model']}/checkpoint_{wandb_kwargs['name']}.pt"
    )
    model_path = (
        f"{metadata_kwargs['path_to_save_model']}/model_{wandb_kwargs['name']}.pth"
    )
    checkpoint_load_path = (
        f"{metadata_kwargs['path_to_load_model']}/checkpoint_{wandb_kwargs['name']}.pt"
    )

    # Create folder to save model
    if metadata_kwargs["path_to_save_model"]:
        if not os.path.isdir(metadata_kwargs["path_to_save_model"]):
            os.mkdir(metadata_kwargs["path_to_save_model"])

    # Initialize WandB
    with wandb.init(**wandb_kwargs):

        epoch = 1

        if metadata_kwargs["path_to_load_model"]:
            checkpoint = torch.load(checkpoint_load_path)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            epoch = checkpoint["epoch"]
            train_loss = checkpoint["train_loss"]
            test_loss = checkpoint["test_loss"]

        # Start Training the model
        while epoch <= wandb_kwargs["config"]["epochs"]:
            print("-------------------------------------------------------------")
            print(f"Epoch {epoch}/{wandb_kwargs['config']['epochs']}")

            train_loss = train_one_epoch(
                train_loader,
                model,
                loss_fn,
                optimizer,
                schedule,
                metadata_kwargs["device"],
            )

            test_loss = test_one_epoch(
                test_loader,
                model,
                loss_fn,
                metadata_kwargs["device"],
            )

            print(f"\n    Train Loss: {train_loss}")
            print(f"\n    Test Loss: {test_loss}")

            wandb.log({"Train Loss": train_loss, "Test Loss": test_loss})
            print("------------------------------------------------------------- \n")

            print(train_loss)
            # Save the model
            if metadata_kwargs["path_to_save_model"]:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_loss": train_loss,
                        "test_loss": test_loss,
                    },
                    checkpoint_path,
                )

                torch.save(model, model_path)

            epoch += 1


##Run a experiment
def run_train(
    wandb: wandb,
    wandb_kwargs: dict = None,
    metadata_kwargs: dict = None,
):

    """
    Run a experiment where a DeepLabv3+ model is trained using resnet50 as backbone. A pre-trained
    encoder can be loaded and finetune during the training.

    Arguments:
    ---------
        metadata_kwargs: metadata required to train the model. Example:
            metadata_kwargs = {
                "path_to_images": path_to_images,
                "path_to_save_model": None,  # Path to save the model that is being trained (do not include the extension .pt or .pth)
                "path_to_load_model": None,  # Path to load a model from a checkpoint (useful to handle notebook disconection)
                "metadata": metadata,
                "metadata_train": metadata_train[0],
                "metadata_test": metadata_test[0],
                "device": device,
            }

        wandb_kwargs: parameteres to initialize wandb runs. Example:
            wandb_kwargs = {
                "project": "RandomInit",
                "entity": "omar_castano",
                "config": hypm,
                "id": None,
                "name": "RandomInit",
                "resume": False,
            }

            Example of hyperparameteres expected:
            hypm = {
                "version": "version",
                "pretrained": "SimCLR", # SSL methodology. If not None path_to_load_backbone must be provided
                "amount_of_ft_data": train_size,
                "bn_momentum": 0.9,
                "input_shape": [100, 100],
                "patch_size" : 40,
                "patch_num" : 16,
                "in_channels" : 3,
                "temperature" : 0.5, # Temperature hyperparameter used in the NTXenLoss function
                "weight_decay": 0.0005,
                "learning_rate": 1e-3,
                "batch_size": 16,
                "epochs": 2,
            }
    """

    if not wandb_kwargs["id"]:
        run_id = wandb.util.generate_id()
        print("--------------------")
        print("run_id", run_id)
        print("--------------------")

    # Define dataset
    ds_train = CustomDaset(
        metadata_kwargs["path_to_images"],
        metadata_kwargs["metadata_train"],
        wandb_kwargs["config"]["patch_size"],
        wandb_kwargs["config"]["patch_num"],
    )

    ds_test = CustomDaset(
        metadata_kwargs["path_to_images"],
        metadata_kwargs["metadata_test"],
        wandb_kwargs["config"]["patch_size"],
        wandb_kwargs["config"]["patch_num"],
    )

    # define dataloader
    train_dataloader = torch.utils.data.DataLoader(
        ds_train,
        batch_size=wandb_kwargs["config"]["batch_size"],
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )
    test_dataloader = torch.utils.data.DataLoader(
        ds_test,
        batch_size=wandb_kwargs["config"]["batch_size"],
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )

    # Instance Deep Lab model    # Model
    deeplab_model = glcnet.DeepLab(
        num_classes=64,
        in_channels=wandb_kwargs["config"]["in_channels"],
        arch="resnet50",
        bn_momentum=wandb_kwargs["config"]["bn_momentum"],
        noStyle=False,
        noGlobal=False,
        noLocal=False,
        patch_size=wandb_kwargs["config"]["patch_size"],
        patch_num=wandb_kwargs["config"]["patch_num"],
    )

    deeplab_model.to(metadata_kwargs["device"])

    # loss  functiction
    loss = [
        contrast_loss.NTXentLoss(
            bs=wandb_kwargs["config"]["batch_size"],
            gpu=metadata_kwargs["device"],
            tau=wandb_kwargs["config"]["temperature"],
            cos_sim=True,
            use_gpu=(metadata_kwargs["device"].type == "cuda"),
        ),
        contrast_loss.NTXentLoss(
            bs=wandb_kwargs["config"]["batch_size"]
            * wandb_kwargs["config"]["patch_num"],
            gpu=metadata_kwargs["device"],
            tau=wandb_kwargs["config"]["temperature"],
            cos_sim=True,
            use_gpu=(metadata_kwargs["device"].type == "cuda"),
        ),
    ]

    # Optimizer
    optimizer = torch.optim.Adam(
        deeplab_model.parameters(),
        weight_decay=wandb_kwargs["config"]["weight_decay"],
        lr=wandb_kwargs["config"]["learning_rate"],
    )

    # learning scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=wandb_kwargs["config"]["epochs"], eta_min=4e-08
    )

    train_model(
        train_dataloader,
        test_dataloader,
        deeplab_model,
        loss,
        optimizer,
        scheduler,
        wandb=wandb,
        wandb_kwargs=wandb_kwargs,
        metadata_kwargs=metadata_kwargs,
    )
