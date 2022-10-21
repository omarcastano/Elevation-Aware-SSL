# import libraries
import numpy as np
import pandas as pd
import os
import albumentations as album
from MasterThesis import EDA
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.nn import Module
import torch
from tqdm import tqdm
from typing import List, Union, Tuple
import matplotlib.patches as mpatches
from MasterThesis.utils.segmentation import randominit
from MasterThesis.models.segmentation.v3p import DeepLab
from torchvision import transforms
import wandb
from sklearn.model_selection import KFold
from . import metrics
from IPython.display import clear_output


AUGMENTATIONS = {
    "horizontal_flip_prob": 0.5,
    "vertical_flip_prob": 0.5,
    "resize_scale": (0.85, 1.0),
    "resize_prob": 0.5,
    "brightness": 0.2,
    "contrast": 0.2,
    "saturation": 0.2,
    "hue": 0.1,
    "color_jitter_prob": 0.5,
    "gray_scale_prob": 0.2,
}

# Data loader
def data_augmentation(img, label, augment: dict = None):

    """
    Data augmentation for such as vertical and horizontal flip,
    random rotation and random sized crop.

    Arguments:
        img: 3D numpy array
            input image with shape (C,H,W)
        label: 1D numpy array
            labels with shape (H,W)
        transforms: dictionary
            kwargs to transform images
    """

    augment = augment or AUGMENTATIONS

    # Defines spatial transformation
    spatial_augment = album.Compose(
        [
            album.VerticalFlip(p=augment["vertical_flip_prob"]),
            album.HorizontalFlip(p=augment["horizontal_flip_prob"]),
            album.RandomResizedCrop(height=img.shape[1], width=img.shape[2], scale=augment["resize_scale"], p=augment["resize_prob"]),
        ]
    )

    # Defines Color Jitter
    color_jitter = transforms.ColorJitter(
        brightness=augment["brightness"],
        contrast=augment["contrast"],
        saturation=augment["saturation"],
        hue=augment["hue"],
    )

    # Defines color augmentation
    color_augment = transforms.Compose(
        [
            transforms.RandomApply([color_jitter], p=augment["color_jitter_prob"]),
            transforms.RandomGrayscale(p=augment["gray_scale_prob"]),
        ]
    )

    img = img.transpose(1, 2, 0)
    augmented = spatial_augment(image=img, mask=label)
    img, label = augmented["image"].transpose(2, 0, 1), augmented["mask"]

    img = torch.from_numpy(img.astype(np.float32))
    img = color_augment(img)

    return img, label


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

        # Load and transform input image
        image = EDA.read_numpy_image(self.path_to_images + self.metadata.Image.tolist()[index])

        # Load labels
        label = EDA.read_geotiff_image(self.path_to_labels + self.metadata.Mask.tolist()[index])
        label[label == 1] = 0
        label[label == 2] = 1

        if len(image.shape) == 4:
            image = EDA.less_cloudy_image(image)

        image = np.clip(image[:3], 0, self.normalizing_factor) / self.normalizing_factor

        image = image.astype(np.float32)
        original_image = image.copy()

        # Data Augmentation
        if self.augment:
            image, label = data_augmentation(image, label, self.augment)

        # Set data types compatible with pytorch
        label = torch.from_numpy(label).long()

        if self.return_original:
            return original_image, image.numpy(), label.numpy()

        return image, label


def visualize_augmented_images(dataset: torch.utils.data.Dataset, n: int = 10, classes_name: List[str] = None) -> None:

    """
    Plots augmented images used to train a segmentation model

    urgumetns:
    ----------
        dataset: pytorch dataset which must return the augmneted
                 views of the same image and the image itslef
        n: number of images to plot
        classes_name: name of each class
    """

    cmap = {i: [np.random.random(), np.random.random(), np.random.random(), 1] for i in range(len(classes_name))}
    labels_map = {i: name for i, name in enumerate(classes_name)}
    patches = [mpatches.Patch(color=cmap[i], label=labels_map[i]) for i in cmap]

    fig, ax = plt.subplots(3, n, figsize=(32, 10))

    for i in range(n):
        original, augmented, label = dataset[i]

        augmented = augmented.transpose(1, 2, 0)
        original = original.transpose(1, 2, 0)
        label = label

        ax[2, i].imshow(label)
        ax[1, i].imshow(augmented + 0.1)
        ax[0, i].imshow(original + 0.1)

        ax[0, i].set_title("Originale")
        ax[1, i].set_title("Augmented")
        ax[2, i].set_title("Label")

        ax[0, i].axis("off")
        ax[1, i].axis("off")
        ax[2, i].axis("off")

        arrayShow = np.array([[cmap[i] for i in j] for j in label])
        ax[2, i].imshow(arrayShow)

    plt.legend(
        handles=patches,
        bbox_to_anchor=(1.05, 1),
        loc=2,
        borderaxespad=0.0,
        markerscale=30,
        fontsize="large",
    )

    plt.show()


# define training one epoch
def train_one_epoch(
    train_dataloader: DataLoader,
    model: Module,
    loss_fn: Module,
    optmizer: torch.optim,
    schedule: torch.optim.lr_scheduler,
    class_name: list,
    device: torch.device,
):

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
    for epoch, (image, label) in enumerate(tqdm(train_dataloader), 1):

        images, labels = image.to(device), label.to(device)

        # set gradies to zero
        optmizer.zero_grad()

        # prediction
        outputs = model(images)

        # compute loss
        loss = loss_fn(outputs, labels)

        # compute gradients
        loss.backward()

        # update weigths
        optmizer.step()

        running_loss += loss.item()

        # compute confusion matrix
        labels = labels.cpu().detach().numpy()
        outputs = outputs.argmax(1).cpu().detach().numpy()
        conf_mt += metrics.pixel_confusion_matrix(labels, outputs, class_num=len(class_name))

    schedule.step()
    running_loss = np.round(running_loss / epoch, 4)
    scores, logs = metrics.model_evaluation(conf_mt, class_name=class_name, dataset_label="Train")
    logs.update({"Train loss": running_loss})

    return scores, logs


def test_one_epoch(
    test_loader: DataLoader,
    model: Module,
    loss_fn: Module,
    class_name: list,
    device: torch.device,
    last_epoch: bool = False,
):

    """
    Evaluate model performance using a test set with different metrics

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
        for epoch, (inputs, labels) in enumerate(tqdm(test_loader), 1):

            inputs, labels = inputs.to(device), labels.to(device)

            # Make prediction
            outputs = model(inputs)
            # Compute de los
            loss = loss_fn(outputs, labels)
            running_loss += loss.item()

            # Confusion matrix
            labels = labels.cpu().detach().numpy()
            output = outputs.argmax(1).cpu().detach().numpy()
            output_proba = torch.nn.functional.softmax(outputs, dim=1).cpu().detach().numpy()

            # I have to apply soft max
            conf_mt += metrics.pixel_confusion_matrix(labels, output, class_num=len(class_name))

            if last_epoch:
                metrics_by_threshold.metric_evaluation(labels, output_proba)

    running_loss = np.round(running_loss / epoch, 4)

    if last_epoch:
        scores, logs = metrics.model_evaluation(conf_mt, class_name=class_name, dataset_label="Test")
        logs.update({"Test loss": running_loss})
        return scores, logs, metrics_by_threshold

    else:
        scores, logs = metrics.model_evaluation(conf_mt, class_name=class_name, dataset_label="Test")
        logs.update({"Test loss": running_loss})
        return scores, logs, None


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
                "path_to_labels": path_to_label,
                "path_to_images": path_to_images,
                "path_to_save_model": None,  # Path to save the model that is being trained (do not include the extension .pt or .pth)
                "path_to_load_model": None,  # Path to load a model from a checkpoint (useful to handle notebook disconection)
                "path_to_load_backbone": None,  # Path to load a pre-trained backbone using ssl
                "metadata": metadata,
                "metadata_train": metadata_train[0],
                "metadata_test": metadata_test[0],
                "select_classes": select_classes,
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
                "fine_tune": True, # wheter or not to fine tune the the whole model (including the pre-trained backbone)
                "ft_epoch": 100, # epoch from which the frozen backbone will be unfreez
                "amount_of_ft_data": train_size,
                "bn_momentum": 0.9,
                "input_shape": [100, 100],
                "num_classes": 3,
                "in_channels": 3,
                "weight_decay": 0.0005,
                "learning_rate": 1e-3,
                "batch_size": 16,
                "epochs": 2,
            }
    """

    # Definte paths to save and load model
    checkpoint_path = f"{metadata_kwargs['path_to_save_model']}/checkpoint_{wandb_kwargs['name']}.pt"
    model_path = f"{metadata_kwargs['path_to_save_model']}/model_{wandb_kwargs['name']}.pth"
    checkpoint_load_path = f"{metadata_kwargs['path_to_load_model']}/checkpoint_{wandb_kwargs['name']}.pt"

    # Create folder to save model
    if metadata_kwargs["path_to_save_model"]:
        if not os.path.isdir(metadata_kwargs["path_to_save_model"]):
            os.mkdir(metadata_kwargs["path_to_save_model"])

    # Initialize WandB
    with wandb.init(**wandb_kwargs):

        wandb.log(
            {
                "Label Distribution": EDA.label_pixel_distributio(
                    metadata_kwargs["path_to_labels"], metadata_kwargs["metadata"], metadata_kwargs["select_classes"]
                )
            }
        )

        epoch = 0

        if metadata_kwargs["path_to_load_model"]:
            checkpoint = torch.load(checkpoint_load_path)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            epoch = checkpoint["epoch"]
            train_loss = checkpoint["train_loss"]
            test_loss = checkpoint["test_loss"]

        train_loss = []
        test_loss = []

        # Start Training the model
        while epoch < wandb_kwargs["config"]["epochs"]:
            print("-------------------------------------------------------------")
            print(f"Epoch {epoch+1}/{wandb_kwargs['config']['epochs']}")

            _, logs_train = train_one_epoch(
                train_loader,
                model,
                loss_fn,
                optimizer,
                schedule,
                metadata_kwargs["select_classes"],
                metadata_kwargs["device"],
            )

            last_epoch = epoch + 1 == wandb_kwargs["config"]["epochs"]
            _, logs_test, metrics_by_threshold = test_one_epoch(
                test_loader,
                model,
                loss_fn,
                metadata_kwargs["select_classes"],
                metadata_kwargs["device"],
                last_epoch,
            )

            print(f'\n    Train Loss: { logs_train["Train loss"] }')
            print(f'\n    Test Loss: { logs_test["Test loss"] }')

            train_loss.append(logs_train["Train loss"])
            test_loss.append(logs_test["Test loss"])

            logs_test.update({"Train loss": logs_train["Train loss"]})
            wandb.log(logs_test)
            print("------------------------------------------------------------- \n")

            if wandb_kwargs["config"]["fine_tune"] and (wandb_kwargs["config"]["ft_epoch"] == epoch):

                for parameters in model.backbone.parameters():
                    parameters.requires_grad = True

                for parameters in model.encoder.parameters():
                    parameters.requires_grad = True

                for g in optimizer.param_groups:
                    g["lr"] = wandb_kwargs["config"]["lr_ft"]

            # Save the model
            if metadata_kwargs["path_to_save_model"]:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "tarin_loss": train_loss,
                        "test_loss": test_loss,
                    },
                    checkpoint_path,
                )

                torch.save(model, model_path)

            epoch += 1

        # Create table with the losses
        loss = pd.DataFrame(
            {
                "train_loss": train_loss,
                "test_loss": test_loss,
                "epoch": np.array(range(len(train_loss))) + 1,
            }
        )

        # Save tables with metrics in wandB
        wandb.log({"Loss": wandb.Table(dataframe=loss)})
        metrics_table = wandb.Table(dataframe=metrics_by_threshold.get_table())
        wandb.log({"Table_Metrics": metrics_table})

        wandb.log({"Per Class Accuracy": metrics.plot_metrics_from_logs(logs_test, metric="Acc_by_Class")})
        wandb.log({"Recall": metrics.plot_metrics_from_logs(logs_test, metric="Recall")})
        wandb.log({"F1 Score": metrics.plot_metrics_from_logs(logs_test, metric="F1_score")})
        wandb.log({"Precision": metrics.plot_metrics_from_logs(logs_test, metric="Precision")})
        wandb.log({"Precision Recall Curve": metrics_by_threshold.plot_PR_curve()})
        wandb.log({"Precision by Threshold": metrics_by_threshold.get_bar_plot(metric="precision")})
        wandb.log({"Recall by Thresholds": metrics_by_threshold.get_bar_plot(metric="recall")})
        wandb.log({"F1_Score by Threshold": metrics_by_threshold.get_bar_plot(metric="f1_score")})
    clear_output(wait=True)


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
                "path_to_labels": #Path to labels
                "path_to_images": # Path to images
                "path_to_save_model": None,  # Path to save the model that is being trained
                "path_to_load_model": None,  # Path to load a model from a checkpoint (useful to handle notebook disconection)
                "path_to_load_backbone": None,  # Path to load a pre-trained backbone using ssl
                "metadata": # Dataframe with metadata to load images and labels
                "metadata_train": # Dataframe with metadata to load images and labels
                "metadata_test": # Dataframe with metadata to load images and labels
                "select_classes": # List with the unique classes
                "device": # device to train  de DeepLearning model
            }

        wandb_kwargs: parameteres to initialize wandb runs. Example:
            wandb_kwargs = {
                "project": "RandomInit",
                "entity": "omar_castano",
                "config": hypm,
                "id": None,
                "name": "RandomInit", # This parameter is used to identify the saved model
                "resume": False,
            }

            Example of hyperparameteres expected:
            hypm = {
                "version": "version",
                "pretrained": "SimCLR", # SSL methodology. If not None path_to_load_backbone must be provided
                "fine_tune": True, # wheter or not to fine tune the the whole model (including the pre-trained backbone)
                "ft_epoch": 100, # epoch from which the frozen backbone will be unfreez
                "amount_of_ft_data": train_size,
                "bn_momentum": 0.9,
                "input_shape": [100, 100],
                "num_classes": 3, # Hyperparameter of deepla model
                "in_channels": 3, # Hyperparameter of deepla model
                "weight_decay": 0.0005, # Hyperparameter of Adam optimizer
                "learning_rate": 1e-3, # Hyperparameter of Adam optimizer
                "batch_size": 16,  # Hyperparameter of deepla model
                "epochs": 2,  # Hyperparameter of deepla model
            }
    """

    if not wandb_kwargs["id"]:
        run_id = wandb.util.generate_id()
        print("run_id", run_id)

    # Define dataset
    ds_train = CustomDataset(
        metadata_kwargs["path_to_images"],
        metadata_kwargs["path_to_labels"],
        metadata_kwargs["metadata_train"],
        normalizing_factor=wandb_kwargs["config"]["normalizing_factor"],
        augment=wandb_kwargs["config"]["augment"],
    )
    ds_test = CustomDataset(
        metadata_kwargs["path_to_images"],
        metadata_kwargs["path_to_labels"],
        metadata_kwargs["metadata_test"],
        normalizing_factor=wandb_kwargs["config"]["normalizing_factor"],
        augment=None,
    )

    ds_train_sample = CustomDataset(
        metadata_kwargs["path_to_images"],
        metadata_kwargs["path_to_labels"],
        metadata_kwargs["metadata"],
        normalizing_factor=wandb_kwargs["config"]["normalizing_factor"],
        return_original=True,
        augment=wandb_kwargs["config"]["augment"],
    )

    visualize_augmented_images(ds_train_sample, classes_name=metadata_kwargs["select_classes"])

    # define dataloader
    train_dataloader = torch.utils.data.DataLoader(
        ds_train,
        batch_size=wandb_kwargs["config"]["batch_size"],
        shuffle=True,
        num_workers=2,
        drop_last=True,
    )
    test_dataloader = torch.utils.data.DataLoader(
        ds_test,
        batch_size=32,
        shuffle=True,
        num_workers=2,
        drop_last=True,
    )

    # Instance Deep Lab model
    torch.manual_seed(42)
    deeplab_model = DeepLab(
        num_classes=wandb_kwargs["config"]["num_classes"],
        in_channels=wandb_kwargs["config"]["in_channels"],
        pretrained=False,
        arch=wandb_kwargs["config"]["backbone"],
        output_stride=16,
        bn_momentum=wandb_kwargs["config"]["bn_momentum"],
        freeze_bn=False,
    )
    deeplab_model.to(metadata_kwargs["device"])

    if wandb_kwargs["config"]["pretrained"]:
        deeplab_model.backbone.load_state_dict(
            torch.load(metadata_kwargs["path_to_load_backbone"], map_location=torch.device("cpu")).backbone.state_dict()
        )
        deeplab_model.encoder.load_state_dict(
            torch.load(metadata_kwargs["path_to_load_backbone"], map_location=torch.device("cpu")).encoder.state_dict()
        )

        for parameters in deeplab_model.backbone.parameters():
            parameters.requires_grad = False

        for parameters in deeplab_model.encoder.parameters():
            parameters.requires_grad = False

        deeplab_model.to(metadata_kwargs["device"])

        print("\n --------------------------------------")
        print("The backbone was transfered successfully")
        print("----------------------------------------\n")

    # loss function
    loss = torch.nn.CrossEntropyLoss()

    # Optimizer
    optimizer = torch.optim.Adam(
        deeplab_model.parameters(),
        weight_decay=wandb_kwargs["config"]["weight_decay"],
        lr=wandb_kwargs["config"]["learning_rate"],
    )

    # Learning schedule
    schedule = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    train_model(
        train_dataloader,
        test_dataloader,
        deeplab_model,
        loss,
        optimizer,
        schedule,
        wandb=wandb,
        wandb_kwargs=wandb_kwargs,
        metadata_kwargs=metadata_kwargs,
    )


def generate_metadata_train_test_cv(
    metadata: pd.DataFrame, train_size: float, n_split: int = 5
) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:

    """
    Provides train/test indices to split data in train/test sets. Split
    dataset into k consecutive foldsenerate train and test dataset via k-fold sets


    Argguments:
    ----------
        metadata: dataframe with the path to images and labels
        train_size: percentage of in each train set
        n_split: numbers of folds. Must be at least 2
    """

    n_total_images = metadata.shape[0]
    n_images_train = int(n_total_images * train_size)

    print(f"Number fo total images : {n_total_images}")
    print(f"Number of images to train: {n_images_train} ({train_size*100:.3f}%)")

    metadata_train, metadata_test = [], []

    kf = KFold(n_splits=n_split)

    X = np.arange(10)
    for train, test in kf.split(metadata):
        metadata_train.append(metadata.iloc[train].sample(n_images_train, random_state=42).copy().reset_index(drop=True))
        metadata_test.append(metadata.iloc[test].copy().reset_index(drop=True))

    return metadata_train, metadata_test


def generate_metadata_train_test(train_size: float, test_size: float, metadata: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:

    """
    Generate a train/test set using via holdout set

    Arguments:
    ---------
        train_size: percentage of data for training
        test_size: percentage of data for test
        metadata: dataframe with the path to images and labels
    """

    # total number of images
    n_total_images = metadata.shape[0]
    n_images_train = int(n_total_images * train_size)
    n_images_test = int(n_total_images * test_size)

    # Metadata to train and test
    metadata_train = metadata.iloc[np.arange(0, n_images_train)].reset_index().copy()
    metadata_test = metadata.iloc[-np.arange(1, n_images_test + 1)].reset_index().copy()

    assert not np.isin(metadata_train.Image, metadata_test.Image).any()

    return metadata_train, metadata_test
