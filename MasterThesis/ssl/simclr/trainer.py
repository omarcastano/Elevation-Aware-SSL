# import libraries
import os
import torch
import wandb
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
from MasterThesis.ssl.simclr import SimCLR
from MasterThesis.ssl.simclr.dataset import CustomDataset
from MasterThesis.ssl.simclr.utils import visualize_augmented_images
from lightly.loss import NTXentLoss

# Train one epoch
def train_one_epoch(
    train_loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: optim,
    scheduler: optim.lr_scheduler,
    device: torch.device,
):
    """
    Train SimCLR model for one epoch

    Arguments:
        train_dataloader: pytorch dataloader with training images and labels
        model: pytorch model for semantic segmentation
        criterion: loss function to train the self-supervised model
        optmizer: pytorch optimizer
        scheduler: pytorch learning scheduler
        device: device to tran the network such as cpu or cuda
    """

    running_loss = 0

    model.train()
    for epoch, (input1, input2) in enumerate(tqdm(train_loader), 1):

        # Set zero gradients for every batch
        optimizer.zero_grad()

        input1, input2 = input1.to(device), input2.to(device)

        # Get the keys and Queries
        q = model(input1)
        k = model(input2)

        # Compute the total loss
        loss = criterion(q, k)

        # compute gradients
        loss.backward()

        # Update weeigths
        optimizer.step()

        running_loss += loss

    scheduler.step()
    running_loss = running_loss / epoch

    return running_loss.item()


# Test one epoch
def test_one_epoch(
    test_loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    device: torch.device,
):

    """
    Evaluate SimCLR representation learning in one epoch

    Arguments:
        test_dataloader: pytorch dataloader with training images and labels
        model: pytorch model for semantic segmentation
        criterion: loss function to train the self-supervised model
        device: device to tran the network such as cpu or cuda
        Lambda: GLCNet loss function hyperparameter
    """

    running_loss = 0

    # model.eval()

    with torch.no_grad():
        for epoch, (input1, input2) in enumerate(tqdm(test_loader), 1):

            input1, input2 = input1.to(device), input2.to(device)

            # Get the keys and Queries
            q = model(input1)
            k = model(input2)

            # Compute the total loss
            loss = criterion(q, k)

            running_loss += loss

    running_loss = running_loss / epoch

    return running_loss.item()


# train model
def train_model(
    train_loader: DataLoader,
    test_loader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer,
    schedule,
    wandb: wandb,
    wandb_kwargs: dict,
    metadata_kwargs: dict,
):
    """
    Train SimCLR model

    Arguments
    ----------
    train_loader : DataLoader
        dataloader to train SimCLR
    test_loader : DataLoader
        dataloader to test SimCLR
    model : Module
        SimCLR model
    loss_fn :
        _description_
    optimizer : _type_
        _description_
    schedule : _type_
        _description_
    wandb : wandb
        wandb connector
    wandb_kwargs : dict
        kwargs used to load metrics to wandb
    metadata_kwargs : dict
        metadata to load data to train and test the model
    """
    if not wandb_kwargs["id"]:
        run_id = wandb.util.generate_id()
        print("--------------------")
        print("run_id", run_id)
        print("--------------------")

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

            train_loss = train_one_epoch(train_loader, model, loss_fn, optimizer, schedule, metadata_kwargs["device"])
            test_loss = test_one_epoch(test_loader, model, loss_fn, metadata_kwargs["device"])

            print(f"\n    Train Loss: {train_loss}")
            print(f"\n    Test Loss: {test_loss}")

            wandb.log({"Train Loss": train_loss, "Test Loss": test_loss})
            print("------------------------------------------------------------- \n")

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


# Run an experiment
def run_train(
    wandb: wandb,
    wandb_kwargs: dict = None,
    metadata_kwargs: dict = None,
):
    """
    Pre-train a backbone using SimCLR methodology

    Arguments:
    ---------
        metadata_kwargs: metadata required to train the model. Example:
            metadata_kwargs = {
                "path_to_images": # path where images are stored,
                "path_to_save_model": # Path to save the model that is being trained (do not include the extension .pt or .pth)
                "path_to_load_model": # Path to load a model from a checkpoint (useful to handle notebook disconnection)
                "metadata": # pandas dataframe with the metadata to load images,
                "metadata_train": # pandas dataframe with the metadata to load images
                "metadata_test": # pandas dataframe with the metadata to load images
                "device": # GPU or CPU,
            }

        wandb_kwargs: parameteres to initialize wandb runs. Example:
            wandb_kwargs = {
                "project": "Classification",
                "entity": "omar_castano",
                "config": hypm,
                "id": None,
                "name": "SSL-SimCLR",
                "resume": True,
            }

            Example of hyperparameter expected:
            hypm = {
            "version": # version reference to show in wandb,
            "pretrained": # SSL methodology. If not None path_to_load_backbone must be provided
            "amount_ss_data": # amount of data used to pre-train
            "input_shape": # shape of input images
            "backbone": # backbone use
            "in_channels": # number of input channels
            "temperature": # Temperature hyperparameter used in the NTXenLoss function
            "weight_decay": # optimizer hyperparameter
            "learning_rate": # learning rate
            "batch_size": # batch size
            "epochs": # epochs to pre-train
            }
    """
    # Define dataset
    ds_train = CustomDataset(
        metadata_kwargs["path_to_images"],
        metadata_kwargs["metadata_train"],
        normalizing_factor=wandb_kwargs["config"]["normalizing_factor"],
        augment=wandb_kwargs["config"]["augment"],
    )

    ds_test = CustomDataset(
        metadata_kwargs["path_to_images"],
        metadata_kwargs["metadata_test"],
        normalizing_factor=wandb_kwargs["config"]["normalizing_factor"],
        augment=wandb_kwargs["config"]["augment"],
    )

    ds_train_sample = CustomDataset(
        metadata_kwargs["path_to_images"],
        metadata_kwargs["metadata_train"],
        return_original=True,
        normalizing_factor=wandb_kwargs["config"]["normalizing_factor"],
        augment=wandb_kwargs["config"]["augment"],
    )

    visualize_augmented_images(ds_train_sample)

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
        batch_size=wandb_kwargs["config"]["batch_size"],
        shuffle=True,
        num_workers=2,
        drop_last=True,
    )

    # Instance Deep Lab model
    model = SimCLR(
        proj_hidden_dim=512,
        proj_output_dim=128,
        backbone=wandb_kwargs["config"]["backbone"],
        cifar=wandb_kwargs["config"]["cifar"],
    )

    model.to(metadata_kwargs["device"])

    # loss  functiction
    loss = NTXentLoss(
        temperature=wandb_kwargs["config"]["temperature"],
    )

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=wandb_kwargs["config"]["learning_rate"],
        weight_decay=wandb_kwargs["config"]["weight_decay"],
    )

    # learning scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=wandb_kwargs["config"]["epochs"], eta_min=4e-08)

    train_model(
        train_dataloader,
        test_dataloader,
        model,
        loss,
        optimizer,
        scheduler,
        wandb=wandb,
        wandb_kwargs=wandb_kwargs,
        metadata_kwargs=metadata_kwargs,
    )
