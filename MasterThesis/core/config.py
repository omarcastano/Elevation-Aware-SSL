import os
import multiprocessing as mp


def config_notebook(copy_data_to_content=True):

    # Install extra libraries
    os.popen("pip install 'opencv-python-headless<4.3'").read()
    os.popen("pip install mlxtend --upgrade --no-deps").read()
    os.popen("pip install -U albumentations").read()
    # os.popen("pip install -U git+https://github.com/albumentations-team/albumentations")

    print(os.popen("nvidia-smi").read())

    # Loggin into wandb
    import wandb

    wandb.login(key="ed6b4f7a25cd803c9ce7a66dbfba2353fe5bb5d2")

    if copy_data_to_content:
        # Copy data to content
        if not os.path.isdir("/content/LabelsGeoTiff"):
            os.popen(
                "cp -r '/content/drive/MyDrive/Colab Notebooks/Maestria Ing/Theses/GeoDataset/LabelsGeoTiff' /content"
            ).read()

        if not os.path.isdir("/content/LabelsGeoTiff"):
            os.popen(
                "cp -r '/content/drive/MyDrive/Colab Notebooks/Maestria Ing/Theses/GeoDataset/Sentinel_2_Images' /content"
            ).read()
