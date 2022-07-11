import os
import multiprocessing as mp

def config_notebook(copy_data_to_content=True):

    #github
    os.popen("git config --global user.email 'omar.castano@udea.edu.co'")
    os.popen("git config --global user.name 'omarcastano'")

    #Clone GLCNET repo
    os.popen("git clone https://github.com/GeoX-Lab/G-RSIM.git").read()
    os.popen("mv /content/G-RSIM/T-SS-GLCNet/* /content").read()
    os.popen("rm -rf G-RSIM").read()


    #Install extra libraries
    os.popen("pip install 'opencv-python-headless<4.3'").read()
    os.popen("pip install mlxtend --upgrade --no-deps").read()
    os.popen("pip install geopandas").read()
    os.popen("pip install earthpy").read()
    os.popen("pip install wandb -qqq").read()
    os.popen("pip install -U albumentations").read()
    #os.popen("pip install -U git+https://github.com/albumentations-team/albumentations")


    print(os.popen("nvidia-smi").read())

    #Loggin into wandb
    import wandb
    wandb.login(key='ed6b4f7a25cd803c9ce7a66dbfba2353fe5bb5d2')

    if copy_data_to_content:
        #Copy data to content
        if not os.path.isdir('/content/LabelsGeoTiff'):
            os.popen("cp -r '/content/drive/MyDrive/Colab Notebooks/Maestria Ing/Theses/GeoDataset/LabelsGeoTiff' /content").read()

        if not os.path.isdir('/content/LabelsGeoTiff'):
            os.popen("cp -r '/content/drive/MyDrive/Colab Notebooks/Maestria Ing/Theses/GeoDataset/Sentinel_2_Images' /content").read()


def load_image_and_labels(img_path, label_path):

    """
    Load Image and label given the path

    Arguments:
        img_path: path to image
        label_path:  path to label
    """
    from MasterThesis import EDA

    img = EDA.read_numpy_image(img_path)
    lbl = EDA.read_geotiff_image(label_path)

def image_label_sanity_check(metadata, path_to_images, path_to_labels):
    
    """
    load all images and labels to verify they exist on the folder

    Arguments:
       metadata: dataframe with the path to label and images
       path_to_images: path to the folder where all images are stored
       path_to_labels: path to the folder where all labels are stored

    """

    with mp.Pool(mp.cpu_count()) as p:
        p.starmap(load_image_and_labels, zip(path_to_images + metadata.Image , path_to_labels + metadata.Mask))
        
