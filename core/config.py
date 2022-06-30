import os

def config_notebook(copy_data_to_content=True):

    #github
    os.popen("git config --global user.email 'omar.castano@udea.edu.co'")
    os.popen("git config --global user.name 'omarcastano'")

    #Clone GLCNET repo
    os.popen("git clone https://github.com/GeoX-Lab/G-RSIM.git").read()
    os.popen("mkdir SSL").read()
    os.popen("cp -r /content/G-RSIM/T-SS-GLCNet/* /content/SSL").read()
    os.popen("rm -rf G-RSIM").read()


    #Install extra libraries
    os.popen("pip install mlxtend --upgrade --no-deps").read()
    os.popen("pip install geopandas").read()
    os.popen("pip install earthpy").read()
    os.popen("pip install wandb -qqq").read()

    print(os.popen("nvidia-smi").read())

    #Loggin into wandb
    import wandb
    wandb.login(key='ed6b4f7a25cd803c9ce7a66dbfba2353fe5bb5d2')

    if copy_data_to_content:
        #Copy data to content
        if not os.path.isdir('/content/LabelsGeoTiff'):
            os.popen("cp -r '/content/drive/MyDrive/Colab Notebooks/Maestria Ing/Theses/GeoDataset/LabelsGeoTiff' /content")

        if not os.path.isdir('/content/LabelsGeoTiff'):
            os.popen("cp -r '/content/drive/MyDrive/Colab Notebooks/Maestria Ing/Theses/GeoDataset/TrainImages' /content")