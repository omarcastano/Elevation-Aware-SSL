from elevation_aware_ssl.EDA import read_geotiff_image, read_numpy_image, less_cloudy_image
from elevation_aware_ssl.preprocessing import from_array_to_geotiff
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import clear_output
import numpy as np
import os
import sys
import elevation
import re
import shutil
from tqdm import tqdm
from elevation_aware_ssl import EDA
import multiprocessing as mp
from contextlib import contextmanager
import sys, os


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


path_to_data = "/media/omar/storage/gdrive/Maestria/Datasets/GeoDataset"

metadata = pd.read_csv("/media/omar/storage/gdrive/Maestria/Datasets/GeoDataset/metadata_v2/metadata.csv", usecols=["Image"])
metadata.Image = metadata.Image.str.replace("/chip.npy", "", regex=False)
metadata.rename({"Image": "image"}, axis=1, inplace=True)


def get_elevation_map(path_to_metadata, path_to_save_map, margin="0"):

    corners = pd.read_pickle(path_to_metadata)["corners"]
    west, north = corners["nw"][::-1]
    east, south = corners["se"][::-1]

    elevation.clip(bounds=(west, south, east, north), output=f"{path_to_save_map}", margin=margin)
    elevation.clean()


letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
letters_map = [{l: i for i, l in enumerate(letters)}][0]


def generate_elevation_maps(metadata):

    metadata_elevations = []
    for path in tqdm(metadata.image):
        print(path)
        if not os.path.isdir(f"{path_to_data}/Elevations/{path}"):
            os.makedirs(f"{path_to_data}/Elevations/{path}")

        get_elevation_map(f"{path_to_data}/Dataset/{path}/metadata.pkl", f"{path_to_data}/Elevations/elevation.tif")

        if not os.path.isdir(f"{path_to_data}/sentinel_geo_images/{path}"):
            os.makedirs(f"{path_to_data}/sentinel_geo_images/{path}")

        # img = read_numpy_image(f"{path_to_data}/Dataset/{path}/chip.npy")
        # from_array_to_geotiff(f"{path_to_data}/sentinel_geo_images/{path}/chip.tif", img[:3], f"{path_to_data}/Dataset/{path}/metadata.pkl", crs=3116)

        elv_map = EDA.read_geotiff_image(f"{path_to_data}/Elevations/elevation.tif")

        if elv_map.shape != (33, 33):
            os.remove(f"{path_to_data}/Elevations/elevation.tif")
            get_elevation_map(f"{path_to_data}/Dataset/{path}/metadata.pkl", f"{path_to_data}/Elevations/elevation.tif", margin="1%")

        shutil.move(f"{path_to_data}/Elevations/elevation.tif", f"{path_to_data}/Elevations/{path}/elevation.tif")

        id_number = re.sub(r"[a-zA-Z_\(\)\,\\\/\s+]", "", path)
        id_letter = re.sub(r"[\d_\(\)\,\\\/\s+]", "", path)
        id_letter = "".join(str(letters_map[l]) for l in id_letter[:3])
        id = int(id_number + id_letter)
        metadata_elevations.append({"Id": id, "Elevation": f"{path}/elevation.tif", "Image": f"{path}/chip.npy"})

    return pd.DataFrame(metadata_elevations)


metadata_elevations = generate_elevation_maps(metadata)
