import geopandas as gpd
import matplotlib.pyplot as plt
import pyproj
import plotly.express as px
import pandas as pd
import h5py    
import os
from shapely.geometry import Point, Polygon, MultiPolygon
import matplotlib.patches as mpatches
import numpy as np
import MasterThesis.DataPreprocessing as DP
from shapely.geometry import Polygon
import seaborn as sns
from osgeo import osr, gdal, ogr
import earthpy as et
import earthpy.spatial as es
import earthpy.plot as ep

#Function that reads geotiff
def read_geotiff_image(path):
    """
    Read geotiff 
    
    Arguments:
        path: string
            path to image
    """

    image = gdal.Open(path).ReadAsArray()
    return image
  
  
# Function that reads numpy image
def read_numpy_image(path):
    """
    Read image stored in .npz format 
    
    Arguments:
        path: string
            path to image
    """

    image = np.load(path)
    return image['arr_0']
 

#Function that plots all bands of a image
def visualize_image_bands(path_to_images, metadata):
    
    """
    Visualize all bands of a given image. The image is expected 
    to be in .npz format with shape (T,C,W,H).
    
    Arguments:
        path_to_images: string
            path to image
        metadata: DataFrame
            dataframe with the name of each image
    """


    image = read_numpy_image(path_to_images + metadata.loc[2].Image)
    print('#########################################')
    print("Image Shape (T,C,W,H):", image.shape)
    print('########################################')
    ep.plot_bands(image[0], cols=6, scale=True, cbar=False, figsize=(25,10));
