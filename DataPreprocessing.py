import geopandas as gpd
import matplotlib.pyplot as plt
import pyproj
import pandas as pd
import h5py    
import os
import numpy as np
from shapely.geometry import Point, Polygon
import folium
from osgeo import osr, gdal, ogr


def create_shapefiel_from_polygons(chip_metadata:dict, chip_name:str, path_to_save:str=None, crs:str='epsg:4326'):

    """
    Function that allow you to create a shapefile from polygons

    Args:
        chip_metadata: dict
            dictionary which contains corners coordintaes in
            epsg:4326 projection. Example
                            {'center_latlon': array([  6.77726963, -76.968011  ]),
                            'chip_id': '(0, 200)',
                            'chip_size': 100,
                            'corners': {'nw': array([  6.78180805, -76.97255079]),
                            'se': array([  6.77273122, -76.9634712 ])},
                            'grid_size': 10,
                            'patch_id': '18NTN_8_5',
                            'patch_size': 1000}
        chip_name: string
            names associated with the chip
        path_to_save: string, optional (default=None)
             path to save the shapefile.
        crs: string, optional (default='epsg:4326')
            projection for the output shapefi;e
    """

    #Defien coordinates
    nw = chip_metadata['corners']['nw'][::-1]
    se = chip_metadata['corners']['se'][::-1]
    ne = np.array([nw[0],  se[1]])
    sw = np.array([se[0],  nw[1]])
    coordinates = [sw, nw, ne, se]

    polygon = Polygon(coordinates)
    gdf = gpd.GeoDataFrame()
    gdf.loc[0,'chip'] = chip_name
    gdf.loc[0, 'geometry'] = polygon
    gdf.crs = 'epsg:4326'

    if crs != 'epsg:4326':
        gdf.to_crs(crs, inplace=True)

    if path_to_save:
        gdf.to_file(path_to_save)
    return gdf
