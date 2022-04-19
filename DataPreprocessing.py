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
from shapely.ops import unary_union


def create_shapefiel_from_polygons(path_to_chip_metadata:str, chip_name:str, path_to_save:str=None, crs:str='epsg:4326'):

    """
    Function that allow you to create a shapefile from chip corners cordinates.

    Args:
        path_to_chip_metadata: string
            path to the chip metadata file which muste 
            be in pickle format. 
        chip_name: string
            names associated with the chip
        path_to_save: string, optional (default=None)
             path to save the shapefile.
        crs: string, optional (default='epsg:4326')
            projection for the output shapefile
    """
    chip_metadata = pd.read_pickle(path_to_chip_metadata)

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


def polygons_intersection(shapefile1, shapefile2, path_to_save=None , crs=None):

    """
    Functon that conputes the intesection between polygons stored in shapefiles.
    shapefile1: string or geo pandas dataframe
        either the path to the folder where a shapefile is stored or
        a geopandas dataframe with the polygons
    shapefile2: string or geo pandas dataframe
        either the path to the folder where a shapefile is stored or
        a geopandas dataframe with the polygons
    path_to_save: string, optional (default=None)
        path to the folder where the shapefile which contains the 
        intersection will be stored
    crs: string, optional (default=None)
        Projection for the output shapefile. If None the output projection
        will be the same of input shapefiles.
    """

    if type(shapefile1) == str:
        shapefile1 = gpd.read_file(shapefile1)

    if type(shapefile2) == str:
        shapefile2 = gpd.read_file(shapefile2)

    data = []
    for indx1, info1 in shapefile1.iterrows():
        for indx2, info2 in shapefile2.iterrows():
            inter = info2['geometry'].intersection(info1['geometry'])
            data.append(inter)

    intersection = gpd.GeoDataFrame(unary_union(data), columns=['geometry'], crs = shapefile1.crs)
    #intersection = gpd.GeoDataFrame(intersection.dissolve(), columns=['geometry'], crs = shapefile1.crs)

    if (crs != shapefile1.crs) & (crs != None):
        intersection.to_crs(crs, inplace=True)

    if path_to_save:
        intersection.to_file(path_to_save)

    return intersection


def from_array_to_geotiff(path_to_save, array, path_to_chip_metadata, crs=3116):

    """
    Function that creates a GeoTiff from a numpy array and chip corners cordinates

    Arguments:
        path_to_save: string, optional (default=None)
            path to the folder where the GeoTiff will be saved.
        array: ndarray
            Image with dimension (C, H, W)
        path_to_chip_metadata: string
            path to the chip metadata file which muste 
            be in pickle format. 
        crs: integer (default=3116)
            EPSG projection for the output GeoTiff
    """
    chip_metadata = pd.read_pickle(path_to_chip_metadata)
    
    driver = gdal.GetDriverByName('GTiff')
    no_bands, heigth, width = array.shape
    DataSet = driver.Create(path_to_save, width, heigth, no_bands, gdal.GDT_Float64)

    nw = chip_metadata['corners']['nw']
    se = chip_metadata['corners']['se']


    InSR = osr.SpatialReference()
    InSR.ImportFromEPSG(4326)       # 4326/Geographic
    OutSR = osr.SpatialReference()
    OutSR.ImportFromEPSG(crs)     # Colombia Bogota zone

    Point = ogr.Geometry(ogr.wkbPoint)
    Point.AddPoint(nw[1], nw[0]) # use your coordinates here
    Point.AssignSpatialReference(InSR)    # tell the point what coordinates it's in
    Point.TransformTo(OutSR)              # project it to the out spatial reference


    DataSet.SetGeoTransform((Point.GetX(), 10, 0, Point.GetY(), 0, -10))


    srs = osr.SpatialReference()
    srs.ImportFromEPSG(crs)
    DataSet.SetProjection(srs.ExportToWkt())


    for i, image in enumerate(array, 1):
        DataSet.GetRasterBand(i).WriteArray(image/np.max(image))
