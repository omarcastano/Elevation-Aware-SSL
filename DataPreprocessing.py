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


def polygons_intersection(path_shapefiel1, path_shapefile2, path_to_save=None , crs=None):

    """
    Functon that conputes the intesection between polygons stored in shapefiles.

    path_shapwfiel1: string
        path to the folder where a shapefile is stored
    path_shapwfiel2: string
        path to the folder where a shapefile is stored
    path_to_save: string, optional (default=None)
        path to the folder where the shapefile which contains the 
        intersection will be stored
    crs: string, optional (default=None)
        Projection for the output shapefile. If None the output projection
        will be the same of input shapefiles.
    """
    shapefile1 = gpd.read_file(path_shapefiel1)
    shapefile2 = gpd.read_file(path_shapefile2)

    data = []
    for indx1, info1 in shapefile1.iterrows():
        for indx2, info2 in shapefile2.iterrows():
            inter = info2['geometry'].intersection(info1['geometry'])
            data.append(inter)

    intersection = gpd.GeoDataFrame(data, columns=['geometry'], crs = shapefile1.crs)
    intersection = gpd.GeoDataFrame(intersection.dissolve(), columns=['geometry'], crs = shapefile1.crs)

    if (crs != shapefile1.crs) & (crs != None):
        intersection.to_crs(crs, inplace=True)

    intersection.to_file(path_to_save)
    return intersection

def from_array_to_geotiff(path_to_save, array, chip_metadata, crs=3116):

    """
    Function that creates a GeoTiff from a numpy array and chip corners cordinates

    Arguments:
        path_to_save: string, optional (default=None)
            path to the folder where the GeoTiff will be saved.
        array: ndarray
            Image with dimension (C, H, W)
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
        crs: integer (default=3116)
            EPSG projection for the output GeoTiff

    
    """
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
