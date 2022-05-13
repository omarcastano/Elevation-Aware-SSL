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


def create_shapefiel_from_polygons(path_to_chip_metadata:str, chip_name:str, chip_padding:float=None, path_to_save:str=None, crs:str='epsg:4326'):

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
    
    if chip_padding:
        nw = nw + np.sign(nw)*chip_padding
        se = se - np.sign(se)*chip_padding
        ne = ne + np.array([-1,-1])*chip_padding
        sw = sw + np.array([1,1])*chip_padding

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


def polygons_intersection(shapefile1, shapefile2, chip_name=None, group_by='elemento' ,path_to_save=None , crs=None):

    """
    Functon that conputes the intesection between polygons stored in shapefiles.
    shapefile1: string or geo pandas dataframe
        either the path to the folder where a shapefile is stored or
        a geopandas dataframe with the polygons. Here you must provide 
        the "Fronteer".
    shapefile2: string or geo pandas dataframe
        either the path to the folder where a shapefile is stored or
        a geopandas dataframe with the polygons. Here you must provide the 
        squared polygons.
    chip_name: string:
        unique id for each chip.
    group_by: string.
        column name from the attribute table of shapefiel1 which will be used to create 
        class labels.
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

    unique_labels = shapefile1[group_by].unique()
    unique_labels.sort()

    label_num = np.arange(len(unique_labels))

    chip_references=[]
    geometry = []
    labels = []
    for label in unique_labels:
        chip_references.append(chip_name)
        sipra_mask = shapefile1.loc[shapefile1[group_by] == label, :]
        
        inter = shapefile2.sample(sipra_mask.shape[0], replace=True).intersection(sipra_mask, align=False)
        inter = inter[~inter.is_empty]

        if inter.shape[0] != 0:
            intersection = gpd.GeoDataFrame(inter, columns=['geometry'], crs = shapefile1.crs)
            intersection = gpd.GeoDataFrame(intersection.dissolve(), columns=['geometry'], crs = shapefile1.crs)
            geometry.append(intersection.geometry.values[0])
        else:
            geometry.append(Polygon())

        labels.append(label)

    intersection = gpd.GeoDataFrame({"chip_name":chip_references,"labels":labels,"labels_num":label_num ,"geometry":geometry}, crs = shapefile1.crs)

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

        
        
def shapefiel_to_geotiff(path_input_shp, path_output_raster, pixel_size, attribute, no_data_value=-999):


    """
    This function allow you to convert a shapefile in a Geotiff. In order to use this function
    shapefile projection must be in Cartesian system in meters.

    Arguments:
        path_input_shp: string.
            path where the shapefile is located.
        path_output_raster: string.
            path to save the output raster
        pixel_size: float.
            size of he pixel in the output raster
        attribute: string.
            attribute to burn pixel values.
        no_data_value: int (default=-999).
            integer for no data values              

    """
    
    #create the input Shapefile object, get the layer information, and finally set the extent values
    open_shp = ogr.Open(path_input_shp)
    shp_layer = open_shp.GetLayer()
    x_min, x_max, y_min, y_max = shp_layer.GetExtent()


    #calculate the resolution distance to pixel value:
    x_res = int((x_max - x_min) / pixel_size)
    y_res = int((y_max - y_min) / pixel_size)

    #image type
    image_type = 'GTiff'

    #Our new raster type is a GeoTiff, so we must explicitly tell GDAL to get this driver.
    #The driver is then able to create a new GeoTiff by passing in the filename or the
    #new raster that we want to create, called the x direction resolution, followed by the y
    #direction resolution, and then the number of bands; in this case, it is 1. Lastly, we set
    #a new type of GDT_Byte raster:
    driver = gdal.GetDriverByName(image_type)
    new_raster = driver.Create(path_output_raster, x_res, y_res, 1, gdal.GDT_Int16)
    new_raster.SetGeoTransform((x_min, pixel_size, 0, y_max, 0, -pixel_size))
    new_raster.SetProjection(shp_layer.GetSpatialRef().ExportToWkt())


    #Now we can access the new raster band and assign the no data values and the inner
    #data values for the new raster. All the inner values will receive a value of 255 similar
    #to what we set in the burn_values variable:


    # get the raster band we want to export too
    raster_band = new_raster.GetRasterBand(1)

    # assign the no data value to empty cells
    raster_band.SetNoDataValue(no_data_value)

    # run vector to raster on new raster with input Shapefile
    gdal.RasterizeLayer(new_raster, [1], shp_layer, options = [f"ATTRIBUTE={attribute}"])
    
    new_raster.FlushCache() 

    
def crop_geotiff_chip(path_to_chip_metadata, path_to_geotiff, path_to_save_cropped_geotiff, new_shape=(100,100), crs=3116):

    """
    Function center crops a geotiff image geotiff and then save the cropped geotiff using 
    coordinates stored in chip metadata.
    
    Arguments:
        path_to_chip_metadata: string
            Path to chip metadata
        path_to_geotiff: string
            path to geotiff image
        new_shape: tuple
            target shape of the new image 
        crs: int
            EPSG projection for the output GeoTiff
    """

    chip_metadata = pd.read_pickle(path_to_chip_metadata)

    nw = chip_metadata['corners']['nw']
    se = chip_metadata['corners']['se']


    InSR = osr.SpatialReference()
    InSR.ImportFromEPSG(4326)       # 4326/Geographic
    OutSR = osr.SpatialReference()
    OutSR.ImportFromEPSG(crs)     # Colombia Bogota zone

    Point = ogr.Geometry(ogr.wkbPoint)
    Point.AddPoint(nw[1], nw[0])  # use your coordinates here
    Point.AssignSpatialReference(InSR)    # tell the point what coordinates it's in
    Point.TransformTo(OutSR)              # project it to the out spatial reference


    ds = gdal.Open(path_to_geotiff)
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()

    nw, nh = ((arr.shape - np.array(new_shape))/2).astype(int)

    arr_out = arr[nw:-nw:,nh:-nh:]
    
    if arr_out.shape != new_shape:
        arr_out = arr_out[0:-1, 0:-1]
    
    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(path_to_save_cropped_geotiff, new_shape[1], new_shape[0], 1, gdal.GDT_Int16)

    geotf = list(ds.GetGeoTransform())
    geotf[0] = Point.GetX()
    geotf[3] = Point.GetY()

    outdata.SetGeoTransform(tuple(geotf))##sets same geotransform as input
    outdata.SetProjection(ds.GetProjection())##sets same projection as input
    outdata.GetRasterBand(1).SetNoDataValue(-9999)##if you want these values transparent
    outdata.GetRasterBand(1).WriteArray(arr_out)
    outdata.FlushCache()    
 

