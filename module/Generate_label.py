import os
import numpy as np
from osgeo import gdal, ogr, osr




gdal.AllRegister()
ogr.RegisterAll()

def generate_label(tif_path,output_path):

    src_ds = gdal.Open(tif_path, gdal.GA_ReadOnly)
    if src_ds is None:
        print(f"unable open: {tif_path}")
        exit(1)


    geotransform = src_ds.GetGeoTransform()
    projection = src_ds.GetProjection()
    x_size = src_ds.RasterXSize
    y_size = src_ds.RasterYSize
    src_ds = None  


    if os.path.exists(output_path):
        print(f"del : {output_path}")
        os.remove(output_path)  


    driver = gdal.GetDriverByName('GTiff')
    target_ds = driver.Create(
        output_path,
        x_size,
        y_size,
        1,
        gdal.GDT_Byte,
        options=['COMPRESS=LZW']
    )

 
    target_ds.SetGeoTransform(geotransform)
    target_ds.SetProjection(projection)


    band = target_ds.GetRasterBand(1)

    band.Fill(0)  


    result = gdal.RasterizeLayer(
        target_ds,
        [1],  
        layer,
        burn_values=[255], 
        options=[
            'ALL_TOUCHED=TRUE',  
            'INIT_VALUE=0'
        ]
    )

    if result != 0:
        band = None
        target_ds = None
        return False

    band_array = band.ReadAsArray()
    if np.max(band_array) == 0:  

        band = None
        target_ds = None  
        os.remove(output_path)
        return False
    
    band = None
    target_ds = None


def has_two_same_chars(s, char):

    return s.count(char) >= 2

def extract_after_char(s, target_char):
    parts = s.split(target_char, 1)  
    return parts[1] if len(parts) > 1 else "not found"


polygon_fc = r"F:\LDZ\data_demo\ship_label2.shp"  


# 打开矢量文件(SHP格式)
driver = ogr.GetDriverByName('ESRI Shapefile')
vector_ds = driver.Open(polygon_fc, 0)
if vector_ds is None:
    print(f"unable open: {polygon_fc}")
    exit(1)

layer = vector_ds.GetLayer()
if layer.GetFeatureCount() == 0:
    print("no data")
    exit(1)


if __name__ == "__main__":
    tif_floder = r"F:\LDZ\data_demo\tif"  
    output_floder = r"F:\LDZ\data_demo\tif\label"

    tif_list = os.listdir(tif_floder)

    total = len(tif_list)

    for i,tif_name in enumerate(tif_list):
        if tif_name.endswith('.tif'):
            print(f"{i}/{total}",end="\r")
            tif_path = os.path.join(tif_floder,tif_name)
            # index = extract_after_char(tif_name,"_")
            # output_name = f"HN{index}"
            output_path = os.path.join(output_floder,tif_name)
            generate_label(tif_path,output_path)
        
