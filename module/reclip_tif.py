import os
from osgeo import gdal
import numpy as np

count = 0

def crop_large_tif(input_tif, output_dir, threshold=800):
    global count

    os.makedirs(output_dir, exist_ok=True)
    

    src_ds = gdal.Open(input_tif)
    if src_ds is None:
        print(f"unable open TIF file: {input_tif}")
        return
    

    x_size = src_ds.RasterXSize
    y_size = src_ds.RasterYSize

    

    if x_size <= threshold and y_size <= threshold:

        output_path = os.path.join(output_dir, f"FJ{count}.tif")
        driver = gdal.GetDriverByName('GTiff')
        driver.CreateCopy(output_path, src_ds)
        count += 1
        return
    
    

    max_dim = int(max(x_size, y_size))
    split_factor = int(max_dim/500) + 1


    
    geotransform = src_ds.GetGeoTransform()
    origin_x = geotransform[0]
    origin_y = geotransform[3]
    pixel_width = geotransform[1]
    pixel_height = geotransform[5]
    

    x_step = x_size // split_factor
    y_step = y_size // split_factor
    

    driver = gdal.GetDriverByName('GTiff')
    

    for i in range(split_factor):
        for j in range(split_factor):

            x_off = j * x_step
            y_off = i * y_step
            

            x_size_crop = x_step if (j < split_factor - 1) else x_size - x_off
            y_size_crop = y_step if (i < split_factor - 1) else y_size - y_off
            

            new_origin_x = origin_x + x_off * pixel_width
            new_origin_y = origin_y + y_off * pixel_height
            

            output_path = os.path.join(output_dir, f"FJ{count}.tif")
            

            out_ds = driver.Create(
                output_path,
                x_size_crop,
                y_size_crop,
                src_ds.RasterCount,
                src_ds.GetRasterBand(1).DataType,
                options=['COMPRESS=LZW']
            )
            

            out_ds.SetGeoTransform((new_origin_x, pixel_width, 0, new_origin_y, 0, pixel_height))
            out_ds.SetProjection(src_ds.GetProjection())
            

            for band_idx in range(1, src_ds.RasterCount + 1):
                src_band = src_ds.GetRasterBand(band_idx)
                out_band = out_ds.GetRasterBand(band_idx)
                data = src_band.ReadAsArray(x_off, y_off, x_size_crop, y_size_crop)
                out_band.WriteArray(data)
            count += 1

            out_ds = None
            

    
    src_ds = None
    


def has_two_same_chars(s, char):

    return s.count(char) >= 2

# 使用示例
if __name__ == "__main__":

    input_tif = r"G:\clipimg\FJimg\FJ26.tif"
    output_dir = r"E:\clipimg\FJimg"
    crop_large_tif(input_tif, output_dir, threshold=800)