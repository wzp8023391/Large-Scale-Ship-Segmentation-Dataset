import os
from osgeo import gdal

def tif_to_png(input_tif, output_png, scale=True, resample_method='nearest', output_format='PNG'):


    gdal.AllRegister()
    

    src_ds = gdal.Open(input_tif, gdal.GA_ReadOnly)
    if src_ds is None:
        print(f"unable open TIF file: {input_tif}")
        return False
    

    width = src_ds.RasterXSize
    height = src_ds.RasterYSize
    bands = src_ds.RasterCount
    

    options = []

    if scale:
        options.append('-scale')
    

    options.extend(['-r', resample_method])
    

    translate_options = gdal.TranslateOptions(
        format=output_format,
        options=options
    )
    

    try:
        gdal.Translate(output_png, src_ds, options=translate_options)

        return True
    except Exception as e:
        print(f"error: {str(e)}")
        return False
    finally:

        src_ds = None


if __name__ == "__main__":
    mask_tiffloder = r"G:\clipimg\FJmask"
    img_tiffloder = r"G:\clipimg\FJtif"
    output_imgfloder = r"E:\img"
    output_maskfloder = r"E:\mask"

    input_path = r"G:\clipimg\FJtif\clip_8216.tif"
    output_path = r"G:\ship8216_0000.png"
    tif_to_png(input_path, output_path, scale=False, resample_method='nearest')