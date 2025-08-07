import numpy as np
from osgeo import gdal
import logging

BINARY_NO_DATA_VAL = 255
PROBA_NO_DATA_VAL = -9999

numpy_to_gdal = {
    np.dtype(np.float64): 7,
    np.dtype(np.float32): 6,
    np.dtype(np.int32): 5,
    np.dtype(np.uint32): 4,
    np.dtype(np.int16): 3,
    np.dtype(np.uint16): 2,
    np.dtype(np.uint8): 1,
}

def make_geotiff(cloud_mask: np.ndarray, dataset_shape: tuple, outfp: str, proba: bool, threshold: float):
    nan_mask = np.isnan(cloud_mask)
    
    mem_driver, tiff_driver = gdal.GetDriverByName('MEM'), gdal.GetDriverByName('GTiff')
    opts = ['COMPRESS=LZW', 'COPY_SRC_OVERVIEWS=YES', 'TILED=YES', 'BLOCKXSIZE=256', 'BLOCKYSIZE=256']

    if proba:
        old_shape = cloud_mask.shape
        cloud_mask[nan_mask] = PROBA_NO_DATA_VAL
        cloud_mask = cloud_mask.reshape((dataset_shape[0], dataset_shape[1], 1))
        ds = mem_driver.Create('', cloud_mask.shape[1], cloud_mask.shape[0], cloud_mask.shape[2], numpy_to_gdal[cloud_mask.dtype])
        ds.GetRasterBand(1).WriteArray(cloud_mask[:,:,0])    
        ds.GetRasterBand(1).SetNoDataValue(PROBA_NO_DATA_VAL)
        
        sp = str(outfp).split('.')
        sp[-1] = '_proba'+sp[-1]
        _ = tiff_driver.CreateCopy('.'.join(sp), ds, options=opts)

        logging.info("Probability cloud mask saved to %s", '.'.join(sp))
        cloud_mask = cloud_mask.reshape(old_shape)

    cloud_mask[cloud_mask < threshold] = 0
    cloud_mask[cloud_mask > 0] = 1
    cloud_mask[nan_mask] = BINARY_NO_DATA_VAL
    cloud_mask = cloud_mask.astype(np.uint8)

    # Reshape into input shape
    cloud_mask = cloud_mask.reshape((dataset_shape[0], dataset_shape[1], 1))

    ds = mem_driver.Create('', cloud_mask.shape[1], cloud_mask.shape[0], cloud_mask.shape[2], numpy_to_gdal[cloud_mask.dtype])
    ds.GetRasterBand(1).WriteArray(cloud_mask[:,:,0])
    ds.GetRasterBand(1).SetNoDataValue(BINARY_NO_DATA_VAL)

    _ = tiff_driver.CreateCopy(outfp, ds, options=opts)
