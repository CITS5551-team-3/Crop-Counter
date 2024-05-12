import rasterio
from rasterio.transform import from_origin
import numpy as np


input_png = 'data/plots.png'
output_tiff = 'data/plots.tif'


with rasterio.open(input_png) as src:
    data = src.read()
    transform = from_origin(0, 0, 0.0001, 0.0001)
    crs = {'init': 'epsg:4326'}

    out_meta = src.meta.copy()
    out_meta.update({
        'driver': 'GTiff',
        'height': src.height,
        'width': src.width,
        'transform': transform,
        'crs': crs
    })

    with rasterio.open(output_tiff, 'w', **out_meta) as dest:
        dest.write(data)
