import numpy as np
from geotiff import GeoTiff

import psutil
process = psutil.Process()

# test result: memory usage is 20KB and constant despite geoTIFF being 200MB

print(process.memory_info().rss)

tiff_file = "data/ortho_ic.tif"
geo_tiff = GeoTiff(tiff_file)
print(process.memory_info().rss)

zarr = geo_tiff.read()
print(process.memory_info().rss)
sum=0
for t,i in enumerate(zarr):
    for j in i:
        for k in j:
            sum+=int(k)  # k is uint8
    print(f"{t}  sum={sum}  rss={process.memory_info().rss}")

rgbimg = np.array(zarr)[:, :, :3]
print(process.memory_info().rss)

