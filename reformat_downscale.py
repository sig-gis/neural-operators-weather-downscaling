import os
import numpy as np
import rasterio as rio
from rasterio.transform import from_bounds

path = 'california_2022_fno_32times_downscale.npy'

map_name = path.split('.')[0] 
if not os.path.isdir(os.path.join('output',map_name)):
    os.makedirs(os.path.join('output',map_name))

arr = np.load(path)

crs = 'epsg:4326'

california_lat_min = 32.3
california_lat_max = 42.2

california_lon_min = -124.7
california_lon_max = -113.9

transform = from_bounds(california_lon_min,california_lat_min,california_lon_max,california_lat_max,width=arr.shape[2],height=arr.shape[3])

for i in range(arr.shape[0]):
    dset = rio.open(
        f'output/{map_name}/{map_name}_{i}.tif',
        'w',
        driver='GTiff',
        height= arr.shape[2],
        width = arr.shape[3],
        count=arr.shape[1],
        dtype=arr.dtype,
        crs=crs,
        transform=transform
    )
    for b in range(arr.shape[1]):
        band = arr[i][b]
        dset.write(band,(b+1))

    dset.close()

