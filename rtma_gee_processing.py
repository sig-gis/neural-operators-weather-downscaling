import ee

ee.Authenticate()
ee.Initialize(project='eofm-benchmark')

california_lat_min = 32.3
california_lat_max = 42.2

california_lon_min = -124.7
california_lon_max = -113.9

california_bbox = ee.Geometry.BBox(california_lon_min,california_lat_min,california_lon_max,california_lat_max)

dataset = ee.ImageCollection('NOAA/NWS/RTMA').filter(ee.Filter.date('2017-01-01','2018-01-01'))
dataset = dataset.filter(ee.Filter.bounds(california_bbox))

variables = dataset.select(['UGRD','VGRD','TMP','GUST','ACPC01'])