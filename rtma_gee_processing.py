import ee

ee.Authenticate()
ee.Initialize(project='eofm-benchmark')

california_lat_min = 32.3
california_lat_max = 42.2

california_lon_min = -124.7
california_lon_max = -113.9

date_start = '2020-01-01'
date_end = '2020-01-02'

california_bbox = ee.Geometry.BBox(california_lon_min,california_lat_min,california_lon_max,california_lat_max)

dataset = ee.ImageCollection('NOAA/NWS/RTMA').filter(ee.Filter.date(date_start,date_end))
dataset = dataset.filter(ee.Filter.bounds(california_bbox))

variables = dataset.select(['UGRD','VGRD','TMP','GUST','ACPC01','HGT'])


request = {
    'expression': variables,
    'fileFormat':'GEOPANDAS_GEODATAFRAME'
}

data = ee.data.computePixels(request)
