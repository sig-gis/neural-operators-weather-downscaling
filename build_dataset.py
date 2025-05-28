import numpy as np
import cdsapi
import datetime
import os
import argparse

parser = argparse.ArgumentParser(description='training parameters')

parser.add_argument('--startyear',type=str,required=False,default='2016')
parser.add_argument('--endyear',type=str,required=False,default='2022')

args = parser.parse_args()

dataset = "reanalysis-era5-single-levels"

start_year = int(args.startyear)
end_year = int(args.endyear)

# years = ['2020']
years = [ str(start_year +i ) for i in range(end_year - start_year + 1)] 
start_day = 1
end_day = 31
days = [ str(start_day +i ).zfill(2) for i in range(end_day - start_day + 1)]

variable_list = [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "2m_temperature",
        "surface_pressure",
        "2m_dewpoint_temperature",
        "total_precipitation",
        # "geopotential",
        "total_column_water"
    ]
file_format = 'netcdf'
download_format = 'unarchived'

for year in years:
    downloaded_file = f'mnt/eofm-benchmark/data/climatedownscaling/ERA5-hourly-{year}.nc'

    request = {
        'product_type':['reanalysis'],
        'variable':variable_list,
        'year':[year],
        'day':days,
        'month': ['01', '02', '03',
                    '04', '05', '06',
                    '07', '08', '09',
                    '10', '11', '12'],
        'time': [
                '00:00', '01:00', '02:00',
                '03:00', '04:00', '05:00',
                '06:00', '07:00', '08:00',
                '09:00', '10:00', '11:00',
                '12:00', '13:00', '14:00',
                '15:00', '16:00', '17:00',
                '18:00', '19:00', '20:00',
                '21:00', '22:00', '23:00',
            ],
        'data_format':file_format,
        'download_format':download_format
    }

    client = cdsapi.Client()

    client.retrieve(dataset,request,downloaded_file)