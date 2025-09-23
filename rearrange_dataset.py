import os
import numpy as np
import h5py

from torchvision import transforms

from src.era5_dataset import ERA5Dataset,means,stds
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--startyear',type=str,required=False,default='2020')
parser.add_argument('--endyear',type=str,required=False,default='2022')
parser.add_argument('--path',type=str,required=False,default='/home/rdemilt/delos_downscale/datasets/era5_california/')
parser.add_argument('--dest',type=str,required=False,default='/home/rdemilt/delos_downscale/datasets/era5_california/')

args = parser.parse_args()
# years = ['2016','2017','2018','2019','2020','2021','2022']
# months  = [
#     str(i).rjust(2,'0') for i in range(1,13,1)
# ]

years = ['2019','2020','2021']
months  = [
    str(i).rjust(2,'0') for i in range(1,13,1)
]


weather_vars = ['sp', 't2m', 'tcw', 'u10', 'v10']

for year in years:
    for month in months:
        orig_path = args.path + f'{year}/{month}/data_stream-oper_stepType-instant.nc'
        path = args.dest + f'{year}/{month}/hourly_data.h5'

        file = h5py.File(orig_path,mode='r')
        hrs = file['expver']
        lat = file['latitude']
        lon = file['longitude']

        new_file = h5py.File(path,mode='w')

        new_file.create_dataset('hrs',data=hrs)
        new_file.create_dataset('latitude',data=lat)
        new_file.create_dataset('longitude',data=lon)
        new_file.create_group('hourly_data')


        for i in range(hrs.shape[0]):
            new_file.create_group(f'hourly_data/{i}')
        for clim_var in weather_vars:
            arr = file[clim_var]
            for j in range(hrs.shape[0]):
                data_point = arr[j,:,:]
                new_file.create_dataset(f'hourly_data/{j}/{clim_var}',data=data_point)
