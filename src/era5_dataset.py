import glob
import os
import pickle

import h5py
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image, ImageFilter
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Sampler, TensorDataset
import subprocess


means = {
    't2m':2.788e2,
    'u10':-4.449e-2,
    'v10':1.777e-1
}
stds = {
    't2m':21.382,
    'u10':5.628,
    'v10':4.775
}
climate_vars = [
    't2m',
    'u10',
    'v10'
]
# train_years = ['2016','2017','2018','2019','2020','2021']
train_years = ['2020','2021']
val_years = ['2022']
months  = [
    str(i).rjust(2,'0') for i in range(1,13,1)
]


class ERA5Dataset(Dataset):

    def __init__(self,
                 location='./datasets/era5_new/', 
                 train=True, 
                 transform=None, 
                 upsampling_factor=4, 
                 noise_ratio= 0.0,
                 crop_size = 64,
                 n_patches = 8,
                 method = 'bicubic',
                 climate_vars = climate_vars
            ):
        self.location = location
        self.upsampling_factor = upsampling_factor
        self.train = train
        self.noise_ratio = noise_ratio
        self.transform = transform

        self.crop_size = crop_size #In this case, the crop_size is the size of the image crop from the HR ERA5 image
        self.n_patches = n_patches
        self.method = method
        self.crop_transform = transforms.RandomCrop(crop_size)
        self.climate_vars = climate_vars
        

        if self.train:
            self.years = train_years
        else:
            self.years = val_years

        self.months = months

        self.file_stats = {}
        for year in self.years:
            self.file_stats[year] = {}
            for month in self.months:
                self.file_stats[year][month] = {}

        self._get_files_stats()
        # we will always crop the image into square patches
        if (self.train == True) and (method == "bicubic"):
            self.bicubicDown_transform = transforms.Resize((int(self.crop_size/upsampling_factor),int(self.crop_size/upsampling_factor)),Image.BICUBIC, antialias=True)  # subsampling the image (half size)
        elif (self.train == False) and (method == "bicubic"):
            self.bicubicDown_transform = transforms.Resize((int((self.img_shape_x-1)/upsampling_factor),int(self.img_shape_y/upsampling_factor)),Image.BICUBIC, antialias=True)  # subsampling the image (half size)


    def _get_files_stats(self):
        self.n_samples_total = 0
        

        self.n_years = len(self.years)

        self.idxs_lookup_table= {}

        idx = 0
        for year in self.years:
            for month in self.months:
                file_path = os.path.join(self.location,year,month,'data_stream-oper_stepType-instant.nc')
                self.file_stats[year][month]['path'] = file_path
                self.file_stats[year][month]['file'] = None
                with h5py.File(file_path) as f:
                    self.file_stats[year][month]['n_samples_per_month'] = f['expver'].shape[0]

                    self.file_stats[year][month]['im_shape_x'] = f['latitude'].shape[0]
                    self.file_stats[year][month]['im_shape_y'] = f['longitude'].shape[0]

                    self.n_samples_total = self.n_samples_total + self.file_stats[year][month]['n_samples_per_month']

                    for i in range(self.file_stats[year][month]['n_samples_per_month']):
                        self.idxs_lookup_table[idx] = {}
                        self.idxs_lookup_table[idx]['year'] = year
                        self.idxs_lookup_table[idx]['month'] = month
                        self.idxs_lookup_table[idx]['day'] = i
                        idx = idx+1


                    f.close()

        self.img_shape_x = self.file_stats[self.years[0]][self.months[0]]['im_shape_x']
        self.img_shape_y = self.file_stats[self.years[0]][self.months[0]]['im_shape_y']
        

    def _open_file(self, year_idx,month_idx):
        file = h5py.File(self.file_stats[str(year_idx)][str(month_idx).zfill(2)]['path'])
        self.file_stats[str(year_idx)][str(month_idx).zfill(2)]['file'] = np.stack([file[clim_var] for clim_var in self.climate_vars],axis=3)

    def __len__(self):


        if self.train == True: 
            return self.n_samples_total * self.n_patches
        return self.n_samples_total

    def __getitem__(self, global_idx):
        year_idx, month_idx, day_idx = self.get_indices(global_idx)

        # Open image file if it's not already open
        if self.file_stats[str(year_idx)][str(month_idx).zfill(2)]['file'] is None:
            self._open_file(year_idx,month_idx)

        # file = h5py.File(self.file_stats[str(year_idx)][str(month_idx).zfill(2)]['path'])
        # data = np.stack([file[clim_var] for clim_var in self.climate_vars],axis=3)

        file = self.file_stats[str(year_idx)][str(month_idx).zfill(2)]['file']
        data = file[day_idx,:,:,:]


        # Apply transform and cut-off
        y = self.transform(data)
        y = y[:,:-1,:]

        # Modify y for training and get X based on method
        if self.train:
            y = self.crop_transform(y)
        X = self.get_X(y)

        return X, y

    def get_indices(self, global_idx):
        if self.train:
            map_idx = global_idx // self.n_patches
            year_idx = int(self.idxs_lookup_table[map_idx]['year'])
            month_idx = int(self.idxs_lookup_table[map_idx]['month'])
            day_idx = int(self.idxs_lookup_table[map_idx]['month'])
        else: 
            year_idx = int(self.idxs_lookup_table[global_idx]['year'])
            month_idx = int(self.idxs_lookup_table[global_idx]['month'])
            day_idx = int(self.idxs_lookup_table[global_idx]['month'])

        return year_idx, month_idx,day_idx

    def get_X(self, y):
        if self.method == "uniform":
            X = y[:, ::self.upsampling_factor, ::self.upsampling_factor]
        elif self.method == "noisy_uniform":
            X = y[:, ::self.upsampling_factor, ::self.upsampling_factor]
            X = X + self.noise_ratio * self.std * torch.randn(X.shape)
        elif self.method == "bicubic":
            X = self.bicubicDown_transform(y)
        else:
            raise ValueError(f"Invalid method: {self.method}")

        return X