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
import xarray as xr

class RTMADataset(Dataset):
    def __init__(self, 
                 location:str, 
                 data_tag:str, 
                 train:bool, 
                 transform, 
                 upsampling_factor:int, 
                 noise_ratio:float, 
                 std:list[float],
                 crop_size:int,
                 n_patches:int,
                 method:str):
        self.location = location
        self.data_tag = data_tag
        self.upsampling_factor = upsampling_factor
        self.train = train
        self.noise_ratio = noise_ratio
        self.std = torch.Tensor(std).view(len(std),1,1)
        self.transform = transform
        self._get_file_stats()
        self.crop_size = crop_size #In this case, the crop_size is the size of the image crop from the HR ERA5 image
        self.n_patches = n_patches
        self.method = method
        self.crop_transform = transforms.RandomCrop(crop_size)
        # we will always crop the image into square patches
        if (self.train == True) and (method == "bicubic"):
            self.bicubicDown_transform = transforms.Resize((int(self.crop_size/upsampling_factor),int(self.crop_size/upsampling_factor)),Image.BICUBIC, antialias=True)  # subsampling the image (half size)
        elif (self.train == False) and (method == "bicubic"):
            self.bicubicDown_transform = transforms.Resize((int((self.img_shape_x-1)/upsampling_factor),int(self.img_shape_y/upsampling_factor)),Image.BICUBIC, antialias=True)  # subsampling the image (half size)
   
    def _get_file_stats(self):
        cmd = f"gcloud storage ls {self.location}"
        out = subprocess.run(cmd,shell=True,stderr=subprocess.PIPE,stdout=subprocess.PIPE).stdout.decode("utf-8")
        files = out.splitlines()
        var_files = [z for z in files if self.data_tag in z]
        
        self.files_paths = var_files # zarr stores
        self.files_paths.sort()
        self.locations = len(self.files_paths)
        self.n_years = 12
        self.n_samples_per_year = 8760
        self.n_samples_total = self.n_years * self.n_samples_per_year * self.locations
        self.files = [None for _ in range(self.locations)]

        print(f"File stats from {self.files_paths[0]}:\n")        
        _f = xr.open_zarr(self.files_paths[0])
     
        self.n_in_channels = _f.sizes['band'] # (8760*12) hourly bands per zarr store, one sample is (CHW): 1,40,40
        self.img_shape_x = _f.sizes['x']
        self.img_shape_y = _f.sizes['y']
  
        print("Number of samples per year: {}".format(self.n_samples_per_year))
        print("Found data at path {}. Number of examples: {}. Image Shape: {} x {} x {}".format(self.location, self.n_samples_total, self.img_shape_x, self.img_shape_y, 1))
        return
            
    def __len__(self):
        if self.train == True: 
            return self.n_samples_total * self.n_patches
        return self.n_samples_total
    
    # kyle: rtma data files are not ID'd by year but by spatial tile. 
    # each file is a spatial tile over CA containing all hours of every year from 2011-2022 (12*8760)
    def get_indices(self, global_idx):
        if self.train:
            tile_idx,local_idx = divmod(global_idx, self.n_samples_per_year*self.n_years*self.n_patches)
        else:
            tile_idx,local_idx = divmod(global_idx, self.n_samples_per_year*self.n_years)

        return tile_idx, local_idx
    
    def _open_file(self, tile_idx):
        
        _file = xr.open_zarr(self.files_paths[tile_idx])
        self.files[tile_idx] = _file 
    
    def __getitem__(self, global_idx):
        tile_idx, local_idx = self.get_indices(global_idx)

        # Open image file if it's not already open
        if self.files[tile_idx] is None:
            self._open_file(tile_idx)

        # Apply transform and cut-off
        # kyle: y is the coarsened sample (?) so what we store as y here should be the one sample in our case  (CHW): 1,40,40 sliced out of our xr.Dataset
        file_data = self.files[tile_idx]
        sample = file_data.isel(band=local_idx).to_array().to_numpy()[1,:,:] # will not work in edge cases where self.n_patches multiplier makes local_idx > (8760*12)
        sample = np.expand_dims(sample, axis=0) # now we're of shape [1,40,40]
        y = self.transform(sample) 
        # y = y[:,:-1,:] # kyle: this was here from GetClimateDatasets __getitem__ method but don't think we want to do this

        # Modify y for training and get X based on method
        if self.train:
            y = self.crop_transform(y)
        X = self.get_X(y)

        return X, y

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


class BatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        assert batch_size % self.dataset.number_of_regions == 0

        ## We want to have equal number of samples from all the regions in a batch
        self.samples_per_region = batch_size // self.dataset.number_of_regions
        total_samples = self.dataset.number_of_regions * self.dataset.total_samples_per_region
        # self.total_batches = (total_samples + self.batch_size - 1) // self.batch_size #to include the last partial batch if it exists
        ## Total number of batches to have in the dataloader
        self.total_batches = total_samples // self.batch_size

    def __len__(self):
        return self.total_batches

    def __iter__(self):
        
        for _ in range(self.total_batches):
            batch = []
            # Create a shuffled list of region indices, repeated for the number of samples needed from each region
            region_sequence = np.repeat(np.arange(self.dataset.number_of_regions), self.samples_per_region)
            np.random.shuffle(region_sequence) 

            # Pick one timestep randomly for each region index in the above list
            for region_index in region_sequence:
                chosen_index = np.random.randint(self.dataset.total_samples_per_region)
                batch.append((region_index, chosen_index))
                if len(batch) == self.batch_size:
                    break

            yield batch

# testing RTMA     
transform = torch.from_numpy

# TODO: made the stats up to get the rest of the code done, full global stats of rtma ws and wd need to be run, would use rtma_compute_stats.py and rtma_pooled_stats.py
mean = [8.63, 132.2] 
std = [4.5, 42.5]

rtma =RTMADataset(location="gs://delos-downscale/data/rtma/zarr", 
                  data_tag = "ws", 
                  train=True, 
                  transform=transform,
                  upsampling_factor=2, 
                  noise_ratio=2, 
                  std=std, 
                  crop_size=30, # this and n_patches are buggy but do we actually want to be cropping a 40x40px sample down more for a sample?
                  n_patches=1, # increasing beyond 1 breaks things..
                  method="uniform")

X,y = rtma.__getitem__(10_000_000)
print(X.shape, y.shape)

X,y = rtma.__getitem__(20_544_150)
print(X.shape, y.shape)

dataloader = DataLoader(rtma,
                        batch_size = int(5),
                        num_workers = 8, 
                        shuffle = True,   # (train == True)
                        sampler = None,
                        drop_last = False,
                        pin_memory = torch.cuda.is_available())
x, y = next(iter(dataloader))