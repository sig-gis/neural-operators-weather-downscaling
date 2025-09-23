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
from torch.utils.data import DataLoader, Dataset, Sampler, TensorDataset,Subset
import subprocess

from src.era5_dataset import ERA5Dataset,means,stds
# from src.rtma_dataset import RTMADataset

def get_data_loader(args,data_tag,train,n_patches):

    climate_vars = ['t2m','u10','v10']
    mean_norm = [means[clim_var] for clim_var in climate_vars]
    std_norm = [stds[clim_var] for clim_var in climate_vars]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=mean_norm,
            std=std_norm
        )
        
    ])

    if args.data_name == 'era5':
        dataset = ERA5Dataset(
            location= args.data_path,
            train=train,
            upsampling_factor=args.upsampling_factor,
            noise_ratio=args.noise_ratio,
            transform=transform,
            crop_size=args.crop_size,
            method=args.method,
            climate_vars=['t2m','u10','v10'],
            n_patches=n_patches
        )

        val_dataset = ERA5Dataset(
            location=args.data_path,
            train=False,
            upsampling_factor=args.upsampling_factor,
            noise_ratio=args.noise_ratio,
            transform=transform,
            crop_size=args.crop_size,
            method=args.method,
            climate_vars=['t2m','u10','v10'],
            n_patches=n_patches
        )

        if len(dataset) > args.max_samples:
            idxs = np.arange(len(dataset))
            dataset = Subset(dataset=dataset,indices=idxs[:args.max_samples])
        if len(val_dataset) > args.max_val_samples:
            idxs = np.arange(len(dataset))
            val_dataset = Subset(dataset=val_dataset,indices=idxs[:args.max_val_samples])
    elif args.data_name == 'rtma':
        raise ValueError('dataset {} not recognized'.format(args.data_name))
    else:
        raise ValueError('dataset {} not recognized'.format(args.data_name))
        

    train_dataloader = DataLoader(dataset,
                                batch_size = int(args.batch_size),
                                num_workers = 0,
                                shuffle=True,  
                                sampler = None,
                                drop_last = False,)
    
    val_dataloader =  DataLoader(val_dataset,
                                batch_size = int(args.batch_size),
                                num_workers = 0,  
                                sampler = None,
                                drop_last = False,)
    return train_dataloader, val_dataloader, mean_norm, std_norm