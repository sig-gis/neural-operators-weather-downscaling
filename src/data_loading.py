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


def getTrainData(args, std):  
    '''
    Loading train and valid data loaders from dataset in folders: (a) datasets/era5; (b) datasets/era_to_wtk
    ===
    std: the channel-wise standard deviation of each dataset, list: [#channels]
    '''
    
    train_loader = get_data_loader(args, '/train', train=True, n_patches=args.n_patches, std=std)
    val_loader = get_data_loader(args, '/valid', train=True, n_patches=args.n_patches, std=std)     

    return train_loader, val_loader 

def getTestData(args, std):  
    '''
    Loading test data loaders from dataset in folders: (a) datasets/era5; (b) datasets/era_to_wtk/region_name
    ERA5->ERA5: our paper only includes results on the extrapolation dataset (test_2), see SuperBench paper for details.
    ERA5->WTK : this has two test folders : test_1 for 5x upsampling factor and test_2 for 15x. If doing zero-shot evaluation in this case, use the 15x, i.e. test_2 folder
    ===
    '''
    test_loader = get_data_loader(args, '/test_2', train=False, n_patches=args.n_patches, std=std)        
    return test_loader 


def get_data_loader(args, data_tag, train, n_patches, std):
    
    transform = torch.from_numpy

    if args.data_name == 'era5':
        dataset = GetClimateDataset(args.data_path+data_tag, train, transform, args.upsampling_factor, args.noise_ratio, std, args.crop_size, n_patches, args.method) 
    
    elif args.data_name == "wtk":
        dataset = GetWTKDataset(args.data_path, data_tag, train, transform, args.crop_size, args.upsampling_factor)
        batch_sampler = BatchSampler(dataset, batch_size=int(args.batch_size))

    else:
        raise ValueError('dataset {} not recognized'.format(args.data_name))

    if args.data_name == "wtk" and train is True:
        #In this case a single model is trained for both regions, so while training, 
        #we ensure that every batch has an equal number of LR and HR pairs from both regions, see batch_sampler
        dataloader = DataLoader(dataset,
                                num_workers=8,   
                                batch_sampler=batch_sampler,  
                                pin_memory=torch.cuda.is_available())  
    else:
        dataloader = DataLoader(dataset,
                                batch_size = int(args.batch_size),
                                num_workers = 8, 
                                shuffle = (train == True),  
                                sampler = None,
                                drop_last = False,
                                pin_memory = torch.cuda.is_available())

    return dataloader


class GetClimateDataset(Dataset):
    '''Dataset for ERA5->ERA5 experiments'''
    def __init__(self, location, train, transform, upsampling_factor, noise_ratio, std, crop_size,n_patches,method):
        self.location = location
        self.upsampling_factor = upsampling_factor
        self.train = train
        self.noise_ratio = noise_ratio
        self.std = torch.Tensor(std).view(len(std),1,1)
        self.transform = transform
        self._get_files_stats()
        self.crop_size = crop_size #In this case, the crop_size is the size of the image crop from the HR ERA5 image
        self.n_patches = n_patches
        self.method = method
        self.crop_transform = transforms.RandomCrop(crop_size)
        # we will always crop the image into square patches
        if (self.train == True) and (method == "bicubic"):
            self.bicubicDown_transform = transforms.Resize((int(self.crop_size/upsampling_factor),int(self.crop_size/upsampling_factor)),Image.BICUBIC, antialias=True)  # subsampling the image (half size)
        elif (self.train == False) and (method == "bicubic"):
            self.bicubicDown_transform = transforms.Resize((int((self.img_shape_x-1)/upsampling_factor),int(self.img_shape_y/upsampling_factor)),Image.BICUBIC, antialias=True)  # subsampling the image (half size)

    def _get_files_stats(self):
        self.files_paths = glob.glob(self.location + "/*.h5")
        self.files_paths.sort()
        self.n_years = len(self.files_paths)
        with h5py.File(self.files_paths[0], 'r') as _f:
            print("Getting file stats from {}".format(self.files_paths[0]))
            self.n_samples_per_year = _f['fields'].shape[0]
            self.n_in_channels = _f['fields'].shape[1]
            self.img_shape_x = _f['fields'].shape[2]
            self.img_shape_y = _f['fields'].shape[3]

        self.n_samples_total = self.n_years * self.n_samples_per_year
        self.files = [None for _ in range(self.n_years)]
        print("Number of samples per year: {}".format(self.n_samples_per_year))
        print("Found data at path {}. Number of examples: {}. Image Shape: {} x {} x {}".format(self.location, self.n_samples_total, self.img_shape_x, self.img_shape_y, self.n_in_channels))

    def _open_file(self, year_idx):
        _file = h5py.File(self.files_paths[year_idx], 'r')
        self.files[year_idx] = _file['fields']  

    def __len__(self):
        if self.train == True: 
            return self.n_samples_total * self.n_patches
        return self.n_samples_total

    def __getitem__(self, global_idx):
        year_idx, local_idx = self.get_indices(global_idx)

        # Open image file if it's not already open
        if self.files[year_idx] is None:
            self._open_file(year_idx)

        # Apply transform and cut-off
        y = self.transform(self.files[year_idx][local_idx])
        y = y[:,:-1,:]

        # Modify y for training and get X based on method
        if self.train:
            y = self.crop_transform(y)
        X = self.get_X(y)

        return X, y

    def get_indices(self, global_idx):
        if self.train:
            year_idx = int(global_idx/(self.n_samples_per_year*self.n_patches))  # which year we are on
            local_idx = int((global_idx//self.n_patches) % self.n_samples_per_year)  # which sample in that year we are on 
        else:
            year_idx = int(global_idx/self.n_samples_per_year)  # which year we are on
            local_idx = int(global_idx % self.n_samples_per_year)  # which sample in that year we are on 

        return year_idx, local_idx

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


class GetWTKDataset(Dataset):
    '''Dataset for ERA5->WTK experiments'''
    def __init__(self, location, data_tag,  train, transform, crop_size, upsampling_factor):
        self.location = location
        self.train = train
        self.transform = transform
        self.crop_size=crop_size # In this case, the crop_size is the size of the image crop from the LR (ERA5) image, the HR (WTK) crop would be of size: crop_size*upsampling_factor
        self.upsampling_factor = upsampling_factor
        self.data_tag = data_tag
        self.regions = []
        self.region_samples = {}
        self._get_files_stats()
       
    def _get_files_stats(self):
        
        for region in sorted(os.listdir(self.location)):
            if self.train:
                region_path = os.path.join(self.location, region)
            else:
                region = os.path.basename(os.path.normpath(self.location))
                region_path = self.location
                
            self.regions.append(region)
            self.region_samples[region]={"lr":[], "hr":[]}
            
            u_hr_file_path, v_hr_file_path = os.path.join(region_path+self.data_tag, "hr_u_10m.pkl"), os.path.join(region_path+self.data_tag, "hr_v_10m.pkl")
            print(u_hr_file_path, v_hr_file_path)
            with open(u_hr_file_path, 'rb') as file:
                u_hr = pickle.load(file)
            with open(v_hr_file_path, 'rb') as file:
                v_hr = pickle.load(file)
            
            u_lr_file_path, v_lr_file_path = os.path.join(region_path+self.data_tag, "lr_u_10m.pkl"), os.path.join(region_path+self.data_tag, "lr_v_10m.pkl")
            with open(u_lr_file_path, 'rb') as file:
                u_lr = pickle.load(file)
            with open(v_lr_file_path, 'rb') as file:
                v_lr = pickle.load(file)

            ## Add the entire test data of full-size u,v LR and HR for each region in the region_samples dictionary 
            self.region_samples[region]["lr"].append(u_lr)
            self.region_samples[region]["lr"].append(v_lr)
            self.region_samples[region]["hr"].append(u_hr)
            self.region_samples[region]["hr"].append(v_hr)

            if self.train is False:
                break

        print("Getting hres, lres data stats from {}".format(region))
        hr = self.region_samples[region]["hr"][0]
        lr = self.region_samples[region]["lr"][0]
        self.height_hres_for_this_region = hr.shape[1]
        self.width_hres_for_this_region = hr.shape[2]
        print("hr image shape for this region: ", self.height_hres_for_this_region, self.width_hres_for_this_region)
        assert lr.shape[0] ==  hr.shape[0]
        height_lres_for_this_region = lr.shape[1]
        width_lres_for_this_region = lr.shape[2]
        print("lr image shape for this region: ", height_lres_for_this_region, width_lres_for_this_region)
      
        self.total_samples_per_region = hr.shape[0]
        self.number_of_regions = len(self.regions)
        print("Found data at path {}. Number regions: {}. Number of examples per region: {}.".format(self.location, self.number_of_regions, self.total_samples_per_region)) 

    def uniform_box_sampler(self, data_shape_0, data_shape_1):
        ## Borrowed from https://github.com/NREL/sup3r/blob/85637d4fb568e59719789e23b7a11f4b1cd05b92/sup3r/utilities/utilities.py#L278
        ## This returns a slice or cut to create cropped samples from an image tile
        shape_1 = data_shape_0 if data_shape_0 < self.crop_size else self.crop_size
        shape_2 = data_shape_1 if data_shape_1 < self.crop_size else self.crop_size
        shape = (shape_1, shape_2)
        start_row = np.random.randint(0, data_shape_0 - shape_1 + 1)
        start_col = np.random.randint(0,data_shape_1 - shape_2 + 1)
        stop_row = start_row + shape[0]
        stop_col = start_col + shape[1]

        return [slice(start_row, stop_row), slice(start_col, stop_col)]
    
    def __len__(self):
        return self.number_of_regions*self.total_samples_per_region
    
    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            region_idx, sample_idx = idx
            return self.get_region_item(region_idx, sample_idx)
        else:
            return self.get_region_item(0, idx)


    def get_region_item(self, region_idx, sample_idx):
        full_u_lr = self.region_samples[self.regions[region_idx]]["lr"][0]
        full_u_hr = self.region_samples[self.regions[region_idx]]["hr"][0]
        full_v_lr = self.region_samples[self.regions[region_idx]]["lr"][1]
        full_v_hr = self.region_samples[self.regions[region_idx]]["hr"][1]

        ## Select the u,v LR for the particular time step from the test year
        u_lr, v_lr = full_u_lr[sample_idx], full_v_lr[sample_idx]
        if self.train:
            ## Get a sample by randomly slicing (a size of crop_size) from the LR image tile 
            lr_slice_rows, lr_slice_cols = self.uniform_box_sampler(u_lr.shape[0], u_lr.shape[1])
            u_lr, v_lr = u_lr[lr_slice_rows, lr_slice_cols], v_lr[lr_slice_rows, lr_slice_cols]
        # u,v are concatenated in the final LR used for train/val/test
        lr = np.stack((u_lr, v_lr), axis=0) 

        ## Select the u,v HR for the particular time step from the test year
        u_hr, v_hr = full_u_hr[sample_idx], full_v_hr[sample_idx]
        if self.train:
            ## Get the corresponding sample of crop_size*upsampling_factor from the HR image tile 
            new_start_row = lr_slice_rows.start * self.upsampling_factor
            new_stop_row = lr_slice_rows.stop * self.upsampling_factor 
            new_start_col = lr_slice_cols.start * self.upsampling_factor
            new_stop_col = lr_slice_cols.stop * self.upsampling_factor 

            hr_slice_rows, hr_slice_cols = slice(new_start_row, new_stop_row), slice(new_start_col, new_stop_col)
            u_hr, v_hr = u_hr[hr_slice_rows, hr_slice_cols], v_hr[hr_slice_rows, hr_slice_cols]
        # u,v are concatenated in the final HR used for train/val/test
        hr = np.stack((u_hr, v_hr), axis=0) 

        if self.transform:
            lr = self.transform(lr)
            hr = self.transform(hr)

        return lr,hr
    

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
     








