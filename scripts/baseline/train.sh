#!/bin/bash
## Configure for your setup

export CUDA_VISIBLE_DEVICES=0
# set the `PYTHONPATH` to the root directory of this project

# For the ERA5→ERA5 experiments, all baseline models are trained twice:
# once with an upsampling factor of 8 (used for standard downscaling), 
# and once with a upsampling factor of 4 (used for zero-shot downscaling).
# For the ERA5→WTK experiments, all baseline models are trained once with an upsampling factor of 5.
# See the paper for more details.

# SRCNN (with the optimal hyperparameters shown below) 
# ERA5->ERA5
srun python scripts/baseline/train.py \
    --method 'bicubic' \
    --upsampling_factor 4 \
    --model SRCNN \
    --data_name era5 \
    --data_path '../datasets/era5' \
    --in_channels 3 \
    --out_channels 3 \
    --crop_size 64 \
    --n_patches 8 \
    --lr 0.001 \
    --batch_size 32 \
    --epochs 400 \
    --loss_type l2 \
    --optimizer_type Adam \
    --wd 1e-05 \
# # change upsampling factor to 8 and crop size to 128 for the standard downscaling ERA5->ERA5 experiments 

# ERA5->WTK 
srun python scripts/baseline/train.py \
    --upsampling_factor 5 \
    --model SRCNN \
    --data_name wtk \
    --data_path '../datasets/era_to_wtk' \
    --in_channels 2 \
    --out_channels 2 \
    --crop_size 32 \
    --lr 0.0001 \
    --batch_size 32 \
    --epochs 400 \
    --loss_type l2 \
    --optimizer_type Adam \
    --wd 1e-05 \


# EDSR (with the optimal hyperparameters shown below) 
# ERA5->ERA5
srun python scripts/baseline/train.py \
    --method 'bicubic' \
    --upsampling_factor 4 \
    --model EDSR \
    --data_name era5 \
    --data_path '../datasets/era5' \
    --in_channels 3 \
    --out_channels 3 \
    --crop_size 64 \
    --n_patches 8 \
    --lr 0.0001 \
    --batch_size 64 \
    --epochs 400 \
    --loss_type l1 \
    --optimizer_type Adam \
    --wd 1e-05 \
    --n_res_block 16 \
    --hidden_channels 64  
# change upsampling factor to 8 and crop size to 128 for the standard downscaling ERA5->ERA5 experiments 

# ERA5->WTK 
srun python scripts/baseline/train.py \
    --method 'bicubic' \
    --upsampling_factor 5 \
    --model EDSR \
    --data_name wtk \
    --data_path '../datasets/era_to_wtk' \
    --in_channels 2 \
    --out_channels 2 \
    --crop_size 32 \
    --lr 0.001 \
    --batch_size 64 \
    --epochs 400 \
    --loss_type l1 \
    --optimizer_type Adam \
    --wd 1e-05 \
    --n_res_block 16 \
    --hidden_channels 64  


# SWINIR (with the optimal hyperparameters shown below) 
# ERA5->ERA5
srun python scripts/baseline/train.py \
    --method 'bicubic' \
    --upsampling_factor 4 \
    --model SwinIR \
    --data_name era5 \
    --data_path '../datasets/era5' \
    --in_channels 3 \
    --out_channels 3 \
    --crop_size 64 \
    --n_patches 8 \
    --lr 0.0001 \
    --batch_size 32 \
    --epochs 400 \
    --loss_type l1 \
    --optimizer_type AdamW \
    --wd 1e-05 \
# change upsampling factor to 8 and crop size to 128 for the standard downscaling ERA5->ERA5 experiments 

# ERA5->WTK 
srun python scripts/baseline/train.py \
    --upsampling_factor 5 \
    --model SwinIR \
    --data_name wtk \
    --data_path '../datasets/era_to_wtk' \
    --in_channels 2 \
    --out_channels 2 \
    --crop_size 32 \
    --lr 0.0001 \
    --batch_size 32 \
    --epochs 400 \
    --loss_type l1 \
    --optimizer_type AdamW \
    --wd 1e-05 \


# ESRGAN (with the optimal hyperparameters shown below, refer to ESRGAN'S config for more hyperparameters - make sure its updated) 
# ERA5->ERA5
srun python scripts/baseline/train.py \
    --method 'bicubic' \
    --upsampling_factor 4 \
    --model ESRGAN \
    --data_name era5 \
    --data_path '../datasets/era5' \
    --crop_size 64 \
    --n_patches 8 \
    --batch_size 32
# change upsampling factor to 8 and crop size to 128 for the standard downscaling ERA5->ERA5 experiments 

# ERA5->WTK 
srun python scripts/baseline/train.py \
    --upsampling_factor 5 \
    --model ESRGAN \
    --data_name wtk \
    --data_path '../datasets/era_to_wtk' \
    --crop_size 32 \
    --batch_size 32
