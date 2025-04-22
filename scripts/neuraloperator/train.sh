#!/bin/bash
## Configure for your setup

export CUDA_VISIBLE_DEVICES=0
# set the `PYTHONPATH` to the root directory of this project

# For the ERA5→ERA5 experiments, all Neural Operator models are trained twice:
# once with an upsampling factor of 8 (used for standard downscaling), 
# and once with a upsampling factor of 4 (used for zero-shot downscaling).
# For the ERA5→WTK experiments, all baseline models are trained once with an upsampling factor of 5.
# See the paper for more details.

# Refer to config file for more hyperparameters, make sure the config file src/neuraloperators/config/config.yaml is updated

# ERA5->ERA5
srun python scripts/neuraloperator/train.py \
    --method 'bicubic' \
    --upsampling_factor 4 \
    --data_name era5 \
    --data_path '../datasets/era5' \
    --crop_size 64 \
    --n_patches 8 \
    --batch_size 32 
# change upsampling factor to 8 and crop size to 128 for the standard downscaling ERA5->ERA5 experiments 

# ERA5->WTK 
srun python scripts/neuraloperator/train.py \
    --upsampling_factor 5 \
    --data_name wtk \
    --data_path '../datasets/era_to_wtk' \
    --crop_size 32 \
    --batch_size 32 \