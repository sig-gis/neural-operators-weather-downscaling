#!/bin/bash
## Configure for your setup

export CUDA_VISIBLE_DEVICES=0
# set the `PYTHONPATH` to the root directory of this project

# For ERA5->ERA5 experiments:
# Standard downscaling evaluation: evaluate the model trained on 8x upsampling factor to produce outputs at the same upsampling factor of 8x.
# Zero-shot downscaling evaluation: evaluate the model trained on 4x upsampling factor to produce outputs at a higher upsampling factor of 8x.

# For ERA5->WTK experiments:
# Standard downscaling evaluation: evaluate the model trained on 5x upsampling factor to produce outputs at the same upsampling factor of 5x.
# Zero-shot downscaling evaluation: evaluate the model trained on 5x upsampling factor to produce outputs at a higher upsampling factor of 15x.

# Following scripts perform zero-shot evaluation, in order to perform standard evaluation remove the zero_shot and zero_shot_upsampling_factor arguements.
# Additonally, for ERA5->ERA5 standard downscaling evaluation: change upsampling_factor to 8 and crop_size to 128.
# For ERA5->WTK standard downscaling evaluation: use the correct test dataloader, modify in src/data_loading.py.
    
# SRCNN    
# ERA5->ERA5
srun python scripts/baseline/eval.py \
    --model SRCNN \
    --data_name era5 \
    --crop_size 64 \
    --n_patches 8 \
    --method 'bicubic' \
    --upsampling_factor 4 \
    --batch_size 4 \
    --data_path '../datasets/era5/' \
    --in_channels 3 \
    --out_channels 3 \
    --lr 0.001 \
    --epochs 400 \
    --loss_type l2 \
    --optimizer_type Adam \
    --wd 1e-05 \
    --zero_shot \
    --zero_shot_upsampling_factor 8

# ERA5->WTK 
srun python scripts/baseline/eval.py \
    --model SRCNN \
    --data_name wtk \
    --upsampling_factor 5 \
    --batch_size 4 \
    --data_path '../datasets/era_to_wtk/' \
    --in_channels 2 \
    --out_channels 2 \
    --crop_size 32 \
    --lr 0.0001 \
    --epochs 400 \
    --loss_type l2 \
    --optimizer_type Adam \
    --wd 1e-05 \
    --zero_shot \
    --zero_shot_upsampling_factor 15


# EDSR    
# ERA5->ERA5
srun python scripts/baseline/eval.py \
    --model EDSR \
    --data_name era5 \
    --crop_size 64 \
    --n_patches 8 \
    --method 'bicubic' \
    --upsampling_factor 4 \
    --batch_size 4 \
    --data_path '../datasets/era5/' \
    --in_channels 3 \
    --out_channels 3 \
    --lr 0.0001 \
    --epochs 400 \
    --loss_type l1 \
    --optimizer_type Adam \
    --wd 1e-05 \
    --n_res_block 16 \
    --hidden_channels 64  \
    --zero_shot \
    --zero_shot_upsampling_factor 8

#ERA5->WTK
srun python scripts/baseline/eval.py \
    --model EDSR \
    --data_name wtk \
    --upsampling_factor 5 \
    --batch_size 4 \
    --data_path '../datasets/era_to_wtk/' \
    --in_channels 2 \
    --out_channels 2 \
    --crop_size 32 \
    --lr 0.001 \
    --epochs 400 \
    --loss_type l1 \
    --optimizer_type Adam \
    --wd 1e-05 \
    --n_res_block 16 \
    --hidden_channels 64  \
    --zero_shot \
    --zero_shot_upsampling_factor 15


# SWINIR
# ERA5->ERA5
srun python scripts/baseline/eval.py \
    --model SwinIR \
    --data_name era5 \
    --crop_size 64 \
    --n_patches 8 \
    --method 'bicubic' \
    --upsampling_factor 4 \
    --batch_size 4 \
    --data_path '../datasets/era5' \
    --in_channels 3 \
    --out_channels 3 \
    --lr 0.0001 \
    --epochs 400 \
    --loss_type l1 \
    --optimizer_type AdamW \
    --wd 1e-05 \
    --zero_shot \
    --zero_shot_upsampling_factor 8

# ERA5->WTK
srun python scripts/baseline/eval.py \
    --model SwinIR \
    --data_name wtk \
    --crop_size 32 \
    --upsampling_factor 5 \
    --batch_size 4 \
    --data_path '../datasets/era_to_wtk/' \
    --in_channels 2 \
    --out_channels 2 \
    --lr 0.0001 \
    --epochs 400 \
    --loss_type l1 \
    --optimizer_type AdamW \
    --wd 1e-05 \
    --zero_shot \
    --zero_shot_upsampling_factor 15


# ESRGAN (make sure ESRGAN's config is updated as needed)
# ERA5->ERA5
srun python scripts/baseline/eval.py \
    --model ESRGAN \
    --data_name era5 \
    --crop_size 64 \
    --n_patches 8 \
    --method 'bicubic' \
    --upsampling_factor 4 \
    --batch_size 4 \
    --data_path '../datasets/era5' \
    --in_channels 3 \
    --out_channels 3 \
    --zero_shot \
    --zero_shot_upsampling_factor 8

# ERA5->WTK 
srun python scripts/baseline/eval.py \
    --model ESRGAN \
    --data_name wtk \
    --crop_size 32 \
    --n_patches 8 \
    --upsampling_factor 5 \
    --batch_size 4 \
    --data_path '../datasets/era_to_wtk/' \
    --in_channels 2 \
    --out_channels 2 \
    --zero_shot \
    --zero_shot_upsampling_factor 15

# Bicubic -- for the bicubic interpolation method, we do not train a model, 
# we directly evaluate setting the upsampling_factor to the desired value for standard or zero-shot downscaling evaluation.
# ERA5->ERA5
srun python scripts/baseline/eval.py \
    --model bicubic \
    --data_name era5 \
    --upsampling_factor 8 \
    --batch_size 4 \
    --method 'bicubic' \
    --data_path '../datasets/era5/' \
    --out_channels 3 

# ERA5->WTK 
srun python scripts/baseline/eval.py \
    --model bicubic \
    --data_name wtk \
    --upsampling_factor 15 \
    --batch_size 4 \
    --data_path '../datasets/era_to_wtk/' 