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

# Make sure the config file src/neuraloperators/config/config.yaml is updated

# ERA5->ERA5
srun python scripts/neuraloperator/eval.py \
    --data_name era5 \
    --crop_size 64 \
    --n_patches 8 \
    --method 'bicubic' \
    --upsampling_factor 4 \
    --batch_size 4 \
    --data_path '../datasets/era5' \
    --zero_shot \
    --zero_shot_upsampling_factor 8

# ERA5->WTK
srun python scripts/neuraloperator/eval.py \
    --data_name wtk \
    --crop_size 32 \
    --upsampling_factor 5 \
    --batch_size 4 \
    --data_path '../datasets/era_to_wtk/' \
    --zero_shot \
    --zero_shot_upsampling_factor 15
