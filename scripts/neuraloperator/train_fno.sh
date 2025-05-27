export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PWD/"


torchrun scripts/neuraloperator/train.py \
    --method 'bicubic' \
    --upsampling_factor 4 \
    --data_name era5 \
    --data_path './datasets/era5' \
    --crop_size 64 \
    --n_patches 8 \
    --batch_size 32 