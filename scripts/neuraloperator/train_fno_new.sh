export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PWD/"

conda activate downscaling

torchrun scripts/neuraloperator/train_new_dataset.py \
    --method 'bicubic' \
    --upsampling_factor 4 \
    --data_name era5 \
    --data_path './datasets/era5_new' \
    --crop_size 64 \
    --n_patches 120 \
    --batch_size 4  