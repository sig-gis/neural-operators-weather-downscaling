export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PWD/"

conda activate downscaling

python scripts/baseline/train_new_dset.py \
    --method 'bicubic' \
    --upsampling_factor 4 \
    --model SRCNN \
    --data_name era5 \
    --data_path './datasets/era5_new' \
    --in_channels 3 \
    --out_channels 3 \
    --crop_size 64 \
    --n_patches 8 \
    --lr 0.001 \
    --batch_size 8 \
    --epochs 400 \
    --loss_type l2 \
    --optimizer_type Adam \
    --wd 1e-05 \
    --max_samples 1000 \
    --max_val_samples 100