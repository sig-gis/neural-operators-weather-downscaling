import os

# Define your arguments as key-value pairs
num_epoch = 400  # number of epochs to train for
epoch = 0  # epochs in current train
## Get current directory path
model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")
checkpoint_dir = os.path.join(model_dir,'checkpoints/wtk_5x_bs32_lr0.0001/')  # path to saved model

in_channels = 2 #2 channels for ERA5->WTK and 3 for ERA5->ERA5 experiments
out_channels = 2 #2 channels for ERA5->WTK and 3 for ERA5->ERA5 experiments
nf = 32  # number of filter in esrgan
b1 = 0.9  # coefficients used for computing running averages of gradient and its square
b2 = 0.999  # coefficients used for computing running averages of gradient and its square
weight_decay = 1e-2  # weight decay
lr = 0.0001 

residual_blocks = 23
warmup_batches = 500
lambda_adv = 0.1
lambda_pixel = 1
lambda_content = 1
#The height, width of the downscaled generated output that goes as an input to the discriminator while training
hr_width = 160 #160 for ERA5->WTK training; 128 for ERA5->ERA5 standard downscaling training and 64 for ERA5->ERA5 zero-shot downscaling training
hr_height = 160 #160 for ERA5->WTK training; 128 for ERA5->ERA5 standard downscaling training and 64 for ERA5->ERA5 zero-shot downscaling training
