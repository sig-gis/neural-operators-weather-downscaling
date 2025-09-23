'''training baselines'''
'''the scripts/baseline and src/baseline_models directories are based on https://github.com/erichson/SuperBench'''

import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from utils import *

from src.baseline_models import EDSR, SRCNN, Bicubic, SwinIR

from src.baseline_models.ESRGAN import train as esrgan_train
from src.baseline_models.ESRGAN.config import config as esrgan_config
from src.data_loading_extra import get_data_loader


# train the model with the given parameters and save the model with the best validation error
def train(args, train_loader, val_loader, model, optimizer, criterion, project_dir):
    best_val = np.inf
    train_loss_list, val_error_list = [], []
    start2 = time.time()
    for epoch in range(args.epochs):
        start = time.time()
        train_loss_total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            # [b,c,h,w]
            data, target = data.to(args.device).float(), target.to(args.device).float()

            # forward
            model.train()
            output = model(data) 
            loss = criterion(output, target)
            train_loss_total += loss.item()
            
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()

        # record train loss
        train_loss_mean = train_loss_total / len(train_loader)
        train_loss_list.append(train_loss_mean)

        # validate
        mse = validate(args, val_loader, model, criterion)
        print("epoch: %s, val error (interp): %.10f" % (epoch, mse))      
        val_error_list.append(mse)

        if mse <= best_val:
            best_val = mse
            save_checkpoint(model, os.path.join(project_dir,'results/model_' + str(args.model) + '_' + str(args.data_name) + '_' + str(args.upsampling_factor) + '_' + str(args.lr) + '_' + str(args.method) +'_' + str(args.seed) + '.pt'))
        end = time.time()
        print('The epoch time is: ', (end - start))
    end2 = time.time()
    print('The training time is: ', (end2 - start2))

    return train_loss_list, val_error_list

# validate the model 
def validate(args, val_loader, model, criterion):
    mse = 0
    c = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.to(args.device).float(), target.to(args.device).float()
            output = model(data) 
            mse += criterion(output, target) * data.shape[0]
            c += data.shape[0]
    mse /= c

    return mse.item()


def main():
    parser = argparse.ArgumentParser(description='training parameters')
    # Arguments for data
    parser.add_argument('--data_name', type=str, default="era5", help='dataset')
    parser.add_argument('--data_path', type=str, default='../datasets/era5', help='the folder path of dataset')
    parser.add_argument('--crop_size', type=int, default=128, help='crop size')
    parser.add_argument('--n_patches', type=int, default=8, help='number of patches')    
    parser.add_argument('--method', type=str, default="bicubic", help='downsample method') #for ERA5 experiments : the LR images are created by coarsening the HR patches with bicubic interpolation.
    parser.add_argument('--model_path', type=str, default='../results/model_EDSR_era5_8_0.0001_bicubic_5544.pt', help='saved model')
    parser.add_argument('--pretrained', default=False, type=lambda x: (str(x).lower() == 'true'), help='load the pretrained model')
    parser.add_argument('--noise_ratio', type=float, default=0.0, help='noise ratio') #this isn't needed as we do bicubic downsampling (to derive LR from HR) in our ERA5 experiments
    parser.add_argument('--max_samples',type=int, default=1000,help='maximum number of patches to use for trianing')
    parser.add_argument('--max_val_samples',type=int, default=32,help='maximum number of patches to use for validation')
    
    # Arguments for model and training
    parser.add_argument('--model', type=str, default='EDSR', help='model')
    parser.add_argument('--epochs', type=int, default=400, help='max epochs')
    parser.add_argument('--device', type=str, default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), help='computing device')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--wd', type=float, default=1e-6, help='weight decay')
    parser.add_argument('--seed', type=int, default=5544, help='random seed')
    parser.add_argument('--step_size', type=int, default=100, help='step size for scheduler')
    parser.add_argument('--gamma', type=float, default=0.97, help='coefficient for scheduler')

    # Arguments for model
    parser.add_argument('--upsampling_factor', type=int, default=8, help='upsampling factor')
    parser.add_argument('--in_channels', type=int, default=3, help='num of input channels')
    parser.add_argument('--hidden_channels', type=int, default=32, help='num of hidden channels')
    parser.add_argument('--out_channels', type=int, default=3, help='num of output channels')
    parser.add_argument('--n_res_blocks', type=int, default=18, help='num of resdiual blocks')
    parser.add_argument('--loss_type', type=str, default='l1', help='L1 or L2 loss')
    parser.add_argument('--optimizer_type', type=str, default='Adam', help='type of optimizer')
    parser.add_argument('--scheduler_type', type=str, default='ExponentialLR', help='type of scheduler')


    parser.add_argument('--project_dir',type=str,default='./')
    args = parser.parse_args()
    print(args)

    ## Get project directory path
    project_dir = args.project_dir

    # Set random seed 
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Load data
    train_loader, val_loader, mean, std = get_data_loader(args,args.data_name,True,args.n_patches)

    # Train the models
    # Some hyper-parameters for SwinIR
    upsample = args.upsampling_factor
    window_size = 8
    crop_size = args.crop_size if args.data_name == 'era5' else args.crop_size * upsample
    height = (crop_size // upsample // window_size + 1) * window_size 
    width = (crop_size // upsample // window_size + 1) * window_size

    if args.model=="ESRGAN":
        trainer = esrgan_train.Trainer(esrgan_config, train_loader, val_loader, mean,std, args.device, args.batch_size, args.upsampling_factor)
        # Model summary
        print('**** Setup ****')
        print('Total params Generator: %.3fM' % (sum(p.numel() for p in trainer.generator.parameters())/1000000.0))
        print('************')  
        generator_loss_list, discriminator_loss_list = trainer.train()

        ## Use the generator and discriminator losses to plot loss curves if needed (uncomment the following lines)
        # x_axis = np.arange(0, len(generator_loss_list))
        # plt.figure()
        # plt.plot(x_axis, generator_loss_list, label = 'generator train loss')
        # plt.yscale('log')
        # plt.legend()
        # plt.savefig(os.path.join(project_dir,'figures/generator_train_loss_' + str(args.model) + '_' + str(args.data_name) + '_' + str(args.upsampling_factor) + '_' + str(args.lr) + '_' + str(args.method) +'_' + str(args.seed) + '.png'), dpi = 300)

        # plt.figure()
        # plt.plot(x_axis, discriminator_loss_list, label = 'discriminator train error')
        # plt.yscale('log')
        # plt.legend()
        # plt.savefig(os.path.join(project_dir,'figures/discrimiator_train_loss_' + str(args.model) + '_' + str(args.data_name) + '_' + str(args.upsampling_factor) + '_' + str(args.lr) + '_' + str(args.method) +'_' + str(args.seed) + '.png'), dpi = 300)


    else:
        model_list = {
                'SRCNN': SRCNN(args.in_channels, args.upsampling_factor, mean, std),
                'EDSR': EDSR(args.in_channels, args.hidden_channels, args.n_res_blocks, args.upsampling_factor, mean, std),
                'SwinIR': SwinIR(upscale=args.upsampling_factor, in_chans=args.in_channels, img_size=(height, width),
                        window_size=window_size, img_range=1., depths=[6, 6, 6, 6, 6, 6],
                        embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv',mean =mean,std=std),
        }

        model = model_list[args.model].to(args.device)
        
        # If pretrain and posttune
        if args.pretrained == True:
            model = load_checkpoint(model, args.model_path)
            model = model.to(args.device)

        # Model summary
        print(model)    
        print('**** Setup ****')
        print('Total params Generator: %.3fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
        print('************')    

        # Set optimizer, loss function and Learning Rate Scheduler
        optimizer = set_optimizer(args, model)
        scheduler = set_scheduler(args, optimizer, train_loader)
        criterion = loss_function(args)

        # Training and validation
        train_loss_list, val_error_list = train(args, train_loader, val_loader, model, optimizer, criterion, project_dir)

        # Plot train and val loss curves
        x_axis = np.arange(0, args.epochs)
        plt.figure()
        plt.plot(x_axis, train_loss_list, label = 'train loss')
        plt.yscale('log')
        plt.legend()
        plt.savefig(os.path.join(project_dir,'figures/train_loss_' + str(args.model) + '_' + str(args.data_name) + '_' + str(args.upsampling_factor) + '_' + str(args.lr) + '_' + str(args.method) +'_' + str(args.seed) + '.png'), dpi = 300)

        plt.figure()
        plt.plot(x_axis, val_error_list, label = 'val error')
        plt.yscale('log')
        plt.legend()
        plt.savefig(os.path.join(project_dir,'figures/val_error_' + str(args.model) + '_' + str(args.data_name) + '_' + str(args.upsampling_factor) + '_' + str(args.lr) + '_' + str(args.method) +'_' + str(args.seed) + '.png'), dpi = 300)

if __name__ =='__main__':
    main()
