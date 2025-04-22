'''
Visualization functions are based on  https://github.com/erichson/SuperBench
'''

import argparse
import os
import sys
from decimal import Decimal

import cmocean
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from configmypy import ArgparseConfig, ConfigPipeline, YamlConfig
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from scripts.baseline.utils import *
from src.baseline_models import *
from src.baseline_models.ESRGAN import test_utils as esrgan_test_utils
from src.baseline_models.ESRGAN.config import config as esrgan_config
from src.data_loading import getTestData
from src.neuraloperators.neuralop import get_model
from src.neuraloperators.neuralop.training import setup


def load_model(args, project_dir, data_name, model_name, upsampling_factor):
    '''load model and test dataset'''
   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    in_channels = args.in_channels
    out_channels = args.out_channels
    data_path = args.data_path # One of the regions is chosen to visualize for the ERA5->WTK dataset

    baseline_models = ['SRCNN', 'ESRGAN', 'EDSR', 'SwinIR']

    # Following bit is borrowed from the eval script
    resol, n_fields, mean, std = get_data_info(data_name)
    window_size = 8
    crop_size = args.crop_size if args.data_name == 'era5' else args.crop_size * upsampling_factor
    height = (crop_size // upsampling_factor // window_size + 1) * window_size 
    width = (crop_size // upsampling_factor // window_size + 1) * window_size
    model_list = {
            'SRCNN': SRCNN(in_channels, upsampling_factor,mean,std),
            'EDSR': EDSR(in_channels, 64, 16, upsampling_factor, mean, std),
            'SwinIR': SwinIR(upscale=upsampling_factor, in_chans=in_channels, img_size=(height, width),
                    window_size=window_size, img_range=1., depths=[6, 6, 6, 6, 6, 6],
                    embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv',mean =mean,std=std),
    }
    
    if model_name == 'bicubic':
        model=Bicubic(upsampling_factor=args.upsampling_factor) # args.upsampling_factor is the zero-shot upsampling factor in case of zero-shot
    elif model_name in baseline_models: 
        if model_name == "ESRGAN":
            ## Make sure the ESRGAN's config file is updated for the current setup
            tester = esrgan_test_utils.Tester(esrgan_config, mean, std, device, upsampling_factor) 
            model = tester.build_model()

        else:
            model = model_list[model_name]
            if data_name == "wtk":
                lr = 0.001 if model_name == "EDSR" else 0.0001
            else:
                lr = 0.001 if model_name == "SRCNN" else 0.0001
            model_path = os.path.join(project_dir,'results/model_'  + str(model_name) + '_' + str(args.data_name) + '_' + str(upsampling_factor) + '_' + str(lr) + '_' + str(args.method) +'_' + str(args.seed)  + '.pt')
            
            model = load_checkpoint(model, model_path)
            model = model.to(device)  
    else:
        ## Make sure the Neural Operator config file is updated for the current setup
        pipe = ConfigPipeline(
            [
                YamlConfig(
                "./config.yaml", config_name="default", config_folder=os.path.join(project_dir,'src/neuraloperators/config')
            ),
            YamlConfig(config_folder=os.path.join(project_dir,'src/neuraloperators/config')),
            ]
        )
        config = pipe.read_conf()
        if model_name == "FNO":
            model_name = 'TFNO2d'
        config["arch"] = model_name.lower() 
        arch = config["arch"]
        config_arch = config.get(arch) 

        # Set-up distributed communication, if using
        device, is_logger = setup(config)

        # Make sure we only print information when needed
        config.verbose = config.verbose and is_logger

        # Print config to screen
        if config.verbose and is_logger:
            pipe.log()
            sys.stdout.flush()

        model = get_model(config, mean, std, args.upsampling_factor)
        model = model.to(device)

        # Use distributed data parallel
        if config.distributed.use_distributed:
            model = DDP(
                model, device_ids=[device.index], output_device=device.index, static_graph=True
            )

        checkpoint_path = os.path.join(project_dir,'src/neuraloperators',config_arch["checkpoint_path"])
        if config.checkpoint.save_checkpoint is True:
            model.load_state_dict(torch.load(os.path.join(checkpoint_path, 'best_model.pt')))


    return model


def get_one_image(model, testloader, snapshot_num, channel_num, zero_shot):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    with torch.no_grad():
        for index, (data,target) in enumerate(testloader):
            data, target = data.to(device).float(), target.to(device).float()
            if index == snapshot_num:
                output = model(data)
                # Refer to the eval scripts to see why this intepolation is done
                if zero_shot or output.shape[2]!=target.shape[2] or output.shape[3]!=target.shape[3]: 
                    output = F.interpolate(output, size=(target.shape[2], target.shape[3]), mode='bicubic', align_corners=False)
                break
    data = data[:,channel_num,...].squeeze().detach().cpu().numpy()
    target = target[:,channel_num,...].squeeze().detach().cpu().numpy()
    output = output[:,channel_num,...].squeeze().detach().cpu().numpy()
    return data, target, output


def get_lim(args, project_dir, data_name, model_name_list, upsampling_factor, zero_shot, snapshot_num, channel_num):
    '''
    Get the min and max of all the snapshots. The goal is to make the plotted snapshots have the same colorbar range. 
    '''
    # get data
    resol, n_fields,  mean, std = get_data_info(data_name)
    data = getTestData(args, std)
 
    i = 2 
    data_list = [] # This list contains the model outputs for all the models including the LR and HR

    for name in model_name_list:
        print(name)
        model = load_model(args, project_dir, data_name=data_name, model_name=name, upsampling_factor=upsampling_factor)
        LR, HR, data_item = get_one_image(model, data, snapshot_num, channel_num, zero_shot)
        if i == 2:
            if data_name=="wtk":
                # Compute windspeed
                u_LR = LR[0,:,:]
                v_LR = LR[1,:,:]
                LR = np.sqrt(u_LR**2 + v_LR**2)
                u_HR = HR[0,:,:]
                v_HR = HR[1,:,:]
                HR = np.sqrt(u_HR**2 + v_HR**2)
           
            data_list.append(LR)
            data_list.append(HR)
        
        if data_name=="wtk":
            u = data_item[0,:,:]
            v = data_item[1,:,:]
            data_item = np.sqrt(u**2 + v**2)
        data_list.append(data_item)
        i += 1

    min_lim_list, max_lim_list = [], []
    for i in range(len(data_list)):
        min_lim_list.append(np.min(data_list[i]))
        max_lim_list.append(np.max(data_list[i]))

    min_lim = float(np.min(np.array(min_lim_list)))
    max_lim = float(np.max(np.array(max_lim_list)))    

    min_lim_round = float(Decimal(min_lim).quantize(Decimal("0.1"), rounding = "ROUND_HALF_UP"))
    max_lim_round = float(Decimal(max_lim).quantize(Decimal("0.1"), rounding = "ROUND_HALF_UP"))    

    lim = []
    if (min_lim_round - min_lim) > 0:
        lim.append(min_lim_round - 0.1)
    else:
        lim.append(min_lim_round)

    if (max_lim_round - max_lim) < 0:
        lim.append(max_lim_round+0.1)
    else:
        lim.append(max_lim_round)

    return lim, data_list 


def plot_all_images(
    args,
    project_dir = "../",
    snapshot_num=55,
    channel_num = 0,
    zoom_loc_x = (820, 900),
    zoom_loc_y = (230, 150),
    figsize=(11,6),
    cmap='coolwarm',
    model_name_list= ['bicubic', 'ESRGAN', 'SwinIR', 'DFNO', 'DUNO', 'DCNO', 'DAFNO']):

    title_list = ['LR','HR']+model_name_list

    data_name = args.data_name
    upsample= args.upsampling_factor
    zero_shot = args.zero_shot
    # Adjust the upsampling factor if needed
    if zero_shot:
        args.upsampling_factor = args.zero_shot_upsampling_factor  

    # Get data and data range
    [vmin, vmax], data_list = get_lim(args, project_dir, data_name, model_name_list, upsample, zero_shot, snapshot_num, channel_num)
    vmean = (vmin + vmax) / 2.0
    vmean = float(Decimal(vmean).quantize(Decimal("0.1"), rounding = "ROUND_HALF_UP")) 
    print('The consistent min, mean and max are: ', vmin, vmean, vmax)

    # Setup the figure definition
    fc = "none"
    font_size = 20
    label_size = 12 
    ec = "0.3" 
    box_color = 'k'
    plt.clf()

    fig, axs = plt.subplots(
    nrows=3,  
    ncols=4,  
    figsize=figsize,
    gridspec_kw={"width_ratios":[1,1,1, 0.04]}
    )

    for i in range(len(data_list)):
        if i == 0:
            # LR
            axs[0,0].imshow(data_list[0], vmin=vmin, vmax=vmax, cmap=cmap)
            axs[0,0].set_title(title_list[0], fontsize=font_size, weight='bold')
            axs[0,0].set_axis_off()
            # Draw zoom in 
            if zoom_loc_x and zoom_loc_y:
                axins = zoomed_inset_axes(axs[0,0], 6, loc='lower left')    
                axins.imshow(data_list[0], vmin=vmin, vmax=vmax, cmap=cmap)
                axins.set_xlim(tuple(val // args.upsampling_factor for val in zoom_loc_x))
                axins.set_ylim(tuple(val // args.upsampling_factor for val in zoom_loc_y))
                plt.xticks(visible=False)
                plt.yticks(visible=False)
                _patch, pp1, pp2 = mark_inset(axs[0,0], axins, loc1=2, loc2=4, fc=fc, ec=ec, lw=1.0, color=box_color) 
                pp1.loc1, pp1.loc2 = 2, 3  
                pp2.loc1, pp2.loc2 = 4, 1  
                plt.draw()
        else:
            im = axs[i//3,i%3].imshow(data_list[i], vmin=vmin, vmax=vmax, cmap=cmap)
            axs[i//3,i%3].set_title(title_list[i], fontsize=font_size, weight = 'bold')
            axs[i//3,i%3].set_axis_off()
            # Draw zoom in
            if zoom_loc_x and zoom_loc_y:
                axins = zoomed_inset_axes(axs[i//3,i%3], 6, loc='lower left')    
                axins.imshow(data_list[i], vmin=vmin, vmax=vmax, cmap=cmap)
                axins.set_xlim(zoom_loc_x)
                axins.set_ylim(zoom_loc_y)
                plt.xticks(visible=False)
                plt.yticks(visible=False)
                _patch, pp1, pp2 = mark_inset(axs[i//3,i%3], axins, loc1=2, loc2=4, fc=fc, ec=ec, lw=1.0, color=box_color)
                pp1.loc1, pp1.loc2 = 2, 3  
                pp2.loc1, pp2.loc2 = 4, 1  
                plt.draw()

 
    cbar = fig.colorbar(im, cax=axs[0,3], fraction=0.046, pad=0.04, extend='both')
    cbar.ax.tick_params(labelsize=label_size)
    cbar = fig.colorbar(im, cax=axs[1,3], fraction=0.046, pad=0.04, extend='both')
    cbar.ax.tick_params(labelsize=label_size)
    cbar = fig.colorbar(im, cax=axs[2,3], fraction=0.046, pad=0.04, extend='both')
    cbar.ax.tick_params(labelsize=label_size)
    
    # Adjust layout 
    fig.tight_layout()
   
    fig.savefig(os.path.join(project_dir,"results/model_plots_"+data_name+"_"+str(args.upsampling_factor)+"x_ch"+str(channel_num)+".png"), dpi=300, bbox_inches='tight', transparent=False)


def main():  
    parser = argparse.ArgumentParser(description='plotting parameters')

    parser.add_argument('--data_name', type=str, default='era5', help='dataset')
    parser.add_argument('--data_path', type=str, default='../datasets/era5', help='the folder path of dataset')
    parser.add_argument('--in_channels', type=int, default=3, help='num of input channels')
    parser.add_argument('--out_channels', type=int, default=3, help='num of output channels')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--seed', type=int, default=5544, help='random seed')
    parser.add_argument('--noise_ratio', type=float, default=0.0, help='noise ratio')
    parser.add_argument('--method', type=str, default="bicubic", help='downsample method')
    parser.add_argument('--crop_size', type=int, default=128, help='crop size')
    parser.add_argument('--n_patches', type=int, default=8, help='number of patches')
    parser.add_argument('--upsampling_factor', type=int, default=4, help='upsampling factor')
    parser.add_argument('--zero_shot', action='store_true', help='is it a zero-shot evaluation', default=False)
    parser.add_argument('--zero_shot_upsampling_factor', type=int, default=8, help='upsampling factor for zero shot evaluation')

    args = parser.parse_args()
    print(args)

    ## Get project directory path
    project_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")
    
    if args.data_name == "era5":
        channel_num = 0 #we plot wind speed, the channels are: 0(wind speed), 1(temperature), 2(total column water vapor)
        # Locations to zoom in for era5 where the spatial domain is 720x1440 (latxlon)
        zoom_loc_x = (820, 900) 
        zoom_loc_y = (230, 150)
        figsize = (11,6) # Customize this for different styles of vizualization
    else:
        channel_num = [0,1] # u,v component of wind velocity in the WTK dataset
        zoom_loc_x = None
        zoom_loc_y = None
        figsize = (10,5) # Customize this for different styles of vizualization

    plot_all_images(args,
                    project_dir = project_dir,
                    snapshot_num = 55, # This is the index of the image we want to plot, change this to plot different images
                    channel_num = channel_num,
                    zoom_loc_x = zoom_loc_x,
                    zoom_loc_y = zoom_loc_y, 
                    figsize = figsize,
                    cmap = 'coolwarm',
                    model_name_list= ['bicubic', 'ESRGAN', 'SwinIR', 'DFNO', 'DUNO', 'DCNO', 'DAFNO']) # We choose 7 model outputs + LR image + HR image (total: 9) since we are making a 3x3 plot


if __name__ == "__main__":
    main()
 
# To run this script within a .sh file, use the following (and update arguments as needed): 
# srun python result_analysis/plot_outputs.py \
#     --data_name wtk \
#     --crop_size 32 \
#     --in_channels 2 \
#     --out_channels 2 \
#     --upsampling_factor 5 \
#     --data_path '../datasets/era_to_wtk/region2_600x1600/' \
#     --zero_shot \
#     --zero_shot_upsampling_factor 15