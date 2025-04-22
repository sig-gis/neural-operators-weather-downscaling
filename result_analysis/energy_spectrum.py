import argparse
import os
import random
import sys

import matplotlib.image as mpimg
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import torch
import torch.nn.functional as F
from configmypy import ConfigPipeline, YamlConfig
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from PIL import Image
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from scripts.baseline.utils import *
from src.baseline_models import *
from src.baseline_models.ESRGAN import test_utils as esrgan_test_utils
from src.baseline_models.ESRGAN.config import config as esrgan_config
from src.data_loading import getTestData
from src.neuraloperators.neuralop import get_model
from src.neuraloperators.neuralop.training import setup


def wavenumber_spectrum(var, x_range=None, axis=0):
    """
    takes in KE as input (from the ERA5 dataset)
    Returns an array of wavenumbers, and portion of the given variable associated with each wavenumber
    function based on https://github.com/NREL/sup3r/blob/main/sup3r/qa/utilities.py
    """
    var_k = np.fft.fftn(var)
    E_k = np.mean(np.abs(var_k)**2, axis=axis)
    if x_range is None:
        k = np.arange(len(E_k))
    else:
        k = np.linspace(x_range[0], x_range[1], len(E_k))
    n_steps = len(k) // 2
    E_k = k**2 * E_k
    E_k_a = E_k[1:n_steps + 1]
    E_k_b = E_k[-n_steps:][::-1]
    E_k = E_k_a + E_k_b
    return k[:n_steps], E_k

def tke_wavenumber_spectrum(u, v, x_range=None, axis=0):
    """
    takes in u and v as a input 
    Returns an array of wavenumners, and the portion of kinetic energy associated with each wavenumber
    function based on https://github.com/NREL/sup3r/blob/main/sup3r/qa/utilities.py
    """
    u_k = np.fft.fftn(u)
    v_k = np.fft.fftn(v)
    E_k = np.mean(np.abs(v_k)**2 + np.abs(u_k)**2, axis=axis)
    if x_range is None:
        k = np.arange(len(E_k))
    else:
        k = np.linspace(x_range[0], x_range[1], len(E_k))
    n_steps = len(k) // 2
    E_k = k**2 * E_k
    E_k_a = E_k[1:n_steps + 1]
    E_k_b = E_k[-n_steps:][::-1]
    E_k = E_k_a + E_k_b
    return k[:n_steps], E_k


def plot_energy_spectra(Energy_Spectrum, project_dir, args):
   
    plt.rc('font', size=16) 
    plt.rc('axes', titlesize=14) 
    plt.rc('axes', labelsize=14) 
    plt.rc('xtick', labelsize=14) 
    plt.rc('ytick', labelsize=14) 
    plt.rc('legend', fontsize=14) 
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ## The following plots a zoomed in image besides the main image

    ## Update the color palatte as needed (and according to the number of models)
    colors = [
    "#0173b2",  # Blue
    "#de8f05",  # Orange
    "#029e73",  # Green
    "#d55e00",  # Vermillion
    "#cc78bc",  # Purple
    "#ca9161",  # Brown
    "#fbafe4",  # Pink
    "#949494",  # Gray
    "#000000",  # Black
    "#56b4e9",  # Sky Blue
    "#f58231",  # Bright Orange
    ]
    for i,model in enumerate(Energy_Spectrum):
        print(len(Energy_Spectrum[model]['x']), len(Energy_Spectrum[model]['y']))
        k = np.mean(Energy_Spectrum[model]['x'], axis=0)
        E = np.mean(Energy_Spectrum[model]['y'], axis=0) 
        axes[0].loglog(k, E, label=model, color = colors[i])

    x_min, x_max = axes[0].get_xlim() 
    y_min, y_max = axes[0].get_ylim()
    x_min = 50 #set according to preference for the zoomed in image
    y_max = 1e-2 #set according to preference for the zoomed in image
    print(x_max,y_min)

    rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                         linewidth=1.5, edgecolor='red', facecolor='none', linestyle='--')
    axes[0].add_patch(rect)
    axes[0].set_xlabel("k (wavenumber)")
    axes[0].set_ylabel("Kinetic Energy")
    axes[0].set_title("Energy Spectrum")
    axes[0].legend()

    for i,model in enumerate(Energy_Spectrum):
        print(len(Energy_Spectrum[model]['x']), len(Energy_Spectrum[model]['y']))
        k = np.mean(Energy_Spectrum[model]['x'], axis=0)
        E = np.mean(Energy_Spectrum[model]['y'], axis=0) 

        mask = (k >= x_min) & (k <= x_max) & (E >= y_min) & (E <= y_max)
        k_filtered = k[mask]
        E_filtered = E[mask] 
        axes[1].loglog(k_filtered, E_filtered, label=model, color=colors[i])

    axes[1].set_xlabel("k (wavenumber)")
    axes[1].set_ylabel("Kinetic Energy")
    axes[1].set_title("Energy Spectrum: zoomed in")
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(project_dir,'results/KE_spectrum_with_zoomed_in_for_'+args.data_name+'_'+str(args.upsampling_factor)+'x.png'), dpi=1000, transparent=True, bbox_inches='tight')


def load_model(args, project_dir, data_name, model_name, upsampling_factor):
    '''load model'''

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
            'SRCNN': SRCNN(in_channels, upsampling_factor, mean, std),
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

def get_energy_spectrum(args, model_name_list, channel_num, project_dir):

    data_name = args.data_name
    upsample = args.upsampling_factor
    zero_shot = args.zero_shot
    # Adjust the upsampling factor if needed
    if zero_shot:
        args.upsampling_factor = args.zero_shot_upsampling_factor  

    # title_list = ['LR','HR']+model_name_list
    title_list = ['HR']+model_name_list 

    Energy_Spectrum = {}
    for name in title_list:
        Energy_Spectrum[name] = {'x':[], 'y':[]} 

    # get data
    resol, n_fields,  mean, std = get_data_info(data_name)
    data = getTestData(args, std)
 
    random_indices = random.sample(range(len(data)), 1000) #Select 1000 random samples to compute energy ("y") and wavenumber ("x") lists for each model. For ERA5→ERA5 experiments, we use all samples.
    for snapshot_num in random_indices:
        i = 2
        data_list = []  # This list contains the model outputs for all the models including the HR
        for name in model_name_list:
            model = load_model(args, project_dir, data_name=data_name, model_name=name, upsampling_factor=upsample)
            LR, target, output = get_one_image(model, data, snapshot_num, channel_num, zero_shot)
            
            if i == 2:
                # LR
                # data_list.append(LR)
                # HR
                HR = target
                data_list.append(HR)
            
            data_list.append(output)
            i += 1

        for i,data_item in enumerate(data_list):
            model_name = title_list[i]
            if data_name=="era5":
                kvals2, ek = wavenumber_spectrum(data_item)
            elif data_name=="wtk":
                kvals2, ek = tke_wavenumber_spectrum(data_item[0,:,:], data_item[1,:,:])
            ek = ek/ek[0] # Normalizing with respect to first energy value
            Energy_Spectrum[model_name]['x'].append(kvals2)
            Energy_Spectrum[model_name]['y'].append(ek)

    return Energy_Spectrum

def main():  
    parser = argparse.ArgumentParser(description='energy spectrum parameters')

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
        channel_num = 0 # wind speed channel in the ERA5 dataset
    else:
        channel_num = [0,1] # u,v component of wind velocity in the WTK dataset
    
    model_name_list= ['bicubic', 'ESRGAN', 'EDSR', 'SwinIR', 'FNO', 'DFNO', 'DUNO', 'DAFNO', 'DCNO'] ## all models to be included in KE spectrum analysis
    Energy_Spectrum = get_energy_spectrum(args, model_name_list, channel_num, project_dir)
    plot_energy_spectra(Energy_Spectrum, project_dir, args)

if __name__ == '__main__':
   main()

# To run this script within a .sh file, use the following (and update arguments as needed): 
# srun python result_analysis/energy_spectrum.py \
#     --data_name wtk \
#     --crop_size 32 \
#     --in_channels 2 \
#     --out_channels 2 \
#     --upsampling_factor 5 \
#     --data_path '../datasets/era_to_wtk/region2_600x1600/' \
#     --zero_shot \
#     --zero_shot_upsampling_factor 15