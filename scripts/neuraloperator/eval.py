'''Evaluation: Computes MSE, MAE, Infinity Norm, PSNR and SSIM to compare all Neural Operator based models'''

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from configmypy import ConfigPipeline, YamlConfig
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from scripts.baseline.utils import *
from src.data_loading import getTestData
from src.neuraloperators.neuralop import get_model
from src.neuraloperators.neuralop.training import setup


def normalize(args, target, mean, std):
    if isinstance(mean[0], list):
        mean = mean[1]
        std = std[1]
    mean = torch.Tensor(mean).to(args.device).view(1,target.shape[1],1,1)
    std = torch.Tensor(std).to(args.device).view(1,target.shape[1],1,1)
    target = (target - mean) / std
    return target


def validate_MSE(args, test1_loader, test2_loader, model,mean,std, channel_num=None):
    '''MSE'''
    error = []   
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate((test1_loader)):
            data, target = data.to(args.device).float(), target.to(args.device).float()
            output = model(data) 
        
            ## The output size is the HR size for NO-based models
            ## For the case of dafno, the output shapes can be off by a bit
            if output.shape[2]!=target.shape[2] or output.shape[3]!=target.shape[3]:
                output = F.interpolate(output, size=(target.shape[2], target.shape[3]), mode='bicubic', align_corners=False)
            
            ## Choose to normalize the error (we did not)
            # output = normalize(args,output,mean,std)
            # target = normalize(args,target,mean,std)

            for i in range(target.shape[0]):
                ##Either get a channel-wise loss, or aggregated over all channels
                if channel_num is not None:
                    j = channel_num
                    err_mse = torch.mean((target[i,j,...] - output[i,j,...]) ** 2)
                    error.append(err_mse)
                else:
                    for j in range(target.shape[1]):
                        err_mse = torch.mean((target[i,j,...] - output[i,j,...]) ** 2)
                        error.append(err_mse)

        if test2_loader is not None:
            for batch_idx, (data, target) in enumerate((test2_loader)):
                data, target = data.to(args.device).float(), target.to(args.device).float()
                output = model(data) 
                ## The output size is the HR size for NO-based models
                ## For the case of dafno, the output shapes can be off by a bit
                if output.shape[2]!=target.shape[2] or output.shape[3]!=target.shape[3]:
                    output = F.interpolate(output, size=(target.shape[2], target.shape[3]), mode='bicubic', align_corners=False)
                
                # output = normalize(args,output,mean,std)
                # target = normalize(args,target,mean,std)
                if channel_num is not None:
                    j = channel_num
                    err_mse = torch.mean((target[i,j,...] - output[i,j,...]) ** 2)
                    error.append(err_mse)
                else:
                    for j in range(target.shape[1]):
                        err_mse = torch.mean((target[i,j,...] - output[i,j,...]) ** 2)
                        error.append(err_mse)

    # Error aggregated over test1 and test2 (if it exists). 
    # We'll have test_2 for ERA5->WTK experiments, so that the error is aggregated over both regions.
    error = torch.mean(torch.tensor(error)).item()

    return error


def validate_MAE(args, test1_loader, test2_loader, model,mean,std, channel_num=None):
    '''MAE'''
    
    error = []   
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate((test1_loader)):
            data, target = data.to(args.device).float(), target.to(args.device).float()
            output = model(data) 
            ## The output size is the HR size for NO-based models
            ## For the case of dafno, the output shapes can be off by a bit
            if output.shape[2]!=target.shape[2] or output.shape[3]!=target.shape[3]:
                output = F.interpolate(output, size=(target.shape[2], target.shape[3]), mode='bicubic', align_corners=False)
            
            ## Choose to normalize the error (we did not)
            # output = normalize(args,output,mean,std)
            # target = normalize(args,target,mean,std)

            for i in range(target.shape[0]):
                ##Either get a channel-wise loss, or aggregated over all channels
                if channel_num is not None:
                    j = channel_num
                    err_mae = torch.mean(torch.abs(target[i,j,...] - output[i,j,...]))
                    error.append(err_mae)
                else:
                    for j in range(target.shape[1]):
                        err_mae = torch.mean(torch.abs(target[i,j,...] - output[i,j,...]))
                        error.append(err_mae)

        if test2_loader is not None:
            for batch_idx, (data, target) in enumerate((test2_loader)):
                data, target = data.to(args.device).float(), target.to(args.device).float()
                output = model(data) 
                ## The output size is the HR size for NO-based models
                ## For the case of dafno, the output shapes can be off by a bit
                if output.shape[2]!=target.shape[2] or output.shape[3]!=target.shape[3]:
                    output = F.interpolate(output, size=(target.shape[2], target.shape[3]), mode='bicubic', align_corners=False)
                
                # output = normalize(args,output,mean,std)
                # target = normalize(args,target,mean,std)
                for i in range(target.shape[0]):
                    if channel_num is not None:
                        j = channel_num
                        err_mae = torch.mean(torch.abs(target[i,j,...] - output[i,j,...]))
                        error.append(err_mae)
                    else:
                        for j in range(target.shape[1]):
                            err_mae = torch.mean(torch.abs(target[i,j,...] - output[i,j,...]))
                            error.append(err_mae)

    # Error aggregated over test1 and test2 (if it exists). 
    # We'll have test_2 for ERA5->WTK experiments, so that the error is aggregated over both regions.
    error = torch.mean(torch.tensor(error)).item()

    return error

def validate_RINE(args, test1_loader, test2_loader, model,mean,std, channel_num=None):
    '''Relative infinity norm error (RINE)'''
   
    ine = [] 
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate((test1_loader)):
            data, target = data.to(args.device).float(), target.to(args.device).float() # [b,c,h,w]
            output = model(data) 
            ## The output size is the HR size for NO-based models
            ## For the case of dafno, the output shapes can be off by a bit
            if output.shape[2]!=target.shape[2] or output.shape[3]!=target.shape[3]:
                output = F.interpolate(output, size=(target.shape[2], target.shape[3]), mode='bicubic', align_corners=False)
            
            ## Choose to normalize the error (we did not)
            # output = normalize(args,output,mean,std)
            # target = normalize(args,target,mean,std)
                
            for i in range(target.shape[0]):
                ##Either get a channel-wise loss, or aggregated over all channels
                if channel_num is not None:
                    j = channel_num
                    err_ine = torch.norm((target[i,j,...]-output[i,j,...]), p=np.inf)
                    ine.append(err_ine)
                else:
                    for j in range(target.shape[1]):
                        err_ine = torch.norm((target[i,j,...]-output[i,j,...]), p=np.inf)
                        ine.append(err_ine)

        if test2_loader is not None:
            for batch_idx, (data, target) in enumerate((test2_loader)):
                data, target = data.to(args.device).float(), target.to(args.device).float() # [b,c,h,w]
                output = model(data) 
                ## The output size is the HR size for NO-based models
                ## For the case of dafno, the output shapes can be off by a bit
                if output.shape[2]!=target.shape[2] or output.shape[3]!=target.shape[3]:
                    output = F.interpolate(output, size=(target.shape[2], target.shape[3]), mode='bicubic', align_corners=False)
                
                # output = normalize(args,output,mean,std)
                # target = normalize(args,target,mean,std)
                for i in range(target.shape[0]):
                    if channel_num is not None:
                        j = channel_num
                        err_ine = torch.norm((target[i,j,...]-output[i,j,...]), p=np.inf)
                        ine.append(err_ine)
                    else:
                        for j in range(target.shape[1]):
                            err_ine = torch.norm((target[i,j,...]-output[i,j,...]), p=np.inf)
                            ine.append(err_ine)

    # Error aggregated over test1 and test2 (if it exists). 
    # We'll have test_2 for ERA5->WTK experiments, so that the error is aggregated over both regions.
    ine = torch.mean(torch.tensor(ine)).item()
        
    return ine


def psnr(true, pred):
    mse = torch.mean((true - pred) ** 2)
    if mse == 0:
        return float('inf')
    max_value = torch.max(true)
    return 20 * torch.log10(max_value / torch.sqrt(mse))


def validate_PSNR(args, test1_loader, test2_loader, model,mean,std, channel_num=None):
    '''Peak signal-to-noise ratio (PSNR)'''
   
    score = []   
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate((test1_loader)):
            data, target = data.to(args.device).float(), target.to(args.device).float()
            output = model(data) 
            ## The output size is the HR size for NO-based models
            ## For the case of dafno, the output shapes can be off by a bit
            if output.shape[2]!=target.shape[2] or output.shape[3]!=target.shape[3]:
                output = F.interpolate(output, size=(target.shape[2], target.shape[3]), mode='bicubic', align_corners=False)
            
            ## Choose to normalize the error (we did not)
            # output = normalize(args,output,mean,std)
            # target = normalize(args,target,mean,std)
           
            for i in range(target.shape[0]):
                ##Either get a channel-wise loss, or aggregated over all channels
                if channel_num is not None:
                    j = channel_num
                    err_psnr = psnr(target[i,j,...], output[i,j,...])
                    score.append(err_psnr)
                else:
                    for j in range(target.shape[1]):
                        err_psnr = psnr(target[i,j,...], output[i,j,...])
                        score.append(err_psnr)

        if test2_loader is not None:
            for batch_idx, (data, target) in enumerate((test2_loader)):
                data, target = data.to(args.device).float(), target.to(args.device).float()
                output = model(data) 
                ## The output size is the HR size for NO-based models
                ## For the case of dafno, the output shapes can be off by a bit
                if output.shape[2]!=target.shape[2] or output.shape[3]!=target.shape[3]:
                    output = F.interpolate(output, size=(target.shape[2], target.shape[3]), mode='bicubic', align_corners=False)
                
                # output = normalize(args,output,mean,std)
                # target = normalize(args,target,mean,std)
                for i in range(target.shape[0]):
                    if channel_num is not None:
                        j = channel_num
                        err_psnr = psnr(target[i,j,...], output[i,j,...])
                        score.append(err_psnr)
                    else:
                        for j in range(target.shape[1]):
                            err_psnr = psnr(target[i,j,...], output[i,j,...])
                            score.append(err_psnr)

    # Error aggregated over test1 and test2 (if it exists). 
    # We'll have test_2 for ERA5->WTK experiments, so that the error is aggregated over both regions.
    score = torch.mean(torch.tensor(score)).item()

    return score


def validate_SSIM(args, test1_loader, test2_loader, model,mean,std, channel_num=None):
    '''Structual Similarity Index Measure (SSIM)'''
    
    from torchmetrics import StructuralSimilarityIndexMeasure
    ssim = StructuralSimilarityIndexMeasure().to(args.device)
    
    score = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate((test1_loader)):
            data, target = data.to(args.device).float(), target.to(args.device).float()
            output = model(data) 
            ## The output size is the HR size for NO-based models
            ## For the case of dafno, the output shapes can be off by a bit
            if output.shape[2]!=target.shape[2] or output.shape[3]!=target.shape[3]:
                output = F.interpolate(output, size=(target.shape[2], target.shape[3]), mode='bicubic', align_corners=False)
            
            ## Choose to normalize the error (we did not)
            # output = normalize(args,output,mean,std)
            # target = normalize(args,target,mean,std)
                
            for i in range(target.shape[0]):
                ##Either get a channel-wise loss, or aggregated over all channels
                if channel_num is not None:
                    j = channel_num
                    err_ssim = ssim(target[i:(i+1),j:(j+1),...], output[i:(i+1),j:(j+1),...])
                    score.append(err_ssim.cpu())
                else:
                    for j in range(target.shape[1]):
                        err_ssim = ssim(target[i:(i+1),j:(j+1),...], output[i:(i+1),j:(j+1),...])
                        score.append(err_ssim.cpu())

        if test2_loader is not None:
            for batch_idx, (data, target) in enumerate((test2_loader)):
                data, target = data.to(args.device).float(), target.to(args.device).float()
                output = model(data) 
                ## The output size is the HR size for NO-based models
                ## For the case of dafno, the output shapes can be off by a bit
                if output.shape[2]!=target.shape[2] or output.shape[3]!=target.shape[3]:
                    output = F.interpolate(output, size=(target.shape[2], target.shape[3]), mode='bicubic', align_corners=False)
                
                # output = normalize(args,output,mean,std) 
                # target = normalize(args,target,mean,std)
                for i in range(target.shape[0]):
                    if channel_num is not None:
                        j = channel_num
                        err_ssim = ssim(target[i:(i+1),j:(j+1),...], output[i:(i+1),j:(j+1),...])
                        score.append(err_ssim.cpu())
                    else:
                        for j in range(target.shape[1]):
                            err_ssim = ssim(target[i:(i+1),j:(j+1),...], output[i:(i+1),j:(j+1),...])
                            score.append(err_ssim.cpu())

    # Error aggregated over test1 and test2 (if it exists). 
    # We'll have test_2 for ERA5->WTK experiments, so that the error is aggregated over both regions.
    score = torch.mean(torch.tensor(score)).item()

    return score

def main():  
    parser = argparse.ArgumentParser(description='evaluation parameters')
    # arguments for data
    parser.add_argument('--data_name', type=str, default='era5', help='dataset')
    parser.add_argument('--data_path', type=str, default='../../datasets/era5', help='the folder path of dataset')
    parser.add_argument('--method', type=str, default="bicubic", help='downsample method')
    parser.add_argument('--crop_size', type=int, default=128, help='crop size')
    parser.add_argument('--n_patches', type=int, default=8, help='number of patches')
    parser.add_argument('--noise_ratio', type=float, default=0.0, help='noise ratio')

    # arguments for evaluation
    parser.add_argument('--device', type=str, default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), help='computing device')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--seed', type=int, default=5544, help='random seed')
    parser.add_argument('--upsampling_factor', type=int, default=4, help='upsampling factor')
    parser.add_argument('--zero_shot', action='store_true', help='is it a zero-shot evaluation', default=False)
    parser.add_argument('--zero_shot_upsampling_factor', type=int, default=8, help='upsampling factor for zero shot evaluation')

    args = parser.parse_args()
    print(args)

    ## Get project directory path
    project_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")

    # Set random seed 
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Load data
    resol, n_fields, mean, std = get_data_info(args.data_name)
    test1_loader, test2_loader = None, None

    if args.zero_shot:
        ## If we are doing zero-shot evaluation
        args.upsampling_factor = args.zero_shot_upsampling_factor
    if args.data_name=="wtk":
        prev_path=args.data_path
        args.data_path = prev_path+ "/32_-122_600x1600/" ## this is the path to the first region for ERA5->WTK experiments
        test1_loader = getTestData(args, std=std)

        args.data_path = prev_path+"/27_-97_800x800/" ## this is the path to the second region for ERA5->WTK experiments
        test2_loader = getTestData(args, std=std)
    else:
        test1_loader = getTestData(args, std=std)

    for batch in test1_loader:
        inputs, targets = batch
        print(f"Input shape: {inputs.shape}")
        print(f"Target shape: {targets.shape}")
        break  
        

    pipe = ConfigPipeline(
        [
            YamlConfig(
                "./config.yaml", config_name="default", config_folder=os.path.join(project_dir,'src/neuraloperators/config')
            ),
            YamlConfig(config_folder=os.path.join(project_dir,'src/neuraloperators/config')),
        ]
    )
    config = pipe.read_conf()
    arch = config["arch"].lower()
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

    out_channels = config_arch["out_channels"]
    model.eval()
   
    # =============== Get evaluation metrics ======================
    with open(os.path.join(project_dir,'results/result.txt'), "a") as f:
        print(" model" + str(arch) + " data: " + str(args.data_name)+ " zero-shot: " + str(args.zero_shot),file = f) # Add more model details to this file if needed
        if args.data_name=="era5":
            ## Save channel-wise results for ERA5
            for channel_num in range(out_channels):
                print("channel num: ", channel_num, file=f)
                mse_error = validate_MSE(args, test1_loader, test2_loader, model , mean, std, channel_num)
                print("MSE --- test error: %.5f" % mse_error, file=f)   

                mae_error = validate_MAE(args, test1_loader, test2_loader, model , mean,std, channel_num)
                print("MAE --- test error: %.5f" % mae_error, file=f)      
                
                in_error  = validate_RINE(args, test1_loader, test2_loader, model, mean,std, channel_num)
                print("Infinity norm --- test error: %.8f" % in_error, file=f)

                psnr_score = validate_PSNR(args, test1_loader, test2_loader, model, mean,std, channel_num)
                print("PSNR --- test error: %.5f" % psnr_score, file=f) 

                ssim_score = validate_SSIM(args, test1_loader, test2_loader, model,  mean,std, channel_num)
                print("SSIM --- test error: %.5f" % ssim_score, file=f) 
        else:
            ## For the ERA->WTK experiments, we save aggregated results over the channels
            ## These results are also aggregated over the two regions in the dataset.
            mse_error = validate_MSE(args, test1_loader, test2_loader, model , mean, std)
            print("MSE --- test error: %.5f" % mse_error, file=f)   

            mae_error = validate_MAE(args, test1_loader, test2_loader, model , mean,std)
            print("MAE --- test error: %.5f" % mae_error, file=f)      
            
            in_error  = validate_RINE(args, test1_loader, test2_loader, model, mean,std)
            print("Infinity norm --- test error: %.8f" % in_error, file=f)

            psnr_score = validate_PSNR(args, test1_loader, test2_loader, model, mean,std)
            print("PSNR --- test error: %.5f" % psnr_score, file=f) 

            ssim_score = validate_SSIM(args, test1_loader, test2_loader, model,  mean,std)
            print("SSIM --- test error: %.5f" % ssim_score, file=f) 

if __name__ =='__main__':
    main()