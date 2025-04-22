'''Evaluation: Computes MSE, MAE, Infinity Norm, PSNR and SSIM to compare all baseline models'''

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from utils import *

from src.baseline_models import *
from src.baseline_models.ESRGAN import test_utils as esrgan_test_utils
from src.baseline_models.ESRGAN.config import config as esrgan_config
from src.data_loading import getTestData


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
        
            if args.zero_shot:
                ## Applying bicubic interpolation to the model output to match the output (HR) size expected in zero-shot experiments
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
                if args.zero_shot:
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

            if args.zero_shot:
                ## Applying bicubic interpolation to the model output to match the output (HR) size expected in zero-shot experiments
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
                if args.zero_shot:
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

            if args.zero_shot:
                ## Applying bicubic interpolation to the model output to match the output (HR) size expected in zero-shot experiments
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
                if args.zero_shot:
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


def validate_PSNR(args, test1_loader, test2_loader,model,mean,std, channel_num=None):
    '''Peak signal-to-noise ratio (PSNR)'''
   
    score = []   
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate((test1_loader)):
            data, target = data.to(args.device).float(), target.to(args.device).float()
            output = model(data) 

            if args.zero_shot:
                ## Applying bicubic interpolation to the model output to match the output (HR) size expected in zero-shot experiments
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
                if args.zero_shot:
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
            if args.zero_shot:
                ## Applying bicubic interpolation to the model output to match the output (HR) size expected in zero-shot experiments
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
                if args.zero_shot:
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
    parser.add_argument('--data_name', type=str, default="era5", help='dataset')
    parser.add_argument('--data_path', type=str, default='../datasets/era5', help='the folder path of dataset')
    parser.add_argument('--crop_size', type=int, default=128, help='crop size')
    parser.add_argument('--n_patches', type=int, default=8, help='number of patches')    
    parser.add_argument('--method', type=str, default="bicubic", help='downsample method')
    parser.add_argument('--noise_ratio', type=float, default=0.0, help='noise ratio') 
    
    # arguments for evaluation
    parser.add_argument('--model', type=str, default='EDSR', help='model')
    parser.add_argument('--device', type=str, default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), help='computing device')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--seed', type=int, default=5544, help='random seed')
    parser.add_argument('--zero_shot', action='store_true', help='is it a zero-shot evaluation', default=False)
    parser.add_argument('--zero_shot_upsampling_factor', type=int, default=8, help='upsampling factor for zero shot evaluation')
    
    # arguments used for model training
    parser.add_argument('--epochs', type=int, default=300, help='max epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--wd', type=float, default=1e-6, help='weight decay')
    parser.add_argument('--step_size', type=int, default=100, help='step size for scheduler')
    parser.add_argument('--gamma', type=float, default=0.97, help='coefficient for scheduler')
    parser.add_argument('--loss_type', type=str, default='l1', help='L1 or L2 loss')
    parser.add_argument('--optimizer_type', type=str, default='Adam', help='type of optimizer')
    parser.add_argument('--scheduler_type', type=str, default='ExponentialLR', help='type of scheduler')
    parser.add_argument('--upsampling_factor', type=int, default=4, help='upsampling factor')
    parser.add_argument('--in_channels', type=int, default=1, help='num of input channels')
    parser.add_argument('--hidden_channels', type=int, default=32, help='num of hidden channels')
    parser.add_argument('--out_channels', type=int, default=1, help='num of output channels')
    parser.add_argument('--n_res_blocks', type=int, default=18, help='num of resdiual blocks')

    args = parser.parse_args()
    print(args)

    ## Get project directory path
    project_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")

    # Set random seed 
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Load data
    resol, n_fields,  mean, std = get_data_info(args.data_name)
    test1_loader, test2_loader = None, None

    upsample = args.upsampling_factor
    
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
        

    # Get model
    ##Following are the parameters used in SwinIR
    window_size = 8
    crop_size = args.crop_size if args.data_name == 'era5' else args.crop_size * upsample
    height = (crop_size // upsample // window_size + 1) * window_size 
    width = (crop_size // upsample // window_size + 1) * window_size

    model_list = {
            'SRCNN': SRCNN(args.in_channels, upsample,mean,std),
            'EDSR': EDSR(args.in_channels, args.hidden_channels, args.n_res_blocks, upsample, mean, std),
            'SwinIR': SwinIR(upscale=upsample, in_chans=args.in_channels, img_size=(height, width),
                    window_size=window_size, img_range=1., depths=[6, 6, 6, 6, 6, 6],
                    embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv',mean =mean,std=std),
    }
    if args.model == "ESRGAN":
        tester = esrgan_test_utils.Tester(esrgan_config, mean, std, args.device, upsample)
        model = tester.build_model()

         # Model summary   
        print('**** Setup ****')
        print('Total params Generator: %.3fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
        print('************')    
        
    elif args.model != 'bicubic':
        model = model_list[args.model]
        model_path = os.path.join(project_dir,'results/model_' + str(args.model) + '_' + str(args.data_name) + '_' + str(upsample) + '_' + str(args.lr) + '_' + str(args.method) +'_' + str(args.seed) + '.pt')

        model = load_checkpoint(model, model_path)
        model = model.to(args.device)

        # Model summary   
        print('**** Setup ****')
        print('Total params Generator: %.3fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
        print('************')    

    else: 
        print('Using bicubic interpolation...')  
        model = Bicubic(upsampling_factor=args.upsampling_factor)

    # =============== Get evaluation metrics ======================
    with open(os.path.join(project_dir,'results/result.txt'), "a") as f:
        print(" model" + str(args.model) + " data: " + str(args.data_name)+ " zero-shot: " + str(args.zero_shot),file = f) # Add more model details to this file if needed
        if args.data_name=="era5":
            ## Save channel-wise results for ERA5
            for channel_num in range(args.out_channels):
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