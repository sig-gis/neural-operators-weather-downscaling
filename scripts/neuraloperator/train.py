'''training neural operators'''
'''the scripts/neuraloperator and src/neuraloperators directories are based on https://github.com/neuraloperator/neuraloperator'''

import argparse
import os
import sys
import time

import numpy as np
import torch
from configmypy import ArgparseConfig, ConfigPipeline, YamlConfig
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

import wandb
from scripts.baseline.utils import get_data_info
from src.data_loading import getTrainData
from src.neuraloperators.neuralop import LpLoss, MSEloss, Trainer, get_model
from src.neuraloperators.neuralop.training import setup
from src.neuraloperators.neuralop.training.callbacks import (
    BasicLoggerCallback, CheckpointCallback, MatplotlibWandBLoggerCallback)
from src.neuraloperators.neuralop.utils import (count_model_params,
                                                get_wandb_api_key)


def train_no(train_loader, val_loader, mean, std, upsampling_factor, project_dir):
    # Read the configuration file
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

    # Set up WandB logging
    wandb_args = None
    if config.wandb.log and is_logger:
        wandb.login(key=get_wandb_api_key(os.path.join(project_dir,'wandb_api_key.txt')))
    
        wandb_args =  dict(
            config=config,
            group=config.wandb.group,
            project=config.wandb.project,
            # entity=config.wandb.entity,
        )
        if config.wandb.sweep:
            for key in wandb.config.keys():
                config.params[key] = wandb.config[key]

    # Make sure we only print information when needed
    config.verbose = config.verbose and is_logger

    # Print config to screen
    if config.verbose and is_logger:
        pipe.log()
        sys.stdout.flush()

    model = get_model(config, mean, std, upsampling_factor)
    model = model.to(device)

    # Use distributed data parallel
    if config.distributed.use_distributed:
        model = DDP(
            model, device_ids=[device.index], output_device=device.index, static_graph=True
        )

    checkpoint_path = os.path.join(project_dir,'src/neuraloperators',config_arch["checkpoint_path"])
    # Create the optimizer and scheduler
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.opt.learning_rate,
        weight_decay=config.opt.weight_decay,
    )

    if config.opt.scheduler == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=config.opt.gamma,
            patience=config.opt.scheduler_patience,
            mode="min",
        )
    elif config.opt.scheduler == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.opt.scheduler_T_max
        )
    elif config.opt.scheduler == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=config.opt.step_size, gamma=config.opt.gamma
        )
    else:
        raise ValueError(f"Got scheduler={config.opt.scheduler}")


    # Creating the losses
    if arch == "tfno2d":
        l2loss = LpLoss(d=3, p=2,reduce_dims=[0]) # Used only for FNO, worked better than mse loss below
    else:
        l2loss = MSEloss() # We are using mse for all Downscaling(D) neural operator models
    train_loss = l2loss
    eval_losses = {"l2": l2loss}

    if config.verbose and is_logger:
        print("\n### MODEL ###\n", model)
        print("\n### OPTIMIZER ###\n", optimizer)
        print("\n### SCHEDULER ###\n", scheduler)
        print("\n### LOSSES ###")
        print(f"\n * Train: {train_loss}")
        print(f"\n * Test: {eval_losses}")
        print(f"\n### Beginning Training...\n")
        sys.stdout.flush()

    trainer = Trainer(
        model=model,
        n_epochs=config.opt.n_epochs,
        device=device,
        data_processor=None,
        amp_autocast=config.opt.amp_autocast,
        wandb_log=config.wandb.log,
        log_test_interval=config.wandb.log_test_interval,
        log_output=config.wandb.log_output,
        use_distributed=config.distributed.use_distributed,
        verbose=config.verbose and is_logger,
        callbacks=[CheckpointCallback(checkpoint_path, save_best = "l2"), 
            MatplotlibWandBLoggerCallback(wandb_args) # This can be extended to log images during training if needed
        ]
                )

    # Log parameter count
    if is_logger:
        n_params = count_model_params(model)

        if config.verbose:
            print(f"\nn_params: {n_params}")
            sys.stdout.flush()

        if config.wandb.log:
            to_log = {"n_params": n_params}
            if config.n_params_baseline is not None:
                to_log["n_params_baseline"] = (config.n_params_baseline,)
                to_log["compression_ratio"] = (config.n_params_baseline / n_params,)
                to_log["space_savings"] = 1 - (n_params / config.n_params_baseline)
            wandb.log(to_log)
            wandb.watch(model)

    trainer.train(
        train_loader=train_loader,
        valid_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        regularizer=False,
        training_loss=train_loss,
        eval_losses=eval_losses,
    )

    if config.wandb.log and is_logger:
        wandb.finish()

def main():
    parser = argparse.ArgumentParser(description='training parameters')

    # arguments for data
    parser.add_argument('--data_name', type=str, default='era5', help='dataset')
    parser.add_argument('--data_path', type=str, default='../datasets/era5', help='the folder path of dataset')
    parser.add_argument('--crop_size', type=int, default=128, help='crop size')
    parser.add_argument('--n_patches', type=int, default=8, help='number of patches')    
    parser.add_argument('--method', type=str, default="bicubic", help='downsample method') #for ERA5 experiments : the LR images are created by coarsening the HR patches with bicubic interpolation.
    parser.add_argument('--pretrained', default=False, type=lambda x: (str(x).lower() == 'true'), help='load the pretrained model')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--upsampling_factor', type=int, default=4, help='upsampling factor')
    parser.add_argument('--seed', type=int, default=5544, help='random seed')
    parser.add_argument('--noise_ratio', type=float, default=0.0, help='noise ratio') #this isn't needed as we do bicubic downsampling (to derive LR from HR) in our ERA5 experiments

    
    parser.add_argument('--project_dir',type=str,default='./')

    args = parser.parse_args()
    print(args)

    ## Get project directory path
    project_dir = args.project_dir

    # Set random seed 
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Load data
    resol, n_fields, mean, std = get_data_info(args.data_name) 
    train_loader, val_loader = getTrainData(args, std=std)
    print("mean is: ",mean)
    print("std is: ",std)

    # Call the train function
    train_no(train_loader, val_loader, mean, std, args.upsampling_factor, project_dir)


if __name__ =='__main__':
    main()

