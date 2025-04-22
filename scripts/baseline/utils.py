import numpy as np
import torch
from torch import nn


def get_data_info(data_name):
    if data_name == 'era5':
        resol = [720, 1440] #whole globe
        n_fields = 3
        # Get the mean and std for the three channels for the ERA5 data
        mean = [6.3024, 278.3945, 18.4262] 
        std = [3.7376, 21.0588, 16.4687]
    
    elif data_name == 'wtk':
        resol = None # Set as none since the dataset includes multiple regions, each with a different resolution
        n_fields = 2
        # Get the mean and std for the two channels for both the ERA5 (x) and WTK (y) datasets
        mean_y = [0.4309, 0.239] 
        std_y = [3.1699, 3.5203]
        mean_x = [0.4113,0.097] 
        std_x = [2.8555, 3.4064]

        mean = [mean_x,mean_y]
        std = [std_x, std_y]

    else:
        raise ValueError('dataset {} not recognized'.format(data_name))

    return resol, n_fields, mean , std


def set_optimizer(args, model):
    if args.optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optimizer_type == 'AdamW':
        # swin transformer
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    else:
        raise ValueError('Optimizer type {} not recognized'.format(args.optimizer_type))
    return optimizer


def get_lr(step, total_steps, lr_max, lr_min):
    """Compute learning rate according to cosine annealing schedule."""
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


def set_scheduler(args, optimizer, train_loader):
    if args.scheduler_type == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda step: get_lr(  # pylint: disable=g-long-lambda
                    step, args.epochs * len(train_loader),
                    1,  # lr_lambda computes multiplicative factor
                    1e-6 / args.lr))  

    elif args.scheduler_type == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step_size, args.gamma)

    elif args.scheduler_type == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.gamma)

    return scheduler


def loss_function(args):
    if args.loss_type == 'l1':
        print('Training with L1 loss...')
        criterion = nn.L1Loss().to(args.device)
    elif args.loss_type == 'l2': 
        print('Training with L2 loss...')
        criterion = nn.MSELoss().to(args.device)
    else:
        raise ValueError('Loss type {} not recognized'.format(args.loss_type))
    return criterion

def save_checkpoint(model, save_path):
    '''save model and optimizer'''
    torch.save({
        'model_state_dict': model.state_dict()
        }, save_path)


def load_checkpoint(model, save_path):
    '''load model and optimizer'''
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    print('Model loaded...')

    return model
