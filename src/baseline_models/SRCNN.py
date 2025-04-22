# -*- coding: utf-8 -*-
"""
SRCNN: Image Super-Resolution Using Deep Convolutional Networks
Ref: https://arxiv.org/pdf/1501.00092v3.pdf

@author: Pu Ren
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ShiftMean(nn.Module):
    # data: [t,c,h,w]
    def __init__(self, mean, std):
        super(ShiftMean, self).__init__()
        len_c = mean.shape[0]
        self.mean = torch.Tensor(mean).view(1, len_c, 1, 1)
        self.std = torch.Tensor(std).view(1, len_c, 1, 1)

    def forward(self, x, mode):
        if mode == 'sub':
            return (x - self.mean.to(x.device)) / self.std.to(x.device)
        elif mode == 'add':
            return x * self.std.to(x.device) + self.mean.to(x.device)
        else:
            raise NotImplementedError

class SRCNN(nn.Module):
    """
    Parameters
    ----------
    upscale_factor : int
        Super-Resolution scale factor. Determines Low-Resolution downsampling.

    """
    def __init__(self, in_feats, upscale_factor,mean = 0,std =1):

        super(SRCNN, self).__init__()
        self.upsacle_factor = upscale_factor
        if isinstance(mean[0], list):
            self.mean = [torch.Tensor(mean[0]), torch.Tensor(mean[1])]
            self.std = [torch.Tensor(std[0]), torch.Tensor(std[1])]
            self.shiftmean_x = ShiftMean(self.mean[0],self.std[0])
            self.shiftmean_y = ShiftMean(self.mean[1],self.std[1])
        else:
            self.mean = torch.Tensor(mean)
            self.std = torch.Tensor(std)
            self.shiftmean_x = ShiftMean(self.mean,self.std)
            self.shiftmean_y = ShiftMean(self.mean,self.std)
        
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=in_feats, out_channels=64, kernel_size=9, stride=1, padding=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=in_feats, kernel_size=5, stride=1, padding=2)
        )

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input Low-Resolution image as tensor

        Returns
        -------
        torch.Tensor
            Super-Resolved image as tensor

        """
        # CNN extracting features
        x = self.shiftmean_x(x,"sub")

        # bicubic upsampling
        x = F.interpolate(x, scale_factor=[self.upsacle_factor,self.upsacle_factor], 
                                      mode='bicubic', align_corners=True)

        x = self.model(x)
        x = self.shiftmean_y(x,"add")
        return x