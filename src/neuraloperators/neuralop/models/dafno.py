import math
from collections import OrderedDict
from copy import Error, deepcopy
from functools import partial, partialmethod
from re import S

import numpy as np
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from numpy.lib.arraypad import pad
from timm.models.layers import DropPath, trunc_normal_
from torch import Tensor
from torch.nn.modules.container import Sequential
from torch.utils.checkpoint import checkpoint_sequential

from src.neuraloperators.neuralop.layers.afno_modules import (AFNO2D, Block,
                                                              Mlp, PatchEmbed,
                                                              PeriodicPad2d)

from .base_model import BaseModel

# RRDB layers are based on https://github.com/Lornatang/ESRGAN-PyTorch/blob/main/model.py
# AFNO model is based on the https://github.com/NVlabs/FourCastNet/blob/master/networks/afnonet.py

class RRDBNet(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 3,
            channels: int = 64,
            growth_channels: int = 32,
            num_rrdb: int = 23
    ) -> None:
        super(RRDBNet, self).__init__()

        # The first layer of convolutional layer.
        self.conv1 = nn.Conv2d(in_channels, channels, (3, 3), (1, 1), (1, 1))

        # Feature extraction backbone network.
        trunk = []
        for _ in range(num_rrdb):
            trunk.append(_ResidualResidualDenseBlock(channels, growth_channels))
        self.trunk = nn.Sequential(*trunk)

        # After the feature extraction network, reconnect a layer of convolutional blocks.
        self.conv2 = nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1))

        # Output layer.
        self.conv3 = nn.Conv2d(channels, out_channels, (3, 3), (1, 1), (1, 1))

        # Initialize all layer
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                module.weight.data *= 0.2
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    # The model should be defined in the Torch.script method.
    def _forward_impl(self, x: Tensor) -> Tensor:
        conv1 = self.conv1(x)
        x = self.trunk(conv1)
        x = self.conv2(x)
        x = torch.add(x, conv1)
        x = self.conv3(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


class _ResidualDenseBlock(nn.Module):
    """Achieves densely connected convolutional layers.
    `Densely Connected Convolutional Networks <https://arxiv.org/pdf/1608.06993v5.pdf>` paper.

    Args:
        channels (int): The number of channels in the input image.
        growth_channels (int): The number of channels that increase in each layer of convolution.
    """

    def __init__(self, channels: int, growth_channels: int) -> None:
        super(_ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels + growth_channels * 0, growth_channels, (3, 3), (1, 1), (1, 1))
        self.conv2 = nn.Conv2d(channels + growth_channels * 1, growth_channels, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(channels + growth_channels * 2, growth_channels, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(channels + growth_channels * 3, growth_channels, (3, 3), (1, 1), (1, 1))
        self.conv5 = nn.Conv2d(channels + growth_channels * 4, channels, (3, 3), (1, 1), (1, 1))

        self.leaky_relu = nn.LeakyReLU(0.2, True)
        self.identity = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out1 = self.leaky_relu(self.conv1(x))
        out2 = self.leaky_relu(self.conv2(torch.cat([x, out1], 1)))
        out3 = self.leaky_relu(self.conv3(torch.cat([x, out1, out2], 1)))
        out4 = self.leaky_relu(self.conv4(torch.cat([x, out1, out2, out3], 1)))
        out5 = self.identity(self.conv5(torch.cat([x, out1, out2, out3, out4], 1)))

        x = torch.mul(out5, 0.2)
        x = torch.add(x, identity)

        return x


class _ResidualResidualDenseBlock(nn.Module):
    """Multi-layer residual dense convolution block.

    Args:
        channels (int): The number of channels in the input image.
        growth_channels (int): The number of channels that increase in each layer of convolution.
    """

    def __init__(self, channels: int, growth_channels: int) -> None:
        super(_ResidualResidualDenseBlock, self).__init__()
        self.rdb1 = _ResidualDenseBlock(channels, growth_channels)
        self.rdb2 = _ResidualDenseBlock(channels, growth_channels)
        self.rdb3 = _ResidualDenseBlock(channels, growth_channels)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        x = self.rdb1(x)
        x = self.rdb2(x)
        x = self.rdb3(x)

        x = torch.mul(x, 0.2)
        x = torch.add(x, identity)

        return x

#------------------------------------------------------------------------------
        
class ShiftMean(nn.Module):
    # data: [t,c,h,w]
    def __init__(self, mean, std):
        super(ShiftMean, self).__init__()
        print(mean, type(mean))
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

class DAFNO(BaseModel, name='DAFNO'):
    def __init__(
            self,
            patch_size=16,
            in_channels=3,
            out_channels=1,
            embed_dim=768,
            depth=12,
            mlp_ratio=4.,
            drop_rate=0.,
            drop_path_rate=0.,
            num_blocks=16,
            sparsity_threshold=0.01,
            hard_thresholding_fraction=1.0,
            num_rrdb = 23,
            mean = None,
            std = None,
            upscale = 8,
            **kwargs
        ):
        super().__init__()
        
        self.patch_size = (patch_size, patch_size)
        self.in_chans = in_channels
        self.out_chans = out_channels 
        self.num_features = self.embed_dim = embed_dim
        self.num_blocks = num_blocks 
        self.num_rrdb = num_rrdb
        
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
        
        self.upsampling_factor = upscale
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        
        self.rrdb = RRDBNet(in_channels=self.in_chans, out_channels=self.in_chans, num_rrdb=self.num_rrdb)
        
        self.patch_embed = PatchEmbed(patch_size=self.patch_size, in_chans=self.in_chans, embed_dim=embed_dim)
        
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
            num_blocks=self.num_blocks, sparsity_threshold=sparsity_threshold, hard_thresholding_fraction=hard_thresholding_fraction) 
        for i in range(depth)])

        self.norm = norm_layer(embed_dim)

        self.head = nn.Linear(embed_dim, self.out_chans*self.patch_size[0]*self.patch_size[1], bias=False)

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward_features(self, x, h, w):
        B = x.shape[0]
        num_patches=h*w
        pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))
        trunc_normal_(pos_embed, std=.02)
        pos_embed = pos_embed.cuda() if torch.cuda.is_available() else pos_embed
        x = self.patch_embed(x)
        x = x + pos_embed
        x = self.pos_drop(x)
        x = x.reshape(B, h, w, self.embed_dim)
        for blk in self.blocks:
            x = blk(x)

        return x

    def forward(self, x, **kwargs):
        x = self.shiftmean_x(x,"sub")

        hr_size_h = x.shape[2]*self.upsampling_factor
        hr_size_w = x.shape[3]*self.upsampling_factor

        h = hr_size_h // self.patch_size[0]
        w = hr_size_w // self.patch_size[1]

        ## RRDB blocks + bicubic interpolation added before AFNO layers
        x = self.rrdb(x)
        x = F.interpolate(x, size=(hr_size_h, hr_size_w), mode='bicubic', align_corners=False)
    
        x = self.forward_features(x,h,w)
        x = self.head(x)
        x = rearrange(
            x,
            "b h w (p1 p2 c_out) -> b c_out (h p1) (w p2)",
            p1=self.patch_size[0],
            p2=self.patch_size[1],
            h=h,
            w=w
        )

        x = self.shiftmean_y(x,"add")
        
        return x
