import os
from glob import glob

import torch
import torch.nn as nn

from src.baseline_models.ESRGAN.model import (Discriminator, FeatureExtractor,
                                              GeneratorRRDB)


class Tester:
    def __init__(self, config,mean, std, device, scale_factor):
        self.device = device 
        self.n_epochs = config.num_epoch
        self.epoch = config.epoch
        self.in_channels = config.in_channels
        self.out_channels = config.out_channels
        self.checkpoint_dir = config.checkpoint_dir
        self.scale_factor = scale_factor
        self.mean = mean
        self.std = std
        self.residual_blocks = config.residual_blocks

    def build_model(self):
        self.generator = GeneratorRRDB(self.in_channels, self.out_channels, mean = self.mean, std = self.std, num_res_blocks=self.residual_blocks, scale_factor=self.scale_factor).to(self.device)
        self.load_model()
        return self.generator

    def load_model(self):
        print(f"[*] Load model from {self.checkpoint_dir}")
        generator = glob(os.path.join(self.checkpoint_dir, f'generator_{self.n_epochs - 1}.pth'))
        self.generator.load_state_dict(torch.load(generator[0]))