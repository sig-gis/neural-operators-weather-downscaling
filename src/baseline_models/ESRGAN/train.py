'''the ESRGAN module is based on https://github.com/lizhuoq/ESRGAN-pytorch/'''

import os
import time
from glob import glob

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.adam import Adam
from torchvision.utils import save_image

from src.baseline_models.ESRGAN.model import (Discriminator, FeatureExtractor,
                                              GeneratorRRDB)


class Trainer:
    def __init__(self, config, train_loader, val_loader, mean, std, device, batch_size, scale_factor):
        self.device = device 
        self.n_epochs = config.num_epoch
        self.epoch = config.epoch
        self.in_channels = config.in_channels
        self.out_channels = config.out_channels
        self.data_loader = train_loader
        self.val_loader = val_loader
        self.checkpoint_dir = config.checkpoint_dir
        self.batch_size = batch_size
        self.scale_factor = scale_factor
        self.mean = mean
        self.std = std
        self.warmup_batches = config.warmup_batches
        self.lambda_adv = config.lambda_adv
        self.lambda_pixel = config.lambda_pixel 
        self.lambda_content = config.lambda_content
        self.hr_shape = (config.hr_height, config.hr_width)
        self.lr = config.lr
        self.generator = GeneratorRRDB(self.in_channels, self.out_channels, mean = self.mean, std = self.std, num_res_blocks=config.residual_blocks, scale_factor=self.scale_factor).to(device)
        self.discriminator = Discriminator(input_shape=(self.out_channels, *self.hr_shape)).to(device)
        self.feature_extractor = FeatureExtractor(self.out_channels).to(device)

        # set feature extractor to inference mode
        self.feature_extractor.eval()
        self.criterion_GAN = torch.nn.BCEWithLogitsLoss().to(device) 
        self.criterion_content = torch.nn.L1Loss().to(device)
        self.criterion_pixel = torch.nn.L1Loss().to(device)

        if not os.path.exists(self.checkpoint_dir):
            self.makedirs = os.makedirs(self.checkpoint_dir)


        if self.epoch != 0:
            self.load_model()

        self.optimizer_G = Adam(self.generator.parameters(), lr=self.lr, betas=(config.b1, config.b2))
                                       
        self.optimizer_D = Adam(self.discriminator.parameters(), lr=self.lr, betas=(config.b1, config.b2))

    def train(self):
        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

        generator_loss_list = []
        discriminator_loss_list = []
        start = time.time()
        for epoch in range(self.epoch, self.n_epochs):
            for i, (data, target) in enumerate(self.data_loader):

                batches_done = epoch * len(self.data_loader) + i

                imgs_lr = Variable(data.type(Tensor))
                imgs_hr = Variable(target.type(Tensor))

                valid = Variable(Tensor(np.ones((imgs_lr.size(0), *self.discriminator.output_shape))), requires_grad=False)
                fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *self.discriminator.output_shape))), requires_grad=False)

                 # ------------------
                #  Train Generator
                # ------------------   

                self.optimizer_G.zero_grad()

                gen_hr = self.generator(imgs_lr)

                loss_pixel = self.criterion_pixel(gen_hr, imgs_hr)

                if batches_done < self.warmup_batches:
                    loss_pixel.backward()
                    self.optimizer_G.step()
                    print(
                        '[Epoch %d/%d] [Batch %d/%d] [G pixel: %f]' % 
                        (epoch, self.n_epochs, i, len(self.data_loader), loss_pixel.item())
                    )
                    continue

                pred_real = self.discriminator(imgs_hr).detach()
                pred_fake = self.discriminator(gen_hr)

                loss_GAN = (self.criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), valid) + 
                            self.criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), fake)) / 2

                gen_features = self.feature_extractor(gen_hr)
                real_features = self.feature_extractor(imgs_hr).detach()
                loss_content = self.criterion_content(gen_features, real_features)

                loss_G = self.lambda_content * loss_content + self.lambda_adv * loss_GAN + self.lambda_pixel * loss_pixel

                loss_G.backward()
                self.optimizer_G.step()

                # --------------------
                # Train Discriminator  
                # --------------------  

                self.optimizer_D.zero_grad()

                pred_real = self.discriminator(imgs_hr)
                pred_fake = self.discriminator(gen_hr.detach()) 

                loss_real = self.criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), valid)
                loss_fake = self.criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), fake)

                loss_D = (loss_real + loss_fake) / 2

                loss_D.backward()
                self.optimizer_D.step()

                # -----------------  
                # Log Progress  
                # -----------------

                print(
                    '[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, content: %f, adv: %f, pixel: %f]' % 
                    (
                        epoch, 
                        self.n_epochs, 
                        i, 
                        len(self.data_loader), 
                        loss_D.item(), 
                        loss_G.item(), 
                        loss_content.item(), 
                        loss_GAN.item(), 
                        loss_pixel.item(), 
                    )
                )
                if i % 1000 == 0:
                    generator_loss_list.append(loss_G.item()) 
                    discriminator_loss_list.append(loss_D.item())

            torch.save(self.generator.state_dict(), os.path.join(self.checkpoint_dir, f"generator_{epoch}.pth"))
            torch.save(self.discriminator.state_dict(), os.path.join(self.checkpoint_dir, f"discriminator_{epoch}.pth"))
        end = time.time()
        print('The training time is: ', (end - start))

        return generator_loss_list, discriminator_loss_list

    def load_model(self):
        print(f"[*] Load model from {self.checkpoint_dir}")

        generator = glob(os.path.join(self.checkpoint_dir, f'generator_{self.epoch}.pth'))
        discriminator = glob(os.path.join(self.checkpoint_dir, f'discriminator_{self.epoch}.pth'))

        if not generator:
            print(f"[!] No checkpoint in epoch {self.epoch}")
            return

        self.generator.load_state_dict(torch.load(generator[0]))
        self.discriminator.load_state_dict(torch.load(discriminator[0]))
