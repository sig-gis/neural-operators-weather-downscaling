import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19


class ShiftMean(nn.Module):
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
        
class FeatureExtractor(nn.Module):
    def __init__(self, channels):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        features = list(vgg19_model.features)
        features[0] = nn.Conv2d(channels, 64, kernel_size=3, stride=1, padding=1) #Updating the first feature to work with channels other than 3
        # self.vgg19_54 = nn.Sequential(*list(vgg19_model.features.children())[:35])
        self.vgg19_54 = nn.Sequential(*features[:35])

    def forward(self, img):
        return self.vgg19_54(img)
    

class DenseResidualBlock(nn.Module):
    def __init__(self, filters, res_scale=0.2):
        '''
        filters = input channels 
        '''
        super(DenseResidualBlock, self).__init__()
        self.res_scale = res_scale

        def block(in_features, non_linearity=True):
            layers = [nn.Conv2d(in_features, filters, 3, 1, 1, bias=True)]
            if non_linearity:
                layers += [nn.LeakyReLU()]
            return nn.Sequential(*layers)
        
        self.b1 = block(in_features=1 * filters)
        self.b2 = block(in_features=2 * filters)
        self.b3 = block(in_features=3 * filters)
        self.b4 = block(in_features=4 * filters)
        self.b5 = block(in_features=5 * filters, non_linearity=False)
        self.blocks = [self.b1, self.b2, self.b3, self.b4, self.b5]

    def forward(self, x):
        inputs = x
        for block in self.blocks:
            out = block(inputs)
            inputs = torch.cat([inputs, out], 1)
        return out.mul(self.res_scale) + x


class ResidualInResidualDenseBlock(nn.Module):
    def __init__(self, filters, res_scale=0.2):
        super(ResidualInResidualDenseBlock, self).__init__()
        self.res_scale = res_scale
        self.dense_blocks = nn.Sequential(
            DenseResidualBlock(filters), DenseResidualBlock(filters), DenseResidualBlock(filters)
        )

    def forward(self, x):
        return self.dense_blocks(x).mul(self.res_scale) + x
    

class GeneratorRRDB(nn.Module):
    def __init__(self, in_channels, out_channels, mean = None, std = None, filters=64, num_res_blocks=16, scale_factor=4):
        super(GeneratorRRDB, self).__init__()
        if isinstance(mean[0], list):
            self.mean = [torch.Tensor(mean[0]), torch.Tensor(mean[1])]
            self.std = [torch.Tensor(std[0]), torch.Tensor(std[1])]
            self.shiftmean_x = ShiftMean(self.mean[0],self.std[0]) #self.shiftmean is different for input and output in case of ERA->WTK experiment
            self.shiftmean_y = ShiftMean(self.mean[1],self.std[1])
        else:
            self.mean = torch.Tensor(mean)
            self.std = torch.Tensor(std)
            self.shiftmean_x = ShiftMean(self.mean,self.std)
            self.shiftmean_y = ShiftMean(self.mean,self.std)

        # First Layer  
        self.conv1 = nn.Conv2d(in_channels, filters, kernel_size=3, stride=1, padding=1)
        # Residual block  
        self.res_blocks = nn.Sequential(*[ResidualInResidualDenseBlock(filters) for _ in range(num_res_blocks)])
        # second conv layer post residual blocks
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
        # Upsampling layers  
        self.scale = scale_factor
        upsample_layers = []
        if (self.scale & (self.scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(self.scale, 2))):
                upsample_layers += [
                    nn.Conv2d(filters, filters * 4, kernel_size=3, stride=1, padding=1), 
                    nn.LeakyReLU(), 
                    nn.PixelShuffle(upscale_factor=2), 
                ]
        else:
            upsample_layers+=[
                    nn.Conv2d(filters, filters * self.scale * self.scale, kernel_size=3, stride=1, padding=1), 
                    nn.LeakyReLU(), 
                    nn.PixelShuffle(upscale_factor=self.scale), 
            ]
        
        self.upsampling = nn.Sequential(*upsample_layers)
        # Final output block  
        self.conv3 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1), 
            nn.LeakyReLU(), 
            nn.Conv2d(filters, out_channels, kernel_size=3, stride=1, padding=1), 
        )

    def forward(self, x):
        x = self.shiftmean_x(x, mode='sub') # Normalize the input
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv3(out)
        x = self.shiftmean_y(x, mode='add') # De-normalize the output
        return out


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height / 2 ** 4), int(in_width/ 2 ** 4)
        self.output_shape = (1, patch_h, patch_w)

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        layers = []
        in_filters = in_channels 
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters
        
        layers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)