import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.neuraloperators.neuralop.layers.cno_modules import *

from .base_model import BaseModel

# RRDB layers are based on https://github.com/Lornatang/ESRGAN-PyTorch/blob/main/model.py
# CNO model is based on the https://github.com/camlab-ethz/ConvolutionalNeuralOperator/tree/main/CNO2d_original_version

class RRDBNet(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 3,
            channels: int = 64,
            growth_channels: int = 32,
            num_rrdb: int = 23,
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

#Depending on in_size, out_size, the CNOBlock can be:
#   -- (D) Block
#   -- (U) Block
#   -- (I) Block
class CNOBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 in_size,
                 out_size,
                 cutoff_den = 2.0001,
                 conv_kernel = 3,
                 filter_size = 6,
                 lrelu_upsampling = 2,
                 half_width_mult  = 0.8,
                 radial = False,
                 batch_norm = True,
                 activation = 'cno_lrelu'
                 ):
        super(CNOBlock, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_size  = in_size
        self.out_size = out_size
        self.conv_kernel = conv_kernel
        self.batch_norm = batch_norm
        
        #---------- Filter properties -----------
        self.citically_sampled = False #We use w_c = s/2.0001 --> NOT critically sampled

        if cutoff_den == 2.0:
            self.citically_sampled = True
        self.in_cutoff  = (self.in_size[0] / cutoff_den, self.in_size[1] / cutoff_den)
        self.out_cutoff = (self.out_size[0] / cutoff_den, self.out_size[1] / cutoff_den)
        
        self.in_halfwidth =  (half_width_mult*self.in_size[0] - self.in_size[0] / cutoff_den, half_width_mult*self.in_size[1] - self.in_size[1] / cutoff_den)
        self.out_halfwidth = (half_width_mult*self.out_size[0] - self.out_size[0] / cutoff_den, half_width_mult*self.out_size[1] - self.out_size[1] / cutoff_den)
        
        #-----------------------------------------

        # We apply Conv -> BN (optional) -> Activation
        # Up/Downsampling happens inside Activation
        
        pad = (self.conv_kernel-1)//2
        self.convolution = torch.nn.Conv2d(in_channels = self.in_channels, out_channels=self.out_channels, 
                                           kernel_size=self.conv_kernel, 
                                           padding = pad)
    
        if self.batch_norm:
            self.batch_norm  = nn.BatchNorm2d(self.out_channels)
        
        if activation == "cno_lrelu":
            self.activation  = LReLu(in_channels           = self.in_channels, #In _channels is not used in these settings
                                     out_channels          = self.out_channels,                   
                                     in_size               = self.in_size,                       
                                     out_size              = self.out_size,                       
                                     in_sampling_rate      = self.in_size,               
                                     out_sampling_rate     = self.out_size,             
                                     in_cutoff             = self.in_cutoff,                     
                                     out_cutoff            = self.out_cutoff,                  
                                     in_half_width         = self.in_halfwidth,                
                                     out_half_width        = self.out_halfwidth,              
                                     filter_size           = filter_size,       
                                     lrelu_upsampling      = lrelu_upsampling,
                                     is_critically_sampled = self.citically_sampled,
                                     use_radial_filters    = False)
        elif activation == "lrelu":
            self.activation  = LReLu_regular(in_channels           = self.in_channels, #In _channels is not used in these settings
                                             out_channels          = self.out_channels,                   
                                             in_size               = self.in_size,                       
                                             out_size              = self.out_size,                       
                                             in_sampling_rate      = self.in_size,               
                                             out_sampling_rate     = self.out_size)
        else:
            raise ValueError("Please specify different activation function")
        
        
    def forward(self, x):
        x = self.convolution(x)
        if self.batch_norm:
            x = self.batch_norm(x)
        x = self.activation(x)
        return x

#------------------------------------------------------------------------------

# Contains CNOBlock -> Convolution -> BN

class LiftProjectBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 in_size,
                 out_size,
                 latent_dim = 64,
                 cutoff_den = 2.0001,
                 conv_kernel = 3,
                 filter_size = 6,
                 lrelu_upsampling = 2,
                 half_width_mult  = 0.8,
                 radial = False,
                 batch_norm = True,
                 activation = 'cno_lrelu'
                 ):
        super(LiftProjectBlock, self).__init__()
    
        self.inter_CNOBlock = CNOBlock(in_channels = in_channels,
                                    out_channels = latent_dim,
                                    in_size = in_size,
                                    out_size = out_size,
                                    cutoff_den = cutoff_den,
                                    conv_kernel = conv_kernel,
                                    filter_size = filter_size,
                                    lrelu_upsampling = lrelu_upsampling,
                                    half_width_mult  = half_width_mult,
                                    radial = radial,
                                    batch_norm = batch_norm,
                                    activation = activation)
        
        pad = (conv_kernel-1)//2
        self.convolution = torch.nn.Conv2d(in_channels = latent_dim, out_channels=out_channels, 
                                           kernel_size=conv_kernel, stride = 1, 
                                           padding = pad)
        
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.batch_norm  = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.inter_CNOBlock(x)
        
        x = self.convolution(x)
        if self.batch_norm:
            x = self.batch_norm(x)
        return x
        
#------------------------------------------------------------------------------

# Residual Block containts:
    # Convolution -> BN -> Activation -> Convolution -> BN -> SKIP CONNECTION

class ResidualBlock(nn.Module):
    def __init__(self,
                 channels,
                 size,
                 cutoff_den = 2.0001,
                 conv_kernel = 3,
                 filter_size = 6,
                 lrelu_upsampling = 2,
                 half_width_mult  = 0.8,
                 radial = False,
                 batch_norm = True,
                 activation = 'cno_lrelu'
                 ):
        super(ResidualBlock, self).__init__()

        self.channels = channels
        self.size  = size
        self.conv_kernel = conv_kernel
        self.batch_norm = batch_norm

        #---------- Filter properties -----------
        self.citically_sampled = False #We use w_c = s/2.0001 --> NOT critically sampled

        if cutoff_den == 2.0:
            self.citically_sampled = True
        self.cutoff  = (self.size[0] / cutoff_den , self.size[1] / cutoff_den)   
        self.halfwidth = (half_width_mult*self.size[0] - self.size[0] / cutoff_den, half_width_mult*self.size[1] - self.size[1] / cutoff_den)
        
        #-----------------------------------------
        
        pad = (self.conv_kernel-1)//2
        self.convolution1 = torch.nn.Conv2d(in_channels = self.channels, out_channels=self.channels, 
                                           kernel_size=self.conv_kernel, stride = 1, 
                                           padding = pad)
        self.convolution2 = torch.nn.Conv2d(in_channels = self.channels, out_channels=self.channels, 
                                           kernel_size=self.conv_kernel, stride = 1, 
                                           padding = pad)
        
        if self.batch_norm:
            self.batch_norm1  = nn.BatchNorm2d(self.channels)
            self.batch_norm2  = nn.BatchNorm2d(self.channels)
        
        if activation == "cno_lrelu":

            self.activation  = LReLu(in_channels           = self.channels, #In _channels is not used in these settings
                                     out_channels          = self.channels,                   
                                     in_size               = self.size,                       
                                     out_size              = self.size,                       
                                     in_sampling_rate      = self.size,               
                                     out_sampling_rate     = self.size,             
                                     in_cutoff             = self.cutoff,                     
                                     out_cutoff            = self.cutoff,                  
                                     in_half_width         = self.halfwidth,                
                                     out_half_width        = self.halfwidth,              
                                     filter_size           = filter_size,       
                                     lrelu_upsampling      = lrelu_upsampling,
                                     is_critically_sampled = self.citically_sampled,
                                     use_radial_filters    = False)
        elif activation == "lrelu":

            self.activation = LReLu_regular(in_channels           = self.channels, #In _channels is not used in these settings
                                            out_channels          = self.channels,                   
                                            in_size               = self.size,                       
                                            out_size              = self.size,                       
                                            in_sampling_rate      = self.size,               
                                            out_sampling_rate     = self.size)
        else:
            raise ValueError("Please specify different activation function")
            

    def forward(self, x):
        out = self.convolution1(x)
        if self.batch_norm:
            out = self.batch_norm1(out)
        out = self.activation(out)
        out = self.convolution2(out)
        if self.batch_norm:
            out = self.batch_norm2(out)
        return x + out

def center_crop_like(source, target):
    """
    Useful when concatenating skip connections where size mismatches may occur.
    """
    _, _, h, w = source.size()
    _, _, th, tw = target.size()
    dh = (h - th) // 2
    dw = (w - tw) // 2
    return source[:, :, dh:dh+th, dw:dw+tw]
    
class DCNO(BaseModel, name='DCNO'):
    def __init__(self,  
                 in_channels = 3,                    # Number of input channels.
                 in_size_h = 128,                   # Input spatial size
                 in_size_w = 128,
                 N_layers = 3,                  # Number of (D) or (U) blocks in the network
                 N_res = 1,                 # Number of (R) blocks per level (except the neck)
                 N_res_neck = 6,            # Number of (R) blocks in the neck
                 channel_multiplier = 32,   # How the number of channels evolve?
                 conv_kernel=3,             # Size of all the kernels
                 cutoff_den = 2.0001,       # Filter property 1.
                 filter_size=6,             # Filter property 2.
                 lrelu_upsampling = 2,      # Filter property 3.
                 half_width_mult  = 0.8,    # Filter property 4.
                 radial = False,            # Filter property 5. Is filter radial?
                 batch_norm = True,         # Add BN? We do not add BN in lifting/projection layer
                 out_channels = 1,               # Target dimension
                 out_size = 1,              # If out_size is 1, Then out_size = in_size. Else must be int
                 expand_input = False,      # Start with original in_size, or expand it (pad zeros in the spectrum)
                 latent_lift_proj_dim = 64, # Intermediate latent dimension in the lifting/projection layer
                 add_inv = True,            # Add invariant block (I) after the intermediate connections?
                 activation = 'cno_lrelu',   # Activation function can be 'cno_lrelu' or 'lrelu'
                 num_rrdb = 23,             # Number of RRDB blocks in the network
                mean = None,                # Mean and std of data for normalization
                std = None,
                upscale = 8,                # Upsampling factor
                **kwargs
                ):
        
        super(DCNO, self).__init__()

        ###################### Define the parameters & specifications #################################################
        # Number of (D) & (U) Blocks
        self.N_layers = int(N_layers)
        
        # Input is lifted to the half on channel_multiplier dimension
        self.lift_dim = channel_multiplier//2         
        self.out_channels  = out_channels
        
        #Should we add invariant layers in the decoder?
        self.add_inv = add_inv
        
        # The growth of the channels : d_e parametee
        self.channel_multiplier = channel_multiplier        
        
        # Is the filter radial? We always use NOT radial
        if radial ==0:
            self.radial = False
        else:
            self.radial = True

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
        in_size = (in_size_h,in_size_w)
        self.in_size=in_size

        self.rrdb = RRDBNet(in_channels=in_channels, out_channels=in_channels, num_rrdb=self.num_rrdb)
        
        ###################### Define evolution of the number features ################################################

        # How the features in Encoder evolve (number of features)
        self.encoder_features = [self.lift_dim]
        for i in range(self.N_layers):
            self.encoder_features.append(2 ** i *   self.channel_multiplier)
        
        # How the features in Decoder evolve (number of features)
        self.decoder_features_in = self.encoder_features[1:]
        self.decoder_features_in.reverse()
        self.decoder_features_out = self.encoder_features[:-1]
        self.decoder_features_out.reverse()

        for i in range(1, self.N_layers):
            self.decoder_features_in[i] = 2*self.decoder_features_in[i] #Pad the outputs of the resnets
        
        self.inv_features = self.decoder_features_in
        self.inv_features.append(self.encoder_features[0] + self.decoder_features_out[-1])
        
        ###################### Define evolution of sampling rates #####################################################
        
        if not expand_input:
            latent_size = in_size # No change in in_size
        else:
            down_exponent = 2 ** N_layers
            # latent_size = in_size - (in_size % down_exponent) + down_exponent # Jump from 64 to 72, for example
            latent_size = (in_size[0] - (in_size[0] % down_exponent) + down_exponent, in_size[1] - (in_size[1] % down_exponent) + down_exponent)
        
        #Are inputs and outputs of the same size? If not, how should the size of the decoder evolve?
        if out_size == 1:
            latent_size_out = latent_size
        else:
            if not expand_input:
                latent_size_out = out_size # No change in in_size
            else:
                down_exponent = 2 ** N_layers
                # latent_size_out = out_size - (out_size % down_exponent) + down_exponent # Jump from 64 to 72, for example
                latent_size_out = (out_size[0] - (out_size[0] % down_exponent) + down_exponent, out_size[1] - (out_size[1] % down_exponent) + down_exponent)
        
        self.encoder_sizes = []
        self.decoder_sizes = []
        for i in range(self.N_layers + 1):
            self.encoder_sizes.append((latent_size[0] // 2 ** i, latent_size[1] // 2 ** i))
            self.decoder_sizes.append((latent_size_out[0] // 2 ** (self.N_layers - i), latent_size_out[1] // 2 ** (self.N_layers - i)))
            print("encoder decoder sizes",self.encoder_sizes[i], self.decoder_sizes[i])
        
        
        
        ###################### Define Projection & Lift ##############################################################
    
        self.lift = LiftProjectBlock(in_channels  = in_channels,
                                     out_channels = self.encoder_features[0],
                                     in_size      = in_size,
                                     out_size     = self.encoder_sizes[0],
                                     latent_dim   = latent_lift_proj_dim,
                                     cutoff_den   = cutoff_den,
                                     conv_kernel  = conv_kernel,
                                     filter_size  = filter_size,
                                     lrelu_upsampling  = lrelu_upsampling,
                                     half_width_mult   = half_width_mult,
                                     radial            = radial,
                                     batch_norm        = False,
                                     activation = activation)
        _out_size = out_size
        if out_size == 1:
            _out_size = in_size
            
        self.project = LiftProjectBlock(in_channels  = self.encoder_features[0] + self.decoder_features_out[-1],
                                        out_channels = out_channels,
                                        in_size      = self.decoder_sizes[-1],
                                        out_size     = _out_size,
                                        latent_dim   = latent_lift_proj_dim,
                                        cutoff_den   = cutoff_den,
                                        conv_kernel  = conv_kernel,
                                        filter_size  = filter_size,
                                        lrelu_upsampling  = lrelu_upsampling,
                                        half_width_mult   = half_width_mult,
                                        radial            = radial,
                                        batch_norm        = False,
                                        activation = activation) 

        ###################### Define U & D blocks ###################################################################

        self.encoder         = nn.ModuleList([(CNOBlock(in_channels  = self.encoder_features[i],
                                                        out_channels = self.encoder_features[i+1],
                                                        in_size      = self.encoder_sizes[i],
                                                        out_size     = self.encoder_sizes[i+1],
                                                        cutoff_den   = cutoff_den,
                                                        conv_kernel  = conv_kernel,
                                                        filter_size  = filter_size,
                                                        lrelu_upsampling = lrelu_upsampling,
                                                        half_width_mult  = half_width_mult,
                                                        radial = radial,
                                                        batch_norm = batch_norm,
                                                        activation = activation))                                  
                                               for i in range(self.N_layers)])
        
        # After the ResNets are executed, the sizes of encoder and decoder might not match (if out_size>1)
        # We must ensure that the sizes are the same, by aplying CNO Blocks
        self.ED_expansion     = nn.ModuleList([(CNOBlock(in_channels = self.encoder_features[i],
                                                        out_channels = self.encoder_features[i],
                                                        in_size      = self.encoder_sizes[i],
                                                        out_size     = self.decoder_sizes[self.N_layers - i],
                                                        cutoff_den   = cutoff_den,
                                                        conv_kernel  = conv_kernel,
                                                        filter_size  = filter_size,
                                                        lrelu_upsampling = lrelu_upsampling,
                                                        half_width_mult  = half_width_mult,
                                                        radial = radial,
                                                        batch_norm = batch_norm,
                                                        activation = activation))                                  
                                               for i in range(self.N_layers + 1)])
        
        self.decoder         = nn.ModuleList([(CNOBlock(in_channels  = self.decoder_features_in[i],
                                                        out_channels = self.decoder_features_out[i],
                                                        in_size      = self.decoder_sizes[i],
                                                        out_size     = self.decoder_sizes[i+1],
                                                        cutoff_den   = cutoff_den,
                                                        conv_kernel  = conv_kernel,
                                                        filter_size  = filter_size,
                                                        lrelu_upsampling = lrelu_upsampling,
                                                        half_width_mult  = half_width_mult,
                                                        radial = radial,
                                                        batch_norm = batch_norm,
                                                        activation = activation))                                  
                                               for i in range(self.N_layers)])
        
        
        self.decoder_inv    = nn.ModuleList([(CNOBlock(in_channels  =  self.inv_features[i],
                                                        out_channels = self.inv_features[i],
                                                        in_size      = self.decoder_sizes[i],
                                                        out_size     = self.decoder_sizes[i],
                                                        cutoff_den   = cutoff_den,
                                                        conv_kernel  = conv_kernel,
                                                        filter_size  = filter_size,
                                                        lrelu_upsampling = lrelu_upsampling,
                                                        half_width_mult  = half_width_mult,
                                                        radial = radial,
                                                        batch_norm = batch_norm,
                                                        activation = activation))                                  
                                               for i in range(self.N_layers + 1)])
        
        
        ####################### Define ResNets Blocks ################################################################

        # Here, we define ResNet Blocks. 
        # We also define the BatchNorm layers applied BEFORE the ResNet blocks 
        
        # Operator UNet:
        # Outputs of the middle networks are patched (or padded) to corresponding sets of feature maps in the decoder 

        self.res_nets = []
        self.N_res = int(N_res)
        self.N_res_neck = int(N_res_neck)

        # Define the ResNet blocks & BatchNorm
        for l in range(self.N_layers):
            for i in range(self.N_res):
                self.res_nets.append(ResidualBlock(channels = self.encoder_features[l],
                                                   size     = self.encoder_sizes[l],
                                                   cutoff_den = cutoff_den,
                                                   conv_kernel = conv_kernel,
                                                   filter_size = filter_size,
                                                   lrelu_upsampling = lrelu_upsampling,
                                                   half_width_mult  = half_width_mult,
                                                   radial = radial,
                                                   batch_norm = batch_norm,
                                                   activation = activation))
        for i in range(self.N_res_neck):
            self.res_nets.append(ResidualBlock(channels = self.encoder_features[self.N_layers],
                                               size     = self.encoder_sizes[self.N_layers],
                                               cutoff_den = cutoff_den,
                                               conv_kernel = conv_kernel,
                                               filter_size = filter_size,
                                               lrelu_upsampling = lrelu_upsampling,
                                               half_width_mult  = half_width_mult,
                                               radial = radial,
                                               batch_norm = batch_norm,
                                               activation = activation))
        
        self.res_nets = torch.nn.Sequential(*self.res_nets)    


    def forward(self, x, **kwargs):

        x = self.shiftmean_x(x,"sub")
        
        hr_size_h = x.shape[2]*self.upsampling_factor
        hr_size_w = x.shape[3]*self.upsampling_factor

        ## RRDB blocks + bicubic interpolation added before CNO layers
        x = self.rrdb(x)
    
        x = F.interpolate(x, size=(hr_size_h, hr_size_w), mode='bicubic', align_corners=False)
       
        # Execute Lift ---------------------------------------------------------
        x = self.lift(x)
        skip = []

        # Execute Encoder -----------------------------------------------------
        for i in range(self.N_layers):
            #Apply ResNet & save the result
            y = x
            for j in range(self.N_res):
                y = self.res_nets[i*self.N_res + j](y)
            skip.append(y)
            
            # Apply (D) block
            x = self.encoder[i](x)   
        #----------------------------------------------------------------------
        
        # Apply the deepest ResNet (bottle neck)
        for j in range(self.N_res_neck):
            x = self.res_nets[-j-1](x)

        # Execute Decoder -----------------------------------------------------
        for i in range(self.N_layers):
            
            # Apply (I) block (ED_expansion) & cat if needed
            if i == 0:
                x = self.ED_expansion[self.N_layers - i](x) #BottleNeck : no cat
            else:
                skip_i = self.ED_expansion[self.N_layers - i](skip[-i])
                skip_i = center_crop_like(skip_i, x)  
                x = torch.cat((x, skip_i), 1)
                # x = torch.cat((x, self.ED_expansion[self.N_layers - i](skip[-i])),1)
            
            if self.add_inv:
                x = self.decoder_inv[i](x)
            # Apply (U) block
            x = self.decoder[i](x)

        # Cat & Execute Projetion ---------------------------------------------
        
        # x = torch.cat((x, self.ED_expansion[0](skip[0])),1)
        skip_0 = self.ED_expansion[0](skip[0])
        skip_0 = center_crop_like(skip_0, x)  
        x = torch.cat((x, skip_0), 1)
        x = self.project(x)
        x = self.shiftmean_y(x,"add")
        del skip
        del y
        
        return x


    def get_n_params(self):
        pp = 0
        
        for p in list(self.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        return pp

    def print_size(self):
        nparams = 0
        nbytes = 0

        for param in self.parameters():
            nparams += param.numel()
            nbytes += param.data.element_size() * param.numel()

        print(f'Total number of model parameters: {nparams} (~{format_tensor_size(nbytes)})')

        return nparams
    
   

