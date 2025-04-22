from functools import partialmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..layers.fno_block import FNOBlocks
from ..layers.local_fno_block import LocalFNOBlocks
from ..layers.mlp import MLP
from ..layers.padding import DomainPadding
from ..layers.spectral_convolution import SpectralConv
from ..layers.spherical_convolution import SphericalConv
from .base_model import BaseModel

# RRDB layers are based on https://github.com/Lornatang/ESRGAN-PyTorch/blob/main/model.py
# FNO model is a part of the https://github.com/neuraloperator/neuraloperator 

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

class DFNO(BaseModel, name='DFNO'):
    """Downscaling Fourier Neural Operator

    Parameters
    ----------
    n_modes_width : int
        number of modes to keep in Fourier Layer, along the width
    n_modes_height : int
        number of Fourier modes to keep along the height
    hidden_channels : int
        width of the FNO (i.e. number of channels)
    in_channels : int, optional
        Number of input channels, by default 3
    out_channels : int, optional
        Number of output channels, by default 1
    lifting_channels : int, optional
        number of hidden channels of the lifting block of the FNO, by default 256
    projection_channels : int, optional
        number of hidden channels of the projection block of the FNO, by default 256
    n_layers : int, optional
        Number of Fourier Layers, by default 4
    max_n_modes : None or int tuple, default is None
        * If not None, this allows to incrementally increase the number of
          modes in Fourier domain during training. Has to verify n <= N
          for (n, m) in zip(max_n_modes, n_modes).

        * If None, all the n_modes are used.

        This can be updated dynamically during training.
    fno_block_precision : str {'full', 'half', 'mixed'}
        if 'full', the FNO Block runs in full precision
        if 'half', the FFT, contraction, and inverse FFT run in half precision
        if 'mixed', the contraction and inverse FFT run in half precision
    stabilizer : str {'tanh'} or None, optional
        By default None, otherwise tanh is used before FFT in the FNO block
    use_mlp : bool, optional
        Whether to use an MLP layer after each FNO block, by default False
    mlp_dropout : float , optional
        droupout parameter of MLP layer, by default 0
    mlp_expansion : float, optional
        expansion parameter of MLP layer, by default 0.5
    non_linearity : nn.Module, optional
        Non-Linearity module to use, by default F.gelu
    norm : F.module, optional
        Normalization layer to use, by default None
    preactivation : bool, default is False
        if True, use resnet-style preactivation
    fno_skip : {'linear', 'identity', 'soft-gating'}, optional
        Type of skip connection to use in fno, by default 'linear'
    mlp_skip : {'linear', 'identity', 'soft-gating'}, optional
        Type of skip connection to use in mlp, by default 'soft-gating'
    separable : bool, default is False
        if True, use a depthwise separable spectral convolution
    factorization : str or None, {'tucker', 'cp', 'tt'}
        Tensor factorization of the parameters weight to use, by default None.
        * If None, a dense tensor parametrizes the Spectral convolutions
        * Otherwise, the specified tensor factorization is used.
    joint_factorization : bool, optional
        Whether all the Fourier Layers should be parametrized by a single tensor
        (vs one per layer), by default False
    rank : float or rank, optional
        Rank of the tensor factorization of the Fourier weights, by default 1.0
    fixed_rank_modes : bool, optional
        Modes to not factorize, by default False
    implementation : {'factorized', 'reconstructed'}, optional, default is 'factorized'
        If factorization is not None, forward mode to use::
        * `reconstructed` : the full weight tensor is reconstructed from the
          factorization and used for the forward pass
        * `factorized` : the input is directly contracted with the factors of
          the decomposition
    decomposition_kwargs : dict, optional, default is {}
        Optionaly additional parameters to pass to the tensor decomposition
    domain_padding : None or float, optional
        If not None, percentage of padding to use, by default None
    domain_padding_mode : {'symmetric', 'one-sided'}, optional
        How to perform domain padding, by default 'one-sided'
    fft_norm : str, optional
        by default 'forward'
    local : bool, default is False
        if True , use the local FNO blocks
    
    Refer to get_model() in base_model.py, where mean, std, and upscale are added to the model arguments.
    """
    def __init__(
        self,
        n_modes_height,
        n_modes_width,
        hidden_channels,
        in_channels=3,
        out_channels=1,
        lifting_channels=256,
        projection_channels=256,
        n_layers=4,
        output_scaling_factor=None,
        max_n_modes=None,
        fno_block_precision="full",
        use_mlp=False,
        mlp_dropout=0,
        mlp_expansion=0.5,
        non_linearity=F.gelu,
        stabilizer=None,
        norm=None,
        preactivation=False,
        fno_skip="linear",
        mlp_skip="soft-gating",
        separable=False,
        factorization=None,
        rank=1.0,
        joint_factorization=False,
        fixed_rank_modes=False,
        implementation="factorized",
        decomposition_kwargs=dict(),
        domain_padding=None,
        domain_padding_mode="one-sided",
        fft_norm="forward",
        SpectralConv=SpectralConv,
        num_rrdb = 23,
        mean = None,
        std = None,
        upscale = 8,
        local=False,
        **kwargs
    ):
        super().__init__()
        n_modes = (n_modes_height,n_modes_width)
        self.n_dim = len(n_modes)

        self._n_modes = n_modes
        self.hidden_channels = hidden_channels
        self.lifting_channels = lifting_channels
        self.projection_channels = projection_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.joint_factorization = joint_factorization
        self.non_linearity = non_linearity
        self.rank = rank
        self.factorization = factorization
        self.fixed_rank_modes = fixed_rank_modes
        self.decomposition_kwargs = decomposition_kwargs
        self.fno_skip = (fno_skip,)
        self.mlp_skip = (mlp_skip,)
        self.fft_norm = fft_norm
        self.implementation = implementation
        self.separable = separable
        self.preactivation = preactivation
        self.fno_block_precision = fno_block_precision
        self.num_rrdb = num_rrdb
        self.n_modes_height = n_modes_height
        self.n_modes_width = n_modes_width
        self.local = local
        
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

        self.rrdb = RRDBNet(in_channels=self.in_channels, out_channels=self.in_channels, num_rrdb=self.num_rrdb)  
         
        if domain_padding is not None and (
            (isinstance(domain_padding, list) and sum(domain_padding) > 0)
            or (isinstance(domain_padding, (float, int)) and domain_padding > 0)
        ):
            self.domain_padding = DomainPadding(
                domain_padding=domain_padding,
                padding_mode=domain_padding_mode,
                output_scaling_factor=output_scaling_factor,
            )
        else:
            self.domain_padding = None

        self.domain_padding_mode = domain_padding_mode

        if output_scaling_factor is not None and not joint_factorization:
            if isinstance(output_scaling_factor, (float, int)):
                output_scaling_factor = [output_scaling_factor] * self.n_layers
        self.output_scaling_factor = output_scaling_factor

        
        if self.local is False:
            self.fno_blocks = FNOBlocks(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                n_modes=self.n_modes,
                output_scaling_factor=output_scaling_factor,
                use_mlp=use_mlp,
                mlp_dropout=mlp_dropout,
                mlp_expansion=mlp_expansion,
                non_linearity=non_linearity,
                stabilizer=stabilizer,
                norm=norm,
                preactivation=preactivation,
                fno_skip=fno_skip,
                mlp_skip=mlp_skip,
                max_n_modes=max_n_modes,
                fno_block_precision=fno_block_precision,
                rank=rank,
                fft_norm=fft_norm,
                fixed_rank_modes=fixed_rank_modes,
                implementation=implementation,
                separable=separable,
                factorization=factorization,
                decomposition_kwargs=decomposition_kwargs,
                joint_factorization=joint_factorization,
                SpectralConv=SpectralConv,
                n_layers=n_layers,
                **kwargs
            )
        else:
            print("choose local blocks")
            diff_layers=[True for _ in range(n_layers)]
            self.fno_blocks = LocalFNOBlocks(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                n_modes=self.n_modes,
                output_scaling_factor=output_scaling_factor,
                use_mlp=use_mlp,
                mlp_dropout=mlp_dropout,
                mlp_expansion=mlp_expansion,
                non_linearity=non_linearity,
                stabilizer=stabilizer,
                norm=norm,
                preactivation=preactivation,
                fno_skip=fno_skip,
                mlp_skip=mlp_skip,
                max_n_modes=max_n_modes,
                fno_block_precision=fno_block_precision,
                rank=rank,
                fft_norm=fft_norm,
                fixed_rank_modes=fixed_rank_modes,
                implementation=implementation,
                separable=separable,
                factorization=factorization,
                decomposition_kwargs=decomposition_kwargs,
                joint_factorization=joint_factorization,
                SpectralConv=SpectralConv,
                n_layers=n_layers,
                diff_layers=diff_layers,
                **kwargs
            )

        # if lifting_channels is passed, make lifting an MLP
        # with a hidden layer of size lifting_channels
        if self.lifting_channels:
            self.lifting = MLP(
                in_channels=in_channels,
                out_channels=self.hidden_channels,
                hidden_channels=self.lifting_channels,
                n_layers=2,
                n_dim=self.n_dim,
            )
        # otherwise, make it a linear layer
        else:
            self.lifting = MLP(
                in_channels=in_channels,
                out_channels=self.hidden_channels,
                hidden_channels=self.hidden_channels,
                n_layers=1,
                n_dim=self.n_dim,
            )
        self.projection = MLP(
            in_channels=self.hidden_channels,
            out_channels=out_channels,
            hidden_channels=self.projection_channels,
            n_layers=2,
            n_dim=self.n_dim,
            non_linearity=non_linearity,
        )


    def forward(self, x, output_shape=None, **kwargs):
        """DFNO's forward pass

        Parameters
        ----------
        x : tensor
            input tensor
        output_shape : {tuple, tuple list, None}, default is None
            Gives the option of specifying the exact output shape for odd shaped inputs.
            * If None, don't specify an output shape
            * If tuple, specifies the output-shape of the **last** FNO Block
            * If tuple list, specifies the exact output-shape of each FNO Block
        """

        if output_shape is None:
            output_shape = [None]*self.n_layers
        elif isinstance(output_shape, tuple):
            output_shape = [None]*(self.n_layers - 1) + [output_shape]

        x = self.shiftmean_x(x,"sub")
        
        hr_size_h = x.shape[2]*self.upsampling_factor
        hr_size_w = x.shape[3]*self.upsampling_factor

        ##RRDB blocks + bicubic interpolation added before FNO layers
        x = self.rrdb(x)

        x = F.interpolate(x, size=(hr_size_h, hr_size_w), mode='bicubic', align_corners=False)

        ##FNO starts
        x = self.lifting(x)

        if self.domain_padding is not None:
            x = self.domain_padding.pad(x)

        for layer_idx in range(self.n_layers):
            x = self.fno_blocks(x, layer_idx, output_shape=output_shape[layer_idx])

        if self.domain_padding is not None:
            x = self.domain_padding.unpad(x)

        x = self.projection(x)

        x = self.shiftmean_y(x,"add")

        return x

    @property
    def n_modes(self):
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes):
        self.fno_blocks.n_modes = n_modes
        self._n_modes = n_modes

