import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..layers.fno_block import FNOBlocks
from ..layers.local_fno_block import LocalFNOBlocks
from ..layers.mlp import MLP
from ..layers.padding import DomainPadding
from ..layers.resample import resample
from ..layers.skip_connections import skip_connection
from ..layers.spectral_convolution import SpectralConv
from .base_model import BaseModel

# RRDB layers are based on https://github.com/Lornatang/ESRGAN-PyTorch/blob/main/model.py
# UNO model is a also part of the https://github.com/neuraloperator/neuraloperator 

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

class DUNO(BaseModel, name='DUNO'):
    """Downscaling U-Shaped Neural Operator 

    Parameters
    ----------
    in_channels : int, optional
        Number of input channels
    out_channels : int, optional
        Number of output channels
    hidden_channels : int
        initial width of the UNO (i.e. number of channels)
    lifting_channels : int, optional
        number of hidden channels of the lifting block of the FNO, by default 256
    projection_channels : int, optional
        number of hidden channels of the projection block of the FNO, by default 256
    n_layers : int, optional
        Number of Fourier Layers, by default 4
    uno_out_channels: list
        Number of output channel of each Fourier Layers.
        Eaxmple: For a Five layer UNO uno_out_channels can be [32,64,64,64,32]
    uno_n_modes: list
        Number of Fourier Modes to use in integral operation of each Fourier Layers (along each dimension).
        Example: For a five layer UNO with 2D input the uno_n_modes can be: [[5,5],[5,5],[5,5],[5,5],[5,5]]
    uno_scalings: list
        Scaling Factors for each Fourier Layers
        Example: For a five layer UNO with 2D input, the uno_scalings can be : [[1.0,1.0],[0.5,0.5],[1,1],[1,1],[2,2]]
    horizontal_skips_map: Dict, optional
                    a map {...., b: a, ....} denoting horizontal skip connection from a-th layer to
                    b-th layer. If None default skip connection is applied.
                    Example: For a 5 layer UNO architecture, the skip connections can be
                    horizontal_skips_map ={4:0,3:1}

    incremental_n_modes : None or int tuple, default is None
        * If not None, this allows to incrementally increase the number of modes in Fourier domain
          during training. Has to verify n <= N for (n, m) in zip(incremental_n_modes, n_modes).

        * If None, all the n_modes are used.

        This can be updated dynamically during training.
    use_mlp : bool, optional
        Whether to use an MLP layer after each FNO block, by default False
    mlp : dict, optional
        Parameters of the MLP, by default None
        {'expansion': float, 'dropout': float}
    non_linearity : nn.Module, optional
        Non-Linearity module to use, by default F.gelu
    norm : F.module, optional
        Normalization layer to use, by default None
    preactivation : bool, default is False
        if True, use resnet-style preactivation
    skip : {'linear', 'identity', 'soft-gating'}, optional
        Type of skip connection to use, by default 'soft-gating'
    separable : bool, default is False
        if True, use a depthwise separable spectral convolution
    factorization : str or None, {'tucker', 'cp', 'tt'}
        Tensor factorization of the parameters weight to use, by default None.
        * If None, a dense tensor parametrizes the Spectral convolutions
        * Otherwise, the specified tensor factorization is used.
    joint_factorization : bool, optional
        Whether all the Fourier Layers should be parametrized by a single tensor (vs one per layer), by default False
    rank : float or rank, optional
        Rank of the tensor factorization of the Fourier weights, by default 1.0
    fixed_rank_modes : bool, optional
        Modes to not factorize, by default False
    implementation : {'factorized', 'reconstructed'}, optional, default is 'factorized'
        If factorization is not None, forward mode to use::
        * `reconstructed` : the full weight tensor is reconstructed from the factorization and used for the forward pass
        * `factorized` : the input is directly contracted with the factors of the decomposition
    decomposition_kwargs : dict, optional, default is {}
        Optionaly additional parameters to pass to the tensor decomposition
    domain_padding : None or float, optional
        If not None, percentage of padding to use, by default None
    domain_padding_mode : {'symmetric', 'one-sided'}, optional
        How to perform domain padding, by default 'one-sided'
    fft_norm : str, optional
        by default 'forward'

    [1] : U-NO: U-shaped Neural Operators, Md Ashiqur Rahman, Zachary E Ross, Kamyar Azizzadenesheli, TMLR 2022
    Refer to get_model() in base_model.py, where mean, std, and upscale are added to the model arguments.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        lifting_channels=256,
        projection_channels=256,
        n_layers=4,
        uno_out_channels=None,
        uno_n_modes=None,
        uno_scalings=None,
        horizontal_skips_map=None,
        incremental_n_modes=None,
        use_mlp=False,
        mlp_dropout=0,
        mlp_expansion=0.5,
        non_linearity=F.gelu,
        norm=None,
        preactivation=False,
        fno_skip="linear",
        horizontal_skip="linear",
        mlp_skip="soft-gating",
        separable=False,
        factorization=None,
        rank=1.0,
        joint_factorization=False,
        fixed_rank_modes=False,
        integral_operator=SpectralConv,
        operator_block=FNOBlocks,
        implementation="factorized",
        decomposition_kwargs=dict(),
        domain_padding=None,
        domain_padding_mode="one-sided",
        fft_norm="forward",
        normalizer=None,
        verbose=False,
        num_rrdb = 23,
        mean = None,
        std = None,
        upscale = 8,
        local = False,
        **kwargs
    ):
        super().__init__()
        self.n_layers = n_layers
        assert uno_out_channels is not None, "uno_out_channels can not be None"
        assert uno_n_modes is not None, "uno_n_modes can not be None"
        assert uno_scalings is not None, "uno_scalings can not be None"
        assert (
            len(uno_out_channels) == n_layers
        ), "Output channels for all layers are not given"
        assert (
            len(uno_n_modes) == n_layers
        ), "number of modes for all layers are not given"
        assert (
            len(uno_scalings) == n_layers
        ), "Scaling factor for all layers are not given"

        self.n_dim = len(uno_n_modes[0])
        self.uno_out_channels = uno_out_channels
        self.uno_n_modes = uno_n_modes
        self.uno_scalings = uno_scalings
        self.hidden_channels = hidden_channels
        self.lifting_channels = lifting_channels
        self.projection_channels = projection_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.horizontal_skips_map = horizontal_skips_map
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
        self._incremental_n_modes = incremental_n_modes
        self.operator_block = operator_block
        self.integral_operator = integral_operator
        self.num_rrdb = num_rrdb
        self.local = local
        if self.local is True:
            print("choose local blocks")
            self.operator_block = LocalFNOBlocks
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
        
        # constructing default skip maps
        if self.horizontal_skips_map is None:
            self.horizontal_skips_map = {}
            for i in range(
                n_layers // 2,
                0,
            ):
                self.horizontal_skips_map[n_layers - i - 1] = i

        # self.uno_scalings may be a 1d list specifying uniform scaling factor at each layer
        # or a 2d list, where each row specifies scaling factors along each dimention.
        # To get the final (end to end) scaling factors we need to multiply
        # the scaling factors (a list) of all layer.

        self.end_to_end_scaling_factor = [1] * len(self.uno_scalings[0])
        # multiplying scaling factors
        for k in self.uno_scalings:
            self.end_to_end_scaling_factor = [
                i * j for (i, j) in zip(self.end_to_end_scaling_factor, k)
            ]

        # list with a single element is replaced by the scaler.
        if len(self.end_to_end_scaling_factor) == 1:
            self.end_to_end_scaling_factor = self.end_to_end_scaling_factor[0]

        if isinstance(self.end_to_end_scaling_factor, (float, int)):
            self.end_to_end_scaling_factor = [
                self.end_to_end_scaling_factor
            ] * self.n_dim

        if verbose:
            print("calculated out factor", self.end_to_end_scaling_factor)

        if domain_padding is not None and (
            (isinstance(domain_padding, list) and sum(domain_padding) > 0)
            or (isinstance(domain_padding, (float, int)) and domain_padding > 0)
        ):
            self.domain_padding = DomainPadding(
                domain_padding=domain_padding,
                padding_mode=domain_padding_mode,
                output_scaling_factor=self.end_to_end_scaling_factor,
            )
        else:
            self.domain_padding = None
        self.domain_padding_mode = domain_padding_mode

        self.lifting = MLP(
            in_channels=in_channels,
            out_channels=self.hidden_channels,
            hidden_channels=self.lifting_channels,
            n_layers=2,
            n_dim=self.n_dim,
        )
        self.fno_blocks = nn.ModuleList([])
        self.horizontal_skips = torch.nn.ModuleDict({})
        prev_out = self.hidden_channels

        for i in range(self.n_layers):
            if i in self.horizontal_skips_map.keys():
                prev_out = (
                    prev_out + self.uno_out_channels[self.horizontal_skips_map[i]]
                )
            
            self.fno_blocks.append(
                self.operator_block(
                    in_channels=prev_out,
                    out_channels=self.uno_out_channels[i],
                    n_modes=self.uno_n_modes[i],
                    use_mlp=use_mlp,
                    mlp_dropout=mlp_dropout,
                    mlp_expansion=mlp_expansion,
                    output_scaling_factor=[self.uno_scalings[i]],
                    non_linearity=non_linearity,
                    norm=norm,
                    preactivation=preactivation,
                    fno_skip=fno_skip,
                    mlp_skip=mlp_skip,
                    incremental_n_modes=incremental_n_modes,
                    rank=rank,
                    SpectralConv=self.integral_operator,
                    fft_norm=fft_norm,
                    fixed_rank_modes=fixed_rank_modes,
                    implementation=implementation,
                    separable=separable,
                    factorization=factorization,
                    decomposition_kwargs=decomposition_kwargs,
                    joint_factorization=joint_factorization,
                    normalizer=normalizer,
                )
            )


            if i in self.horizontal_skips_map.values():
                self.horizontal_skips[str(i)] = skip_connection(
                    self.uno_out_channels[i],
                    self.uno_out_channels[i],
                    skip_type=horizontal_skip,
                    n_dim=self.n_dim,
                )

            prev_out = self.uno_out_channels[i]

        self.projection = MLP(
            in_channels=prev_out,
            out_channels=out_channels,
            hidden_channels=self.projection_channels,
            n_layers=2,
            n_dim=self.n_dim,
            non_linearity=non_linearity,
        )

    
    def forward(self, x, **kwargs):

        x = self.shiftmean_x(x,"sub")
  
        hr_size_h = x.shape[2]*self.upsampling_factor
        hr_size_w = x.shape[3]*self.upsampling_factor

        ##RRDB blocks + bicubic interpolation added before UNO layers
        x = self.rrdb(x)
       
        x = F.interpolate(x, size=(hr_size_h, hr_size_w), mode='bicubic', align_corners=False)

        x = self.lifting(x)

        if self.domain_padding is not None:
            x = self.domain_padding.pad(x)
        output_shape = [
            int(round(i * j))
            for (i, j) in zip(x.shape[-self.n_dim :], self.end_to_end_scaling_factor)
        ]

        skip_outputs = {}
        cur_output = None
        for layer_idx in range(self.n_layers):
            if layer_idx in self.horizontal_skips_map.keys():
                skip_val = skip_outputs[self.horizontal_skips_map[layer_idx]]
                output_scaling_factors = [
                    m / n for (m, n) in zip(x.shape, skip_val.shape)
                ]
                output_scaling_factors = output_scaling_factors[-1 * self.n_dim :]
                t = resample(
                    skip_val, output_scaling_factors, list(range(-self.n_dim, 0))
                )
                x = torch.cat([x, t], dim=1)

            if layer_idx == self.n_layers - 1:
                cur_output = output_shape
            x = self.fno_blocks[layer_idx](x, output_shape=cur_output)

            if layer_idx in self.horizontal_skips_map.values():
                skip_outputs[layer_idx] = self.horizontal_skips[str(layer_idx)](x)

        if self.domain_padding is not None:
            x = self.domain_padding.unpad(x)

        x = self.projection(x)

        x = self.shiftmean_y(x,"add")
        
        return x
