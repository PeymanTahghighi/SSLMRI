import warnings
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from monai.networks.blocks.convolutions import Convolution, ResidualUnit
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.simplelayers import SkipConnection
from monai.utils import alias, deprecated_arg, export

__all__ = ["UNet", "Unet"]

class UpLayer(nn.Module):
        def __init__(self,
                    in_channels: int, 
                    out_channels: int, 
                    strides: int, 
                    is_top: bool = False,
                    kernel_size: Union[Sequence[int], int] = 3,
                    num_res_units: int = 0,
                    act: Union[Tuple, str] = Act.PRELU,
                    norm: Union[Tuple, str] = "BATCH",
                    dropout: float = 0.0,
                    bias: bool = True,
                    adn_ordering: str = "NDA",
                    spatial_dims: Optional[int] = None,
                    
            ):
            super().__init__()
            """
            Returns the decoding (up) part of a layer of the network. This typically will upsample data at some point
            in its structure. Its output is used as input to the next layer up.

            Args:
                in_channels: number of input channels.
                out_channels: number of output channels.
                strides: convolution stride.
                is_top: True if this is the top block.
            """
            self.dimensions = spatial_dims
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.strides = strides
            self.kernel_size = kernel_size
            self.num_res_units = num_res_units
            self.act = act
            self.norm = norm
            self.dropout = dropout
            self.bias = bias
            self.adn_ordering = adn_ordering

            self.upsample = Convolution(
                self.dimensions,
                in_channels,
                out_channels,
                strides=strides,
                kernel_size=kernel_size,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                conv_only=is_top and self.num_res_units == 0,
                is_transposed=True,
                adn_ordering=self.adn_ordering,
            )

            if self.num_res_units > 0:
                self.conv = ResidualUnit(
                    self.dimensions,
                    out_channels,
                    out_channels,
                    strides=1,
                    kernel_size=self.kernel_size,
                    subunits=self.num_res_units,
                    act=self.act,
                    norm=self.norm,
                    dropout=self.dropout,
                    bias=self.bias,
                    last_conv_only=is_top,
                    adn_ordering=self.adn_ordering,
                )
        def forward(self, x):
            x = self.upsample(x);
            return self.conv(x);


@export("monai.networks.nets")
@alias("Unet")
class UNet(nn.Module):

    @deprecated_arg(
        name="dimensions", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead."
    )
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Union[Sequence[int], int] = 3,
        up_kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 0,
        act: Union[Tuple, str] = Act.PRELU,
        norm: Union[Tuple, str] = "BATCH",
        dropout: float = 0.0,
        bias: bool = True,
        adn_ordering: str = "NDA",
        dimensions: Optional[int] = None,
    ) -> None:

        super().__init__()

        if len(channels) < 2:
            raise ValueError("the length of `channels` should be no less than 2.")
        delta = len(strides) - (len(channels) - 1)
        if delta < 0:
            raise ValueError("the length of `strides` should equal to `len(channels) - 1`.")
        if delta > 0:
            warnings.warn(f"`len(strides) > len(channels) - 1`, the last {delta} values of strides will not be used.")
        if dimensions is not None:
            spatial_dims = dimensions
        if isinstance(kernel_size, Sequence):
            if len(kernel_size) != spatial_dims:
                raise ValueError("the length of `kernel_size` should equal to `dimensions`.")
        if isinstance(up_kernel_size, Sequence):
            if len(up_kernel_size) != spatial_dims:
                raise ValueError("the length of `up_kernel_size` should equal to `dimensions`.")

        self.dimensions = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.adn_ordering = adn_ordering

        self.down_layers = nn.ModuleList();
        self.up_layers = nn.ModuleList();

        c = in_channels;
        for idx in range(len(channels)):
            self.down_layers.append(self._get_down_layer(c, channels[idx], 2 if idx != len(channels)-1 else 1, True if idx == 0 else False))
            c = channels[idx];
        
        c = channels[-1] + channels[-2];
        rev_channels = list(reversed(channels))
        for idx in range(2,len(rev_channels)):
            self.up_layers.append(UpLayer(c, 
                            rev_channels[idx], 
                            2,
                            spatial_dims=self.dimensions, 
                            kernel_size=self.kernel_size, 
                            num_res_units=self.num_res_units, 
                            act = self.act, 
                            norm=self.norm, 
                            dropout=self.dropout, 
                            adn_ordering=self.adn_ordering, 
                            bias=self.bias));
            c = rev_channels[idx] *2;
        
        self.up_layers.append(UpLayer(channels[0]*2, 
                           channels[0],
                           2, 
                           spatial_dims=self.dimensions, 
                           kernel_size=self.kernel_size, 
                           num_res_units=self.num_res_units, 
                           act = self.act, 
                           norm=self.norm, 
                           dropout=self.dropout, 
                           adn_ordering=self.adn_ordering, 
                           bias=self.bias));

        # self.down1 = self._get_down_layer(in_channels, channels[0], 2, True);
        # self.down2 = self._get_down_layer(channels[0], channels[1], 2, False);
        # self.down3 = self._get_down_layer(channels[1], channels[2], 2, False);
        # self.down4 = self._get_down_layer(channels[2], channels[3], 2, False);

        #bottleneck
        #self.down5 = self._get_down_layer(channels[3], channels[4], 1, False);
    
        # self.up4 = UpLayer(channels[3]+channels[4], 
        #                    channels[2], 
        #                    2, 
        #                    False, 
        #                    spatial_dims=self.dimensions, 
        #                    kernel_size=self.kernel_size, 
        #                    num_res_units=self.num_res_units, 
        #                    act = self.act, 
        #                    norm=self.norm, 
        #                    dropout=self.dropout, 
        #                    adn_ordering=self.adn_ordering, 
        #                    bias=self.bias);
        
        # self.up3 = UpLayer(channels[2]*2, 
        #                    channels[1], 
        #                    2, 
        #                    False, 
        #                    spatial_dims=self.dimensions, 
        #                    kernel_size=self.kernel_size, 
        #                    num_res_units=self.num_res_units, 
        #                    act = self.act, norm=self.norm, 
        #                    dropout=self.dropout, 
        #                    adn_ordering=self.adn_ordering, 
        #                    bias=self.bias);
        
        # self.up2 = UpLayer( channels[1]*2, 
        #                     channels[0], 2, 
        #                     False, 
        #                     spatial_dims=self.dimensions, 
        #                     kernel_size=self.kernel_size, 
        #                     num_res_units=self.num_res_units, 
        #                     act = self.act, norm=self.norm, 
        #                     dropout=self.dropout, 
        #                     adn_ordering=self.adn_ordering, 
        #                     bias=self.bias);

        
        
        self.output = Convolution(
            spatial_dims=self.dimensions,
            in_channels = channels[0],
            out_channels=out_channels,
            kernel_size=1,
            conv_only=True,
        )

        self._init_weights();

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight);
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def _get_down_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        """
        Returns the encoding (down) part of a layer of the network. This typically will downsample data at some point
        in its structure. Its output is used as input to the next layer down and is concatenated with output from the
        next layer to form the input for the decode (up) part of the layer.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            strides: convolution stride.
            is_top: True if this is the top block.
        """
        mod: nn.Module
        if self.num_res_units > 0:

            mod = ResidualUnit(
                self.dimensions,
                in_channels,
                out_channels,
                strides=strides,
                kernel_size=self.kernel_size,
                subunits=self.num_res_units,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                adn_ordering=self.adn_ordering,
            )
            return mod
        mod = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            adn_ordering=self.adn_ordering,
        )
        return mod
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.down_layers[0](x);
        d2 = self.down_layers[1](d1);
        d3 = self.down_layers[2](d2);
        d4 = self.down_layers[3](d3);
        d5 = self.down_layers[4](d4);

        u4 = self.up_layers[0](torch.cat([d5,d4], dim=1));
        u3 = self.up_layers[1](torch.cat([u4,d3], dim=1));
        u2 = self.up_layers[2](torch.cat([u3,d2], dim=1));
        u1 = self.up_layers[3](torch.cat([u2,d1], dim=1));
        #out = self.output(u1);

        return d1, d2, d3, d4, d5, u4, u3, u2, u1;

Unet = UNet
