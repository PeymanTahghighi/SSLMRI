#===============================================================
#===============================================================
from copy import deepcopy
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import config
from resnet import resnet50
from monai.networks.blocks.convolutions import Convolution, ResidualUnit
from monai.networks.layers.factories import Act
from typing import Optional, Sequence, Tuple, Union
import warnings
import numpy as np
#===============================================================
#===============================================================


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

#---------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, norm=True, act=True) -> None:
        super().__init__();
        ops = [];
        ops.append(nn.Conv3d(in_channels, out_channels, kernel_size, stride, bias=False, padding=kernel_size//2));
        if norm:
            ops.append(nn.BatchNorm3d(out_channels));
        if act:
            ops.append(nn.LeakyReLU(0.2, inplace=True));
        self.net = nn.Sequential(*ops);
    def forward(self, x):
        return self.net(x);
#---------------------------------------------------------------

#---------------------------------------------------------------
class ResConvBlock(nn.Module):
    '''
    Based on Z. Liu, H. Mao, C. -Y. Wu, C. Feichtenhofer, T. Darrell and S. Xie, "A ConvNet for the 2020s," 
    2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), New Orleans, LA, USA, 2022, pp. 
    11966-11976, doi: 10.1109/CVPR52688.2022.01167.
    '''
    def __init__(self, in_channels, out_channels, kernel_size) -> None:
        super().__init__();
        self.res_block = nn.Sequential(
        nn.Conv3d(in_channels, 
                  out_channels,
                  kernel_size, 
                  padding=kernel_size//2),
        nn.BatchNorm3d(out_channels),
        nn.Conv3d(out_channels, out_channels*2, kernel_size=1),
        nn.GELU(),
        nn.Conv3d(out_channels*2, out_channels, kernel_size=1)
        )

        if in_channels != out_channels:
            self.extra_conv = nn.Conv3d(in_channels, out_channels, kernel_size);
    def forward(self, x):
        return self.res_block(x) + self.extra_conv(x) if hasattr(self, 'extra_conv') else x;
#---------------------------------------------------------------

#---------------------------------------------------------------
class Upsample(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, type) -> None:
        super().__init__();

        if type == 'convtrans':
            self.net = nn.ConvTranspose3d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size, stride=2, padding=1);
        else:
            self.net = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv3d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size, stride=1)
            )
        
        self.conv_after = nn.Sequential(
            ConvBlock(in_channels=out_features, out_channels=out_features, kernel_size=3, stride=1),
            ConvBlock(in_channels=out_features, out_channels=out_features, kernel_size=3, stride=1))
    
    def forward(self, x):
        x = self.net(x);
        x = self.conv_after(x);
        return x;
#---------------------------------------------------------------

#---------------------------------------------------------------
class Upblock(nn.Module):
    def __init__(self, in_features, out_features, concat_features = None) -> None:
        super().__init__();
        if concat_features == None:
            concat_features = out_features*2;

        self.upsample = Upsample(in_features, out_features, 4, 'convtrans');
        self.conv1 = ConvBlock(in_channels=concat_features, out_channels=out_features, kernel_size=3, stride=1);
        self.conv2 = ConvBlock(in_channels=out_features, out_channels=out_features, kernel_size=3, stride=1);

    def forward(self, x1, x2):
        x1 = self.upsample(x1);
        ct = torch.cat([x1,x2], dim=1);
        ct = self.conv1(ct);
        out = self.conv2(ct);
        return out;
#---------------------------------------------------------------

class AttentionBlock(nn.Module):
    def __init__(self, feature_size) -> None:
        super().__init__();
        self.feature_size = feature_size;
        self.m1 = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten()
        )
        self.model = nn.Sequential(
            nn.Linear(feature_size, feature_size*2),
            nn.LeakyReLU(0.2),
            nn.Linear(feature_size*2, feature_size),
        )

    def forward(self, x):
        x = self.m1(x);
        out = x + self.model(x);
        return torch.sigmoid(out);

#---------------------------------------------------------------
class CrossAttention(nn.Module):
    def __init__(self, channel, num_heads) -> None:
        super().__init__();
        self.channels = channel;
        self.ca = nn.MultiheadAttention(channel, num_heads, batch_first=True);
        self.ln = nn.LayerNorm([self.channels]);
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channel]),
            nn.Linear(channel, channel),
            nn.GELU(),
            nn.Linear(channel, channel)
        )
    
    def get_pos_embedding(self, tokens, channels):
        pe = torch.zeros((1,tokens, channels) , device= config.DEVICE, requires_grad=False);
        inv_freq_even = 1.0/((10000)**(torch.arange(0,channels,2) / channels));
        inv_freq_odd = 1.0/((10000)**(torch.arange(1,channels,2) / channels));
        pe[:,:,0::2] = torch.sin(torch.arange(0,tokens).unsqueeze(dim=1) * inv_freq_even.unsqueeze(dim=0));
        pe[:,:,1::2] = torch.cos(torch.arange(0,tokens).unsqueeze(dim=1) * inv_freq_odd.unsqueeze(dim=0));
        return pe;

    def forward(self, x1, x2):
        B,C,W,H,D = x1.shape;
        
        x1 = x1.view(B, C, W*H*D).swapaxes(1,2);
        x2 = x2.view(B, C, W*H*D).swapaxes(1,2);

        x1 = self.get_pos_embedding(W*H*D, C) + x1;
        x2 = self.get_pos_embedding(W*H*D, C) + x2;

        x1_ln = self.ln(x1);
        x2_ln = self.ln(x2);

        attntion_value, _ = self.ca(x1_ln, x2_ln, x2_ln);

        attntion_value = self.ff_self(attntion_value) + attntion_value;
        return attntion_value.swapaxes(2,1).view(B,C,W,H,D);
#---------------------------------------------------------------

#---------------------------------------------------------------
class ResUnet3D(nn.Module):
    def __init__(self) -> None:
        super().__init__();
        resnet = resnet50();
        ckpt = torch.load('resnet_50.pth')['state_dict'];
        modified_keys = {};
        for k in ckpt.keys():
            new_k = k.replace('module.','');
            modified_keys[new_k] = ckpt[k];
        resnet.load_state_dict(modified_keys, strict=False);
        self.input_blocks = ConvBlock(1,64,3,2);
        self.input_pool = list(resnet.children())[3];
        self.down_blocks = nn.ModuleList();
        for btlnck in list(resnet.children()):
            if isinstance(btlnck, nn.Sequential):
                self.down_blocks.append(btlnck);

        self.bottle_neck = nn.Sequential(
            ConvBlock(2048, 2048, 3, 1),
            ConvBlock(2048, 2048, 3, 1)
        );

        self.inp_conv = ConvBlock(1, 64, 3, 1);

        self.up_1 = Upblock(2048,1024);
        self.up_2 = Upblock(1024,512);
        self.up_3 = Upblock(512,256);
        self.up_4 = Upblock(256, 128, 128+64)
        self.up_5 = Upblock(128, 64, 128);

        self.feature_selection_modules = nn.ModuleList();
        self.feature_refinement_modules = nn.ModuleList();
        self.feature_attention_modules = nn.ModuleList();

        feats = [2048,1024,512,256,64,64];
        for f in feats:
            layers = self._make_squeeze_excitation(f);
            self.feature_selection_modules.append(layers[0]);
            self.feature_refinement_modules.append(layers[1]);
            self.feature_attention_modules.append(layers[2]);

        self.final = nn.Sequential(
            ConvBlock(64,1,1,1),
        )


        self.__initial_weights = deepcopy(self.state_dict());
    
    def _make_squeeze_excitation(self, feature_size):
        feature_selection = nn.Sequential(
            ConvBlock(feature_size*2, feature_size, 1, 1),
        )
        refinement = nn.Sequential(
            ConvBlock(feature_size, feature_size, 3, 1),
            ConvBlock(feature_size, feature_size, 3, 1),
        )
        atten = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Sigmoid()
        )

        return feature_selection, refinement, atten;

    def down_stream(self, inp):
        inp_feat = self.inp_conv(inp);
        d_1 = self.input_blocks(inp);

        d_2 = self.down_blocks[0](d_1);
        d_2 = self.input_pool(d_2);
        d_3 = self.down_blocks[1](d_2);
        d_4 = self.down_blocks[2](d_3);
        d_5 = self.down_blocks[3](d_4);

        d_5 = self.bottle_neck(d_5);
        return d_5, d_4, d_3, d_2, d_1, inp_feat;

    def squeeze_excitation_block(self, inp1, inp2, idx):
        d = torch.concat([inp1, inp2], dim=1);
        d_selection = self.feature_selection_modules[idx](d);
        d_refine = self.feature_refinement_modules[idx](d_selection);
        d_attn = self.feature_attention_modules[idx](d_refine);
        d_refine = d_refine * d_attn;
        return F.leaky_relu(d_refine + d_selection, 0.2, inplace=True);

    def forward(self, inp1, inp2):
        
        inp1_d5, inp1_d4, inp1_d3, inp1_d2, inp1_d1, inp_feat_1 = self.down_stream(inp1);
        inp2_d5, inp2_d4, inp2_d3, inp2_d2, inp2_d1, inp_feat_2 = self.down_stream(inp2);

        d_5 = self.squeeze_excitation_block(inp1_d5, inp2_d5, 0);
        d_4 = self.squeeze_excitation_block(inp1_d4, inp2_d4, 1);
        d_3 = self.squeeze_excitation_block(inp1_d3, inp2_d3, 2);
        d_2 = self.squeeze_excitation_block(inp1_d2, inp2_d2, 3);
        d_1 = self.squeeze_excitation_block(inp1_d1, inp2_d1, 4);
        inp_feat = self.squeeze_excitation_block(inp_feat_1, inp_feat_2, 5);
    
        u_1 = self.up_1(d_5, d_4);
        u_2 = self.up_2(u_1, d_3);
        u_3 = self.up_3(u_2, d_2);
        u_4 = self.up_4(u_3, d_1);
        u_5 = self.up_5(u_4, inp_feat);

        out = self.final(u_5);

        return out;
#---------------------------------------------------------------

class UNet3D(nn.Module):

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


        self.feature_selection_modules = nn.ModuleList();
        self.feature_refinement_modules = nn.ModuleList();
        self.feature_attention_modules = nn.ModuleList();

        for f in rev_channels:
            layers = self._make_squeeze_excitation(f);
            self.feature_selection_modules.append(layers[0]);
            self.feature_refinement_modules.append(layers[1]);
            self.feature_attention_modules.append(layers[2]);

        self.final = nn.Sequential(
            nn.Conv3d(rev_channels[-1], 1, 1, 1, bias=False, padding=0),
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
    
    def _make_squeeze_excitation(self, feature_size):
        feature_selection = nn.Sequential(
            ConvBlock(feature_size*2, feature_size, 1, 1, act=False),
        )
        refinement = ResConvBlock(
                feature_size,
                feature_size,
                kernel_size=self.kernel_size,
            )
        atten = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Sigmoid()
        )

        return feature_selection, refinement, atten;


    def squeeze_excitation_block(self, inp1, inp2, idx):
        d = torch.concat([inp1, inp2], dim=1);
        d_selection = self.feature_selection_modules[idx](d);
        d_refine = self.feature_refinement_modules[idx](d_selection);
        d_attn = self.feature_attention_modules[idx](d_refine);
        d_refine = d_refine * d_attn;
        return F.leaky_relu(d_refine + d_selection, 0.2, inplace=True);

    def _down_path(self, x:torch.Tensor):
        outputs = [];
        out =  x;
        for l in self.down_layers:
            out = l(out);
            outputs.append(out);

        return list(reversed(outputs));

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        inp1_outputs = self._down_path(x1);
        inp2_outputs = self._down_path(x2);

        merged = [];
        for i in range(len(self.down_layers)):
            merged.append(self.squeeze_excitation_block(inp1_outputs[i], inp2_outputs[i], i));

   
        up = self.up_layers[0](torch.cat([merged[0],merged[1]], dim=1));
        for i in range(2, len(merged)):
            up = self.up_layers[i-1](torch.cat([up, merged[i]], dim=1));
        
        out = self.final(up);

        return out;
#---------------------------------------------------------------

class AttentionBlock(nn.Module):
    def __init__(self, feature_size) -> None:
        super().__init__();
        self.feature_size = feature_size;
        self.m1 = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten()
        )
        self.model = nn.Sequential(
            nn.Linear(feature_size, feature_size*2),
            nn.LeakyReLU(0.2),
            nn.Linear(feature_size*2, feature_size),
        )

    def forward(self, x):
        x = self.m1(x);
        out = x + self.model(x);
        return torch.sigmoid(out);

class UNet3DSSL(nn.Module):

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
        act: Union[Tuple, str] = "PRELU",
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

        c = in_channels;
        for idx in range(len(channels)):
            self.down_layers.append(self._get_down_layer(c, channels[idx], 2 if idx != len(channels)-1 else 1, True if idx == 0 else False))
            c = channels[idx];
        
        c = channels[-1] + channels[-2];
        rev_channels = list(reversed(channels))
        

        self.upsample1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(rev_channels[0], rev_channels[0], kernel_size=3, padding=1),
            nn.InstanceNorm3d(rev_channels[0]),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(rev_channels[0], rev_channels[0], kernel_size=3, padding=1),
            nn.InstanceNorm3d(rev_channels[0]),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(rev_channels[0], rev_channels[0], kernel_size=3, padding=1),
            nn.InstanceNorm3d(rev_channels[0]),
            nn.LeakyReLU(),
        )

        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(rev_channels[1], rev_channels[1], kernel_size=3, padding=1),
            nn.InstanceNorm3d(rev_channels[1]),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(rev_channels[1], rev_channels[1], kernel_size=3, padding=1),
            nn.InstanceNorm3d(rev_channels[1]),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(rev_channels[1], rev_channels[1], kernel_size=3, padding=1),
            nn.InstanceNorm3d(rev_channels[1]),
            nn.LeakyReLU(),
        )

        self.upsample3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(rev_channels[2], rev_channels[2], kernel_size=3, padding=1),
            nn.InstanceNorm3d(rev_channels[2]),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(rev_channels[2], rev_channels[2], kernel_size=3, padding=1),
            nn.InstanceNorm3d(rev_channels[2]),
            nn.LeakyReLU(),
        )

        self.upsample4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(rev_channels[3], rev_channels[3], kernel_size=3, padding=1),
            nn.InstanceNorm3d(rev_channels[3]),
            nn.LeakyReLU(),
        )


        self.feature_selection_modules = nn.ModuleList();
        self.feature_refinement_modules = nn.ModuleList();
        self.feature_attention_modules = nn.ModuleList();

        for f in rev_channels:
            layers = self._make_squeeze_excitation(f);
            self.feature_selection_modules.append(layers[0]);
            self.feature_refinement_modules.append(layers[1]);
            self.feature_attention_modules.append(layers[2]);

        self.final = nn.Sequential(
            nn.Conv3d(np.sum(channels), 1, 1, 1, bias=False, padding=0),
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
    
    def _make_squeeze_excitation(self, feature_size):
        feature_selection = nn.Sequential(
            ConvBlock(feature_size*2, feature_size, 1, 1, act=False),
        )
        refinement = ResConvBlock(
                feature_size,
                feature_size,
                kernel_size=self.kernel_size,
            )
        atten = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Sigmoid()
        )

        return feature_selection, refinement, atten;


    def squeeze_excitation_block(self, inp1, inp2, idx):
        d = torch.concat([inp1, inp2], dim=1);
        d_selection = self.feature_selection_modules[idx](d);
        d_refine = self.feature_refinement_modules[idx](d_selection);
        d_attn = self.feature_attention_modules[idx](d_refine);
        d_refine = d_refine * d_attn;
        return F.leaky_relu(d_refine + d_selection, 0.2, inplace=True);

    def _down_path(self, x:torch.Tensor):
        outputs = [];
        out =  x;
        for l in self.down_layers:
            out = l(out);
            outputs.append(out);

        return list(reversed(outputs));

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        inp1_outputs = self._down_path(x1);
        inp2_outputs = self._down_path(x2);

        merged = [];
        for i in range(len(self.down_layers)):
            merged.append(self.squeeze_excitation_block(inp1_outputs[i], inp2_outputs[i], i));

   
        up1 = self.upsample1(merged[0]);
        up2 = self.upsample2(merged[1]);
        up3 = self.upsample3(merged[2]);
        up4 = self.upsample4(merged[3]);
        merged = torch.concat([up1, up2, up3, up4], dim = 1);
        
        out = self.final(merged);

        return out;

    def load_pretrained_monai_unet3d(self):
        ckpt = torch.load('pretrained/spleen_ct_segmentation/model.pt');

        sdn = self.state_dict()

        sdn_keys = list(sdn.keys());
        sdm_keys = list(ckpt.keys());
        n = np.zeros((5,5));
        
        for i in range(len(sdm_keys)):
            k = sdn_keys[i]
            if sdn[k].shape ==  ckpt[sdm_keys[i]].shape:
                sdn[k] = ckpt[sdm_keys[i]];
                print(k);
        self.load_state_dict(sdn, strict=False);
#---------------------------------------------------------------

class CrossAttentionUNet3D(nn.Module):

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
            self.down_layers.append(self._get_down_layer(c, channels[idx], strides[idx] if idx != len(channels)-1 else 1, True if idx == 0 else False))
            c = channels[idx];
        
        c = channels[-1] + channels[-2];
        rev_channels = list(reversed(channels))
        rev_strides = list(reversed(strides))
        for idx in range(2,len(rev_channels)):
            self.up_layers.append(UpLayer(c, 
                            rev_channels[idx], 
                            rev_strides[idx-2],
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
                           strides[0], 
                           spatial_dims=self.dimensions, 
                           kernel_size=self.kernel_size, 
                           num_res_units=self.num_res_units, 
                           act = self.act, 
                           norm=self.norm, 
                           dropout=self.dropout, 
                           adn_ordering=self.adn_ordering, 
                           bias=self.bias));

        self.output = Convolution(
            spatial_dims=self.dimensions,
            in_channels = channels[0],
            out_channels=out_channels,
            kernel_size=1,
            conv_only=True,
        )

        self.cross_attention_modules = nn.ModuleList();
        for idx in range(len(rev_channels)):
            self.cross_attention_modules.append(CrossAttention(rev_channels[idx], num_heads=8));

        self.final = nn.Sequential(
            ConvBlock(rev_channels[-1],1,1,1),
            nn.Tanh()
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

    def _down_path(self, x:torch.Tensor):
        outputs = [];
        out =  x;
        for l in self.down_layers:
            out = l(out);
            outputs.append(out);

        return list(reversed(outputs));

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        inp1_outputs = self._down_path(x1);
        inp2_outputs = self._down_path(x2);

        # out4 = self.ca4(inp1_outputs[0], inp2_outputs[0]);
        # out3 = self.ca3(inp1_outputs[1], inp2_outputs[1]);
        # out2 = self.ca2(inp1_outputs[2], inp2_outputs[2]);
        # out1 = self.ca1(inp1_outputs[3], inp2_outputs[3]);

        merged = [];
        for i in range(len(self.down_layers)):
            merged.append(self.cross_attention_modules[i](inp1_outputs[i], inp2_outputs[i]));

   
        up = self.up_layers[0](torch.cat([merged[0],merged[1]], dim=1));
        for i in range(2, len(merged)):
            up = self.up_layers[i-1](torch.cat([up, merged[i]], dim=1));
        
        out = self.final(up);

        return out;

def test():

    sample = torch.rand((1,1,64,128,128)).to('cuda');

    net = UNet3D(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(64, 128, 256, 512),
        strides=(2, 2, 2),
        num_res_units=2,
        ).to('cuda')
    
    total_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad);
    print(f'total parameters: {total_parameters}')
    

    out = net(sample, sample);
    print(out.size());
#---------------------------------------------------------------

#---------------------------------------------------------------
if __name__ == "__main__":
    test();
#---------------------------------------------------------------