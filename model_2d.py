#===============================================================
#===============================================================
from copy import deepcopy
from shutil import copy
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import config
from monai.networks.nets.swin_unetr import SwinUNETR
#===============================================================
#===============================================================

#---------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride) -> None:
        super().__init__();
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=False, padding=kernel_size//2),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
    def forward(self, x):
        return self.net(x);
#---------------------------------------------------------------

#---------------------------------------------------------------
class Upsample(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, type) -> None:
        super().__init__();

        if type == 'convtrans':
            self.net = nn.ConvTranspose2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size, stride=2, padding=1);
        else:
            self.net = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size, stride=1)
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

class AttenUnet3D(nn.Module):
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

        self.ca5 = CrossAttention(2048);
        self.ca4 = CrossAttention(1024);
        self.ca3 = CrossAttention(512);
        self.ca2 = CrossAttention(256);
        self.ca1 = CrossAttention(64);


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
            nn.Tanh()
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

        d_5 = self.ca5(inp1_d5, inp2_d5);
        d_4 = self.ca4(inp1_d4, inp2_d4);
        d_3 = self.ca3(inp1_d3, inp2_d3);
        d_2 = self.ca2(inp1_d2, inp2_d2);
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

#---------------------------------------------------------------
def test():

    sample = torch.rand((1,1,128,128));
    net = Unet();
    out = net(sample, sample);
    print(out.size());
#---------------------------------------------------------------

#---------------------------------------------------------------
if __name__ == "__main__":
    test();
#---------------------------------------------------------------