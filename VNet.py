import torch
from torch import nn

"""
    modified version from Wu, Yicheng, et al. "Coactseg: Learning from heterogeneous data for new multiple sclerosis lesion segmentation." 
    International conference on medical image computing and computer-assisted intervention. Cham: Springer Nature Switzerland, 2023.
"""

class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlock, self).__init__()

        ops = []
        input_channel = n_filters_in
        for i in range(n_stages):

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False
            ops.append(nn.ReLU(inplace=True))
            input_channel = n_filters_out

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class ResidualConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ResidualConvBlock, self).__init__()

        ops = []
        input_channel = n_filters_in
        for i in range(n_stages):
            

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False

            if i != n_stages-1:
                ops.append(nn.ReLU(inplace=True))

            input_channel = n_filters_out

        if n_filters_in != n_filters_out:
            self.conv_on_input = nn.Sequential(
                nn.Conv3d(n_filters_in, n_filters_out, 3, 1, 1),
                nn.BatchNorm3d(n_filters_out)
            )
        self.conv = nn.Sequential(*ops)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = (self.conv(x) + x if hasattr(self, 'conv_on_input') is False else self.conv_on_input(x));
        x = self.relu(x)
        return x


class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(DownsamplingConvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class UpSampling(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none', mode_upsampling = 1):
        super(UpSampling, self).__init__()

        ops = []
        if mode_upsampling == 0:
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
        if mode_upsampling == 1:
            ops.append(nn.Upsample(scale_factor=stride, mode="trilinear", align_corners=False))
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))
        elif mode_upsampling == 2:
            ops.append(nn.Upsample(scale_factor=stride, mode="nearest"))
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))

        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != 'none':
            assert False
        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x
    
class Encoder(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, has_residual=False):
        super(Encoder, self).__init__()
        self.has_dropout = has_dropout
        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.block_one = convBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = convBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
        
        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def forward(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)

        if self.has_dropout:
            x5 = self.dropout(x5)

        res = [x1, x2, x3, x4, x5]
        return res
    
    
class Decoder(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, has_residual=False, up_type=0):
        super(Decoder, self).__init__()
        self.has_dropout = has_dropout

        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.block_five_up = UpSampling(n_filters * 16, n_filters * 8, normalization=normalization, mode_upsampling=up_type)

        self.block_six = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = UpSampling(n_filters * 8, n_filters * 4, normalization=normalization, mode_upsampling=up_type)

        self.block_seven = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = UpSampling(n_filters * 4, n_filters * 2, normalization=normalization, mode_upsampling=up_type)

        self.block_eight = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = UpSampling(n_filters * 2, n_filters, normalization=normalization, mode_upsampling=up_type)

        self.block_nine_1 = convBlock(1, n_filters, n_filters, normalization=normalization)
        self.block_nine_2 = convBlock(1, n_filters, n_filters, normalization=normalization)
        self.block_nine_3 = convBlock(1, n_filters, n_filters, normalization=normalization)

        self.block_c2f = convBlock(1, n_filters*3, n_filters, normalization=normalization)

        self.out_conv_3 = nn.Conv3d(n_filters, n_classes, 1, padding=0)
        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def forward(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]
        
        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1

        x91 = self.block_nine_1(x8_up)
        if self.has_dropout:
            x91 = self.dropout(x91)

        x92 = self.block_nine_2(x8_up)
        if self.has_dropout:
            x92 = self.dropout(x92)

        x93 = self.block_nine_3(x8_up)
        if self.has_dropout:
            x93 = self.dropout(x93)

        out_seg_diff = torch.cat(((x92 - x91), x91, x93), dim=1)
        out_seg_3 = self.block_c2f(out_seg_diff)
        out_seg_3 = self.out_conv_3(out_seg_3)
        
        return out_seg_3
    
class SSLHead(nn.Module):
    def __init__(self, n_fiters) -> None:
        super().__init__();
        self.first_upsample = nn.Sequential(
            UpSampling(n_fiters*16, n_fiters*8, normalization='instancenorm'),
            UpSampling(n_fiters*8, n_fiters*4, normalization='instancenorm'),
            UpSampling(n_fiters*4, n_fiters*2, normalization='instancenorm'),
            UpSampling(n_fiters*2, n_fiters, normalization='instancenorm'),
        )

        # self.second_upsample = nn.Sequential(
        #     UpSampling(n_fiters*8, n_fiters*4, normalization='instancenorm'),
        #     UpSampling(n_fiters*4, n_fiters*2, normalization='instancenorm'),
        #     UpSampling(n_fiters*2, n_fiters, normalization='instancenorm'),
        # )

        # self.third_upsample = nn.Sequential(
        #     UpSampling(n_fiters*4, n_fiters*2, normalization='instancenorm'),
        #     UpSampling(n_fiters*2, n_fiters, normalization='instancenorm'),
        # )

        # self.fourth_upsample = nn.Sequential(
        #     UpSampling(n_fiters*2, n_fiters, normalization='instancenorm'),
        # )

        self.final_conv = nn.Conv3d(16, 1, 3, 1, 1);
    
    def forward(self, x):
        u1 = self.first_upsample(x[-1]);
        # u2 = self.second_upsample(x[-2]);
        # u3 = self.third_upsample(x[-3]);
        # u4 = self.fourth_upsample(x[-4]);
        out = self.final_conv(u1);
        return out;
        
class VNet(nn.Module):
    def __init__(self, model_type, n_channels=3, n_classes=1, n_filters=16, normalization='none', has_dropout=False, has_residual=False):
        super(VNet, self).__init__()

        self.encoder = Encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        self.modle_type = model_type;

        if model_type == 'segmentation':
            self.decoder = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 0)
        else:
            self.ssl_head = SSLHead(n_filters);

        
    def forward(self, input):
        features = self.encoder(input)
        if self.modle_type == 'segmentation':
            out_seg_3 = self.decoder(features)
            return out_seg_3
        out = self.ssl_head(features);
        return out;
    
def test():
    model = VNet(model_type='pretraining', n_channels=3, n_classes=2, normalization='batchnorm', has_dropout=True);
    inp = torch.rand((2, 3, 256, 256, 256));
    model(inp);
if __name__ == '__main__':
    test();
    
