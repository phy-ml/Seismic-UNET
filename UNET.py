# source : https://github.com/davidiommi/Pytorch-Unet3D-single_channel/blob/master/UNet.py

from torch.nn import Module, Sequential
from torch.nn import Conv3d, ConvTranspose3d, BatchNorm3d, MaxPool3d, AvgPool1d, Dropout3d
from torch.nn import ReLU, Sigmoid
import torch

class UNet(Module):

    def __init__(self,num_channels=2, feat_channels=[64,256,256,512,1024], residual='conv'):
        super(UNet,self).__init__()

        # Encoder downsampler
        self.pool1 = MaxPool3d((2,2,2))
        self.pool2 = MaxPool3d((2, 2, 2))
        self.pool3 = MaxPool3d((2, 2, 2))
        self.pool4 = MaxPool3d((2, 2, 2))

        # Encoder Convolution
        self.conv_blk1 = Conv3D_Block(num_channels, feat_channels[0], residual=residual)
        self.conv_blk2 = Conv3D_Block(feat_channels[0], feat_channels[1], residual=residual)
        self.conv_blk3 = Conv3D_Block(feat_channels[1], feat_channels[2], residual=residual)
        self.conv_blk4 = Conv3D_Block(feat_channels[2], feat_channels[3], residual=residual)
        self.conv_blk5 = Conv3D_Block(feat_channels[3], feat_channels[4], residual=residual)

        # Decoder Convolution
        self.decod_conv_blk4 = Conv3D_Block(2*feat_channels[3], feat_channels[3], residual=residual)
        self.decod_conv_blk3 = Conv3D_Block(2 * feat_channels[2], feat_channels[2], residual=residual)
        self.decod_conv_blk2 = Conv3D_Block(2 * feat_channels[1], feat_channels[1], residual=residual)
        self.decod_conv_blk1 = Conv3D_Block(2 * feat_channels[0], feat_channels[0], residual=residual)

        # Decoder Upsampler
        self.conv_upsamp_blk4 = Deconv3D_Block(feat_channels[4], feat_channels[3])
        self.conv_upsamp_blk3 = Deconv3D_Block(feat_channels[3], feat_channels[2])
        self.conv_upsamp_blk2 = Deconv3D_Block(feat_channels[2], feat_channels[1])
        self.conv_upsamp_blk1 = Deconv3D_Block(feat_channels[1], feat_channels[0])

        # 1X1 conv segmentation map
        self.one_conv = Conv3d(feat_channels[0], num_channels, kernel_size=1, stride=1, padding=0, bias=True)

        # Activation function
        self.sigmoid = Sigmoid()

    def forward(self, x):
        # Encoder part
        x1 = self.conv_blk1(x)
        x_low_1 = self.pool1(x1)

        x2 = self.conv_blk2(x_low_1)
        x_low_2 = self.pool2(x2)

        x3 = self.conv_blk3(x_low_2)
        x_low_3 = self.pool3(x3)

        x4 = self.conv_blk4(x_low_3)
        x_low_4 = self.pool4(x4)

        base = self.conv_blk5(x_low_4)

        # Decoder Part
        d4 = torch.cat([self.conv_upsamp_blk4(base),x4], dim=1)
        d4_high = self.decod_conv_blk4(d4)

        d3 = torch.cat([self.conv_upsamp_blk3(d4_high), x3], dim=1)
        d3_high = self.decod_conv_blk3(d3)
        d3_high = Dropout3d(p=0.5)(d3_high)

        d2 = torch.cat([self.conv_upsamp_blk2(d3_high), x2], dim=1)
        d2_high = self.decod_conv_blk2(d2)
        d2_high = Dropout3d(p=0.5)(d2_high)

        d1 = torch.cat([self.conv_upsamp_blk1(d2_high), x1], dim=1)
        d1_high = self.decod_conv_blk1(d1)

        final = self.sigmoid(self.one_conv(d1_high))

        return final


class Conv3D_Block(Module):
    def __init__(self,in_feat, out_feat, kernel=3, stride=1, padding=1, residual=None):
        super(Conv3D_Block, self).__init__()

        self.conv1 = Sequential(
            Conv3d(in_feat, out_feat, kernel_size=kernel, stride=stride, padding=padding, bias=True),
            BatchNorm3d(out_feat),
            ReLU())

        self.conv2 = Sequential(
            Conv3d(out_feat, out_feat, kernel_size=kernel, stride=stride, padding=padding, bias=True),
            BatchNorm3d(out_feat),
            ReLU()
        )

        self.residual = residual

        if self.residual is not None:
            self.residual_upsampler = Conv3d(in_feat, out_feat, kernel_size=1,bias=False)


    def forward(self, x):
        res = x

        if not self.residual:
            return self.conv2(self.conv1(x))

        else:
            return self.conv2(self.conv1(x)) + self.residual_upsampler(res)

class Deconv3D_Block(Module):
    def __init__(self, in_feat, out_feat, kernel=3, stride=2, padding=1):
        super(Deconv3D_Block,self).__init__()

        self.deconv = Sequential(
            ConvTranspose3d(in_feat, out_feat, kernel_size=(kernel, kernel, kernel),
                            stride=(stride,stride,stride),
                            padding=(padding,padding,padding),
                            output_padding=1, bias=True),
            ReLU()
        )

    def forward(self,x):
        return self.deconv(x)

class ChannelPool3D(AvgPool1d):

    def __init__(self,kernel, stride, padding):
        super(ChannelPool3D).__init__(kernel, stride, padding)

        self.pool_1d = AvgPool1d(self.kernel_size, self.stride, self.padding, self.ceil_mode)

    def forward(self,x):
        n,c,d,w,h = x.size()
        x = x.view(n,c,d*w*h).permute(0,2,1)
        pooled = self.pool_1d(x)
        c = int(c/self.kernel_size[0])
        return x.view(n,c,d,w,h)

if __name__ == '__main__':
    model = UNet().eval()

    x = torch.randn(1,2,128,128,128)
    out = model(x)
    print(out)
    #print(model)