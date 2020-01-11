""" Full assembly of the parts to form the complete network """
import sys
sys.path.append("/home/disk2/internship_anytime/liuhaohe/he_workspace/github/music_separator/")
from config.wavenetConfig import Config
import torch.nn.functional as F
import torch
import torch.nn as nn
from models.dense_net import _DenseBlock
'''
in: [batchsize,x,y,channels]
out: [batchsize,x,y,channels]
'''


class UNet(nn.Module):
    dense_layers = 4
    dense_bn = 4
    dense_growth_rate = 12
    if (Config.layer_numbers_unet == 5):
        def __init__(self, n_channels, n_classes, bilinear=True):
            super(UNet, self).__init__()
            self.n_channels = n_channels
            self.cnt = 0
            self.n_classes = n_classes
            self.bilinear = bilinear
            self.increase_rate = UNet.dense_layers*UNet.dense_growth_rate
            self.inc = DoubleConv(self.n_channels,self.getCh(1),TIF_in=1025)
            self.down1 = Down(self.getCh(1),self.getCh(2),TIF_in=512)
            self.down2 = Down(self.getCh(2),self.getCh(3),TIF_in=256)
            self.down3 = Down(self.getCh(3),self.getCh(4),TIF_in=128)
            self.down4 = Down(self.getCh(4),self.getCh(5),TIF_in=64)
            self.down5 = Down(self.getCh(5), self.getCh(6),TIF_in=32)
            # self.down6 = Down(2048, 2048)
            # self.up1 = Up(2048, 1024, bilinear)
            self.up2 = Up(self.getCh(5,6), 512, bilinear,TIF_in = 64)
            self.up3 = Up(self.getCh(4)+512, 256, bilinear,TIF_in = 128)
            self.up4 = Up(self.getCh(3)+256, 128, bilinear,TIF_in = 256)
            self.up5 = Up(self.getCh(2)+128, 64, bilinear,TIF_in = 512)
            self.up6 = Up(self.getCh(1)+64, 64, bilinear,TIF_in = 1025)
            self.outc = OutConv(64, n_classes)

        def getCh(self,n,n2 = None):
            if(n2 == None):return self.n_channels+n*self.increase_rate
            else:return self.n_channels+n*self.increase_rate+self.n_channels+n2*self.increase_rate

        def forward(self, x):
            x = x.permute(0, 3, 1, 2)
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            x6 = self.down5(x5)
            # x7 = self.down6(x6)
            # x = self.up1(x7, x6)
            x = self.up2(x6, x5)
            x = self.up3(x, x4)
            x = self.up4(x, x3)
            x = self.up5(x, x2)
            x = self.up6(x, x1)
            logits = self.outc(x)
            return logits.permute(0,2,3,1)

    if(Config.layer_numbers_unet == 4):
        def __init__(self, n_channels, n_classes, bilinear=True):
            super(UNet, self).__init__()
            self.cnt = 0
            self.n_channels = n_channels
            self.n_classes = n_classes
            self.bilinear = bilinear
            self.inc = DoubleConv(n_channels, 64,TIF_in=1025)
            self.down1 = Down(64, 128,TIF_in=512)
            self.down2 = Down(128, 256,TIF_in=256)
            self.down3 = Down(256, 512,TIF_in=128)
            self.down4 = Down(512, 512,TIF_in=64)
            self.up1 = Up(1024, 256, bilinear,TIF_in = 128)
            self.up2 = Up(512, 128, bilinear,TIF_in = 256)
            self.up3 = Up(256, 64, bilinear,TIF_in = 512)
            self.up4 = Up(128, 64, bilinear,TIF_in = 1025)
            self.outc = OutConv(64, n_classes)

        def forward(self, x):
            # in: (N,x,y,ch)
            x = x.permute(0,3,1,2)
            # (N,ch,x,y)
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
            logits = self.outc(x)
            return logits.permute(0,2,3,1)

    if (Config.layer_numbers_unet == 6):
        def __init__(self, n_channels, n_classes, bilinear=True):
            super(UNet, self).__init__()
            self.cnt = 0
            self.n_channels = n_channels
            self.n_classes = n_classes
            self.bilinear = bilinear

            self.inc = DoubleConv(n_channels, 64)
            self.down1 = Down(64, 128)
            self.down2 = Down(128, 256)
            self.down3 = Down(256, 512)
            self.down4 = Down(512, 1024)
            self.down5 = Down(1024, 2048)
            self.down6 = Down(2048, 2048)
            self.up1 = Up(4096, 1024, bilinear)
            self.up2 = Up(2048, 512, bilinear)
            self.up3 = Up(1024, 256, bilinear)
            self.up4 = Up(512, 128, bilinear)
            self.up5 = Up(256, 64, bilinear)
            self.up6 = Up(128, 64, bilinear)
            self.outc = OutConv(64, n_classes)

        def forward(self, x):
            x = x.permute(0, 3, 1, 2)
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            x6 = self.down5(x5)
            x7 = self.down6(x6)
            x = self.up1(x7, x6)
            x = self.up2(x, x5)
            x = self.up3(x, x4)
            x = self.up4(x, x3)
            x = self.up5(x, x2)
            x = self.up6(x, x1)
            logits = self.outc(x)
            return logits.permute(0,2,3,1)




class TIFblock(nn.Module):
    """(FC => [BN] => ReLU) * 2"""
    # Input: (N,ch, x, y)
    def __init__(self, in_channels,batchNorm_ch):
        super().__init__()
        self.double_FC = nn.Sequential(
            nn.Linear(in_channels,in_channels*2),
            nn.Linear(in_channels * 2, in_channels),
            nn.BatchNorm2d(batchNorm_ch),
            nn.ReLU(inplace=True),

            nn.Linear(in_channels,in_channels*2),
            nn.Linear(in_channels * 2, in_channels),
            nn.BatchNorm2d(batchNorm_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        x = x.permute(0,1,3,2)
        x = self.double_FC(x)
        return x.permute(0,1,3,2)

class _DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels,TIF_in = 2048):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.tif = TIFblock(TIF_in,out_channels)

    def forward(self, x):
        res = self.double_conv(x)
        return self.tif(res)+res

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels,TIF_in = 2048):
        super().__init__()
        self.dense_block = _DenseBlock(
                num_layers=UNet.dense_layers,
                num_input_features=in_channels,
                bn_size=UNet.dense_bn,
                growth_rate= UNet.dense_growth_rate,
                drop_rate=0.2,
                efficient=False,
            )
        self.tif = TIFblock(TIF_in,out_channels)

    def forward(self, x):
        res = self.dense_block(x)
        return self.tif(res)+res

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels,TIF_in):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels,TIF_in)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True,TIF_in = 2048):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = _DoubleConv(in_channels, out_channels,TIF_in = TIF_in)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

def get_model_parm_nums(model):
    # model = models.alexnet()
    total = sum([param.nelement() for param in model.parameters()])
    print('  + Number of params: %.2fM' % (total / 1e6))

if __name__ == "__main__":
    model = UNet(n_channels = 2,n_classes = 2)# .cuda(Config.device)
    # print_model_parm_nums(model)
    get_model_parm_nums(model)