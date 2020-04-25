""" Full assembly of the parts to form the complete network """
import sys
sys.path.append("..")
from config.mainConfig import Config
import torch.nn.functional as F
import torch
import torch.nn as nn
import os
from util.wave_util import save_pickle

class MbandUnet(nn.Module):
    def __init__(self,inchannel = 2,outchannel = 2):
        super(MbandUnet, self).__init__()
        self.unet_full = UNet(n_channels=inchannel,n_classes=outchannel)
        self.unet_1_4 = UNet(n_channels=inchannel,n_classes=outchannel)
        self.unet_2_4 = UNet(n_channels=inchannel,n_classes=outchannel)
        self.unet_3_4 = UNet(n_channels=inchannel,n_classes=outchannel)
        self.unet_4_4 = UNet(n_channels=inchannel,n_classes=outchannel)
        self.outcov = nn.Conv2d(in_channels=4,out_channels=2,kernel_size=1)
        
    def forward(self,x):
        logits = torch.zeros(size=x.size()).cuda(Config.device)
        i1 = x[:, :, :256, :]
        i2 = x[:, :, 256:512, :]
        i3 = x[:, :, 512:768, :]
        i4 = x[:, :, 768:1024, :]

        o_full = self.unet_full(x)
        o1 = self.unet_1_4(i1)
        o2 = self.unet_2_4(i2)
        o3 = self.unet_3_4(i3)
        o4 = self.unet_4_4(i4)
        x = torch.cat((o1,o2,o3,o4),dim=2)
        logits[:, :, :1024, :] = x
        logits = torch.cat((logits,o_full),dim=1)
        return self.outcov(logits)

class UNet(nn.Module):
    if(Config.layer_numbers_unet == 4):
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
            self.down4 = Down(512, 512)
            self.up1 = Up(1024, 256, bilinear)
            self.up2 = Up(512, 128, bilinear)
            self.up3 = Up(256, 64, bilinear)
            self.up4 = Up(128, 64, bilinear)
            self.outc = OutConv(64, n_classes)

        def forward(self, x):
            x = x.permute(0,3,1,2)
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
        def __init__(self, n_channels, n_classes, bilinear=True,dropout = Config.drop_rate):
            super(UNet, self).__init__()
            self.n_channels = n_channels
            self.cnt = 0
            self.n_classes = n_classes
            self.bilinear = bilinear
            self.dropout = dropout

            self.inc = DoubleConv(n_channels, 64)
            self.down1 = Down(64, 128)
            self.down2 = Down(128, 256)
            self.down3 = Down(256, 512)
            self.down4 = Down(512, 1024)
            self.down5 = Down(1024, 2048)
            self.mid = Down(2048, 2048)
            self.up1 = Up(4096, 1024, bilinear)
            if (dropout != 0): self.drop1 = torch.nn.Dropout2d(dropout)
            self.up2 = Up(2048, 512, bilinear)
            if(dropout != 0):self.drop2 = torch.nn.Dropout2d(dropout)
            self.up3 = Up(1024, 256, bilinear)
            if (dropout != 0): self.drop3 = torch.nn.Dropout2d(dropout)
            self.up4 = Up(512, 128, bilinear)
            if (dropout != 0): self.drop4 = torch.nn.Dropout2d(dropout)
            self.up5 = Up(256, 64, bilinear)
            if (dropout != 0): self.drop5 = torch.nn.Dropout2d(dropout)
            self.up6 = Up(128, 64, bilinear)
            if (dropout != 0): self.drop6 = torch.nn.Dropout2d(dropout)

            self.outc = OutConv(64, n_classes)

        def forward(self, x):
            x1 = self.inc(x)

            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            x6 = self.down5(x5)
            x7 = self.mid(x6)
            x = self.up1(x7, x6)
            if (self.dropout != 0): x = self.drop1(x)
            x = self.up2(x, x5)
            if(self.dropout != 0):x = self.drop2(x)
            x = self.up3(x, x4)
            if (self.dropout != 0): x = self.drop3(x)
            x = self.up4(x, x3)
            if (self.dropout != 0): x = self.drop4(x)
            x = self.up5(x, x2)
            if (self.dropout != 0): x = self.drop5(x)
            x = self.up6(x, x1)
            if (self.dropout != 0): x = self.drop6(x)

            x = self.outc(x)
            return x

    # if (Config.layer_numbers_unet == 5):
    #     def __init__(self, n_channels, n_classes, bilinear=True,dropout = 0):
    #         super(UNet, self).__init__()
    #         self.n_channels = n_channels
    #         self.cnt = 0
    #         self.n_classes = n_classes
    #         self.bilinear = bilinear
    #         self.dropout = dropout
    #
    #         self.inc_out_40 = DoubleConv(n_channels, 40)
    #         self.inc_out_12 = DoubleConv(n_channels, 12)
    #         self.inc_out_8 = DoubleConv(n_channels, 8)
    #         self.inc_out_4 = DoubleConv(n_channels, 4)
    #         self.down1 = Down(64, 128)
    #         self.down2 = Down(128, 256)
    #         self.down3 = Down(256, 512)
    #         self.down4 = Down(512, 1024)
    #         self.down5 = Down(1024, 1024)
    #         # self.mid = Down(2048, 2048)
    #         # self.up1 = Up(4096, 1024, bilinear)
    #         # if (dropout != 0): self.drop1 = torch.nn.Dropout2d(dropout)
    #         self.up2 = Up(2048, 512, bilinear)
    #         if(dropout != 0):self.drop2 = torch.nn.Dropout2d(dropout)
    #         self.up3 = Up(1024, 256, bilinear)
    #         if (dropout != 0): self.drop3 = torch.nn.Dropout2d(dropout)
    #         self.up4 = Up(512, 128, bilinear)
    #         if (dropout != 0): self.drop4 = torch.nn.Dropout2d(dropout)
    #         self.up5 = Up(256, 64, bilinear)
    #         if (dropout != 0): self.drop5 = torch.nn.Dropout2d(dropout)
    #         self.up6 = Up(128, 64, bilinear)
    #         if (dropout != 0): self.drop6 = torch.nn.Dropout2d(dropout)
    #
    #         self.outc_in_40 = OutConv(40, n_classes)
    #         self.outc_in_12 = OutConv(12, n_classes)
    #         self.outc_in_8 = OutConv(8, n_classes)
    #         self.outc_in_4 = OutConv(4, n_classes)
    #
    #
    #     def forward(self, x):
    #         x = x.permute(0, 3, 1, 2)
    #         logits = torch.zeros(size=x.size()).cuda(Config.device)
    #
    #         i1 = x[:, :, :256, :]
    #         i2 = x[:, :, 256:512, :]
    #         i3 = x[:, :, 512:768, :]
    #         i4 = x[:, :, 768:1024, :]
    #
    #         o1 = self.inc_out_40(i1)
    #         o2 = self.inc_out_12(i2)
    #         o3 = self.inc_out_8(i3)
    #         o4 = self.inc_out_4(i4)
    #         x1 = torch.cat((o1,o2,o3,o4),dim = 1)
    #
    #         x2 = self.down1(x1)
    #         x3 = self.down2(x2)
    #         x4 = self.down3(x3)
    #         x5 = self.down4(x4)
    #         x6 = self.down5(x5)
    #         # x7 = self.mid(x6)
    #         # x = self.up1(x7, x6)
    #         # if (self.dropout != 0): x = self.drop1(x)
    #         x = self.up2(x6, x5)
    #         if(self.dropout != 0):x = self.drop2(x)
    #         x = self.up3(x, x4)
    #         if (self.dropout != 0): x = self.drop3(x)
    #         x = self.up4(x, x3)
    #         if (self.dropout != 0): x = self.drop4(x)
    #         x = self.up5(x, x2)
    #         if (self.dropout != 0): x = self.drop5(x)
    #         x = self.up6(x, x1)
    #         if (self.dropout != 0): x = self.drop6(x)
    #
    #         i1 = x[:, :40, :, :]
    #         i2 = x[:, 40:52, :, :]
    #         i3 = x[:, 52:60, :, :]
    #         i4 = x[:, 60:64, :, :]
    #         o1 = self.outc_in_40(i1)
    #         o2 = self.outc_in_12(i2)
    #         o3 = self.outc_in_8(i3)
    #         o4 = self.outc_in_4(i4)
    #         x = torch.cat((o1,o2,o3,o4),dim=2)
    #         logits[:, :, :1024, :] = x
    #
    #         return logits.permute(0,2,3,1)

    if (Config.layer_numbers_unet == 5):
        def __init__(self, n_channels, n_classes, bilinear=True,dropout = Config.drop_rate):
            super(UNet, self).__init__()
            self.n_channels = n_channels
            # self.cnt = 0
            self.n_classes = n_classes
            self.bilinear = bilinear
            self.dropout = dropout

            self.inc = DoubleConv(n_channels, 64)
            self.down1 = Down(64, 128)
            self.down2 = Down(128, 256)
            self.down3 = Down(256, 512)
            self.down4 = Down(512, 1024)
            self.down5 = Down(1024, 1024)
            # self.down6 = Down(2048, 2048)
            # self.up1 = Up(2048, 1024, bilinear)
            self.up2 = Up(2048, 512, bilinear)
            if(dropout != 0):self.drop2 = torch.nn.Dropout2d(dropout)
            self.up3 = Up(1024, 256, bilinear)
            if (dropout != 0): self.drop3 = torch.nn.Dropout2d(dropout)
            self.up4 = Up(512, 128, bilinear)
            if (dropout != 0): self.drop4 = torch.nn.Dropout2d(dropout)
            self.up5 = Up(256, 64, bilinear)
            if (dropout != 0): self.drop5 = torch.nn.Dropout2d(dropout)
            self.up6 = Up(128, 64, bilinear)
            if (dropout != 0): self.drop6 = torch.nn.Dropout2d(dropout)

            self.outc = OutConv(64, n_classes)


        def forward(self, x):
            x1 = self.inc(x)

            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            x6 = self.down5(x5)
            # x7 = self.down6(x6)
            # x = self.up1(x7, x6)
            x = self.up2(x6, x5)
            if(self.dropout != 0):x = self.drop2(x)
            x = self.up3(x, x4)
            if (self.dropout != 0): x = self.drop3(x)
            x = self.up4(x, x3)
            if (self.dropout != 0): x = self.drop4(x)
            x = self.up5(x, x2)
            if (self.dropout != 0): x = self.drop5(x)
            x = self.up6(x, x1)
            if (self.dropout != 0): x = self.drop6(x)
            x = self.outc(x)
            return x


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

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

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

if __name__ == "__main__":
    model = UNet(n_channels=2, n_classes=2)
    print(model)
    print(get_parameter_number(model))