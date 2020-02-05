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

class Demixer(nn.Module):
    if (Config.layer_numbers_unet == 3):
        def __init__(self, n_channels,
                     n_classes,
                     bilinear=True):
            super(Demixer, self).__init__()
            self.n_channels = n_channels
            self.cnt = 0
            self.n_classes = n_classes
            self.bilinear = bilinear
            self.increase_rate = Config.dense_block*Config.dense_layers*Config.dense_growth_rate

            self.inc = DoubleConv(self.n_channels,self.getCh(1),TIF_in=1025)
            self.down1 = Down(self.getCh(1),self.getCh(2),TIF_in=512)
            self.down2 = Down(self.getCh(2),self.getCh(3),TIF_in=256)
            self.down3 = Down(self.getCh(3), self.getCh(4),TIF_in=128)
            self.down4 = Down(self.getCh(4), self.getCh(5),TIF_in=64)
            self.middle = DoubleConv(self.getCh(5), self.getCh(5),TIF_in=0)
            self.up1 = Up(self.getCh(4,5), self.getCh(4), bilinear = True,TIF_in=128)
            self.up2 = Up(self.getCh(3,4), self.getCh(3), bilinear = True,TIF_in=256)
            self.up3 = Up(self.getCh(2,3), self.getCh(2), bilinear = True,TIF_in = 512)
            self.up4 = Up(self.getCh(1,2), self.getCh(1), bilinear = True,TIF_in = 1025)
            self.outc = OutConv(self.getCh(1), 64)
            self.outc2 = OutConv(64, n_classes)

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
            # x6 = self.middle(x5)
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
            logits = self.outc(x)
            logits = self.outc2(logits)
            return logits.permute(0,2,3,1)

    if (Config.layer_numbers_unet == 5):
        def __init__(self, n_channels = 2, n_classes = 2, bilinear=True,mmask = False):
            super(Demixer, self).__init__()
            self.n_channels = n_channels
            self.cnt = 0
            self.n_classes = n_classes
            self.bilinear = bilinear
            self.first_ch = 64
            self.increase_rate = Config.dense_block*Config.dense_layers*Config.dense_growth_rate
            self.inc = DoubleConv(self.n_channels,self.getCh(1),TIF_in=1025)
            self.down1 = Down(self.getCh(1),self.getCh(2),TIF_in=512)
            self.down2 = Down(self.getCh(2),self.getCh(3),TIF_in=256)
            self.down3 = Down(self.getCh(3),self.getCh(4),TIF_in=128)
            self.down4 = Down(self.getCh(4),self.getCh(5),TIF_in=64)
            self.down5 = Down(self.getCh(5), self.getCh(6),TIF_in=32)
            # self.down6 = Down(self.getCh(6), self.getCh(6),TIF_in=0)
            self.middle1 = nn.Conv2d(self.getCh(6), self.getCh(6)*2, kernel_size=3, padding=1)
            self.middle2 = nn.Conv2d(self.getCh(6)*2, self.getCh(6), kernel_size=3, padding=1)
            self.middle3 = nn.Conv2d(self.getCh(6), self.getCh(6)*2, kernel_size=3, padding=1)
            self.middle4 = nn.Conv2d(self.getCh(6)*2, self.getCh(6), kernel_size=3, padding=1)
            self.up1 = Up(self.getCh(5,6), 512, bilinear = True,TIF_in = 64)
            self.up2 = Up(self.getCh(4)+512, 256, bilinear = True,TIF_in = 128)
            self.up3 = Up(self.getCh(3)+256, 128, bilinear = True,TIF_in = 256)
            self.up4 = Up(self.getCh(2)+128, 64, bilinear = True,TIF_in = 512)
            self.up5 = Up(self.getCh(1)+64, 64, bilinear = True,TIF_in = 1025)
            self.outc = OutConv(64, n_classes)
            # if(mmask == True):
            #     self.mmask =

        def getCh(self,n,n2 = None):
            if(n2 == None):
                if(n == 1):
                    return self.first_ch
                else:
                    return self.first_ch+(n-1)*self.increase_rate
            else:
                return self.getCh(n)+self.getCh(n2)

        def forward(self, x):
            x = x.permute(0, 3, 1, 2)
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            x6 = self.down5(x5)
            m1 = self.middle1(x6)
            m2 = self.middle2(m1)+x6
            m3 = self.middle3(m2)
            m4 = self.middle4(m3)+m2
            x = self.up1(m4, x5)
            x = self.up2(x, x4)
            x = self.up3(x, x3)
            x = self.up4(x, x2)
            x = self.up5(x, x1)
            logits = self.outc(x)
            return logits.permute(0,2,3,1)

class RNNblock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.rnn = nn.GRU(input_size=in_channels,hidden_size=in_channels)
        # self.rnn2 = nn.GRU(input_size=2*in_channels,hidden_size=in_channels)
    # Input: (N,ch, x, y)
    def forward(self, x):
        x = x.permute(0,1,3,2)
        sz = x.size()
        x = torch.reshape(x,shape=(sz[0],sz[1]*sz[2],sz[3]))
        x,_ = self.rnn(x)
        # x,_ = self.rnn2(x)
        x = torch.reshape(x,shape=(sz[0],sz[1],sz[2],sz[3]))
        return x.permute(0,1,3,2)

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

class DoubleConv(nn.Module):
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
        # self.TIF_in = TIF_in
        # if(self.TIF_in!=0):self.tif = TIFblock(TIF_in,out_channels)

    def forward(self, x):
        res = self.double_conv(x)
        return  res #   self.tif(res)+res if(self.TIF_in!=0) else

class DenseConv(nn.Module):
    def __init__(self, in_channels, out_channels,TIF_in = 2048):
        super().__init__()
        for i in range(Config.dense_block):
            exec('''self.dense_block{}=_DenseBlock(
                num_layers=Config.dense_layers,
                num_input_features=in_channels+i*(Config.dense_growth_rate*Config.dense_layers),
                bn_size=Config.dense_bn,
                growth_rate=Config.dense_growth_rate,
                drop_rate=Config.drop_rate,
                efficient=False,
                )'''.format(i))
        self.TIF_in = TIF_in
        if(TIF_in != 0):
            if(Config.block == "DNN"):
                self.tif = TIFblock(TIF_in,out_channels)
            elif(Config.block == "GRU"):
                self.tif = RNNblock(TIF_in)
            else:
                raise ValueError("Error: "+Config.block+" non exist")

    def forward(self, x):
        x0 = x
        retVal = None
        for i in range(1,Config.dense_block+1):
            # print("x{} = self.dense_block{}(x{})".format(i,i-1,i-1))
            exec("x{} = self.dense_block{}(x{})".format(i,i-1,i-1))
        names = locals()
        _x = names.get('x' + str(Config.dense_block))
        return _x  + self.tif(_x) if(self.TIF_in != 0) else _x

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels,TIF_in):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DenseConv(in_channels, out_channels,TIF_in)
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

        self.conv = DoubleConv(in_channels, out_channels,TIF_in = TIF_in)

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
    net = Demixer()
    print(net)
    print(get_parameter_number(net))

