import sys
from config.mainConfig import Config
sys.path.append("..")
from models.unet_model import UNet,MbandUnet
from models.demixer import Demixer
from models.denseunet import DenseUnet
from torch import nn
from models.demixer import RNNblock

from config.mainConfig import Config

# class Mmasker(nn.Module):
#     def __init__(self):
#         super(Mmasker, self).__init__()
#         self.dnn = nn.Sequential(
#             nn.Linear(in_features=1025,out_features=512),
#             nn.ReLU(inplace=True),
#             nn.Linear(in_features=512, out_features=256),
#             nn.ReLU(inplace=True),
#             nn.Linear(in_features=256, out_features=128),
#             nn.ReLU(inplace=True),
#             nn.Linear(in_features=128, out_features=64),
#             nn.ReLU(inplace=True),
#             nn.Linear(in_features=64, out_features=1),
#             nn.Sigmoid(),
#         )
#     def forward(self, x):
#         in_ = x.permute(0,3,2,1)
#         return x*(self.dnn(in_).permute(0,3,2,1))

class Mmasker(nn.Module):
    def __init__(self):
        super(Mmasker, self).__init__()
        self.mmask = nn.Sequential(
            nn.Conv1d(in_channels=1025, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=256, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=64, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=16, out_channels=4, kernel_size=3, padding=1),
            nn.BatchNorm1d(4),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=4, out_channels=1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        sz = x.size()
        in_ = x.reshape(sz[0],sz[1],-1)
        mmask = self.mmask(in_).reshape(sz[0],1,sz[2],sz[3])
        return mmask*x

# # torch.Size([1, 1025, 376, 2])
class Spleeter(nn.Module):
    def __init__(self,
                 channels = 2,
                 unet_inchannels = 1,
                 unet_outchannels = 1,
                 use_cpu = False
                 ):
        super(Spleeter, self).__init__()
        self.channels = Config.channels
        self.cnt = 0
        self.sigmoid = nn.Sigmoid()
        # self.mmask = Mmasker()
        for channel in range(channels):
            if(Config.model_name == "Unet"):model = UNet(n_channels = unet_inchannels,n_classes = unet_outchannels)
            elif(Config.model_name == "Demixer"):model = Demixer(n_channels = unet_inchannels,n_classes = unet_outchannels)
            elif(Config.model_name == "DenseUnet"):model = DenseUnet(n_channels_in=unet_inchannels,
                                                                     n_channel_out=unet_outchannels,
                                                                     block_number=Config.dense_block,
                                                                     denselayers=Config.dense_layers,
                                                                     bn_size=Config.dense_bn,
                                                                     growth_rate=Config.dense_growth_rate,
                                                                     dropout=Config.drop_rate)
            elif(Config.model_name == "MbandUnet"):model = MbandUnet(inchannel=unet_inchannels,outchannel=unet_outchannels)
            else:raise ValueError("Error: Non-exist model name")
            if(use_cpu):exec("self.unet{}=model".format(channel))
            else:exec("self.unet{}=model.cuda(Config.device)".format(channel))
            # else:exec("self.unet{}=model".format(channel))

    def forward(self,track_i,zxx):
        # [Batchsize,channels,x,y]
        # out = [model.forward(zxx) for model in self.models]
        layer = self.__dict__['_modules']['unet'+str(track_i)]
        out = layer(zxx)
        return self.sigmoid(out)
