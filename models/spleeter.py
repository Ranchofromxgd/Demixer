import sys
from config.wavenetConfig import Config
sys.path.append("/home/disk2/internship_anytime/liuhaohe/he_workspace/github/music_separator/")
from models.unet_model import UNet
from models.demixer import Demixer
from torch import nn

from config.wavenetConfig import Config
class Spleeter(nn.Module):
    def __init__(self,
                 channels = 2,
                 unet_inchannels = 1,
                 unet_outchannels = 1,
                 # load_model_from = "",
                 # use_cpu = False
                 ):
        super(Spleeter, self).__init__()
        self.channels = Config.channels
        self.cnt = 0
        self.sigmoid = nn.Sigmoid()
        for channel in range(channels):
            if(Config.model_name == "Unet"):model = UNet(n_channels = unet_inchannels,n_classes = unet_outchannels)
            elif(Config.model_name == "Demixer"):model = Demixer(n_channels = unet_inchannels,n_classes = unet_outchannels)
            else:raise ValueError("Error: Non-exist model name")
            if(Config.device == 'cpu'):exec("self.unet{}=model".format(channel))
            else:exec("self.unet{}=model.cuda(Config.device)".format(channel))
            # else:exec("self.unet{}=model".format(channel))

    def forward(self,track_i,zxx):
        # [Batchsize,channels,x,y]
        # out = [model.forward(zxx) for model in self.models]
        layer = self.__dict__['_modules']['unet'+str(track_i)]
        out = layer.forward(zxx)
        return  self.sigmoid(out) if(Config.OUTPUT_MASK) else out
