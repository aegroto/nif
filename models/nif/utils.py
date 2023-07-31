from torch import nn

from modules.dumper import FeaturesDumper

class Reshaper(nn.Module):
    def __init__(self):
        super(Reshaper, self).__init__()

    def set_target_shape(self, channels, width, height):
        self.target_shape = (channels, 1, width, height)

    def forward(self, x):
        return x.transpose(-1, -2).reshape(self.target_shape)

def build_dumper(writer, tag):
    if writer:
        return FeaturesDumper({
            "tag": tag,
            "mode": "signed_split",
            "interval": 100,
            "channels_dim": 0,
            "unsqueeze_dim": 0,
            "expand": True,
            "dataformats": "NCHW"
        }, writer)
    else:
        return None
