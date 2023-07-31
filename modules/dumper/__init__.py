from torch import nn
import torch

from utils import fill_config

DEFAULT = {
    "dataformats": "NCHW",
    "mode": "signed_split",
    "unsqueeze_dim": 0,
    "channels_dim": 0,
    "interval": 1,
    "expand": False,
}

def cast_for_dump(tensors):
    return tensors.detach().clamp(0.0, 255.0).to(torch.uint8).cpu().numpy()

def expand(tensor):
    min_value = tensor.min()
    max_value = tensor.max()
    return tensor.sub(min_value).div(max_value - min_value).mul(255.0)

class FeaturesDumper(nn.Module):
    def __init__(self, config, writer):
        super().__init__()

        fill_config(config, DEFAULT)

        self.config = config
        self.writer = writer
        self.__last_dump_interval = self.config["interval"] - 1
        self.__iterations = 0

        mode = self.config["mode"]
        if mode == "signed_split":
            self.__dump_function = self.__signed_split_dump
        elif mode == "split":
            self.__dump_function = self.__split_dump
        elif mode == "rgb":
            self.__dump_function = self.__rgb_dump

    def __signed_split_dump(self, features):
        channels = features.unbind(self.config["channels_dim"])
        tensors = None
        unsqueeze_dim = self.config["unsqueeze_dim"]
        for channel in channels:
            negative_features = channel.neg().relu().unsqueeze(unsqueeze_dim)
            positive_features = channel.relu().unsqueeze(unsqueeze_dim)
            dummy_channel = torch.zeros(channel.shape).to(features.device).unsqueeze(unsqueeze_dim)
            
            signed_features = torch.cat([negative_features, positive_features, dummy_channel], unsqueeze_dim)

            if self.config["expand"]:
                signed_features = expand(signed_features)

            signed_features = torch.unsqueeze(signed_features, unsqueeze_dim).clone()
            if tensors is not None:
                tensors = torch.cat([tensors, signed_features], unsqueeze_dim)
            else:
                tensors = signed_features

        self.writer.add_images(self.config["tag"], cast_for_dump(tensors), self.__iterations, dataformats = self.config["dataformats"])

    def __split_dump(self, features):
        if self.config["unsqueeze_dim"]:
            features = features.unsqueeze(self.config["unsqueeze_dim"])
        
        self.writer.add_images(self.config["tag"], cast_for_dump(features), self.__iterations, dataformats = self.config["dataformats"])

    def __rgb_dump(self, features):
        self.writer.add_images(self.config["tag"], cast_for_dump(features), self.__iterations, dataformats = self.config["dataformats"])

    def dump_scalar(self, name, value):
        self.writer.add_scalar(f"{self.config['tag']}_{name}", 
                                        value, 
                                        self.__iterations)

    def forward(self, features):
        self.__iterations += 1
        self.__last_dump_interval += 1

        if self.__last_dump_interval == self.config["interval"]:
            self.__dump_function(features)
            self.__last_dump_interval = 0

        return features
