import math
import torch
from torch import nn

class Sine(nn.Module):
    def __init__(self, omega=30.0):
        super().__init__()
        self.omega = omega

    def forward(self, x):
        return torch.sin(self.omega * x)

    def __repr__(self):
        return f"Sine({self.omega})"

class Cosine(nn.Module):
    def __init__(self, omega=30.0):
        super().__init__()
        self.omega = omega

    def forward(self, x):
        return torch.cos(self.omega * x)

    def __repr__(self):
        return f"Cosine({self.omega})"

def initialize_first_layer(layer):
    if hasattr(layer, "weight"):
        w_std = 1.0 / layer.in_features
        nn.init.uniform_(layer.weight, -w_std, w_std)

def build_linear_initializer(c, w0):
    def __init(layer):
        if hasattr(layer, "weight"):
            w_std = math.sqrt(c / layer.in_features) / w0
            nn.init.uniform_(layer.weight, -w_std, w_std)
    return __init
