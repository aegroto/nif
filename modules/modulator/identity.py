import torch
from torch import nn

class IdentityModulator(nn.Module):
    def __init__(self):
        super(IdentityModulator, self).__init__()

    def initialize_weights(self):
        pass

    def group_features(self, y):
        return y

    def ungroup_features(self, y):
        return y

    def forward(self, x):
        return torch.tensor([0.0], device=x.device)
