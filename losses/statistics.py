import torch

class VarianceLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, reconstructed, original):
        return (reconstructed.var() - original.var()).abs()

class MeanLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, reconstructed, original):
        return (reconstructed.mean() - original.mean()).abs()