import torch

class ScaledMSELoss:
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, img1, img2):
        img1 = img1 * self.scale
        img2 = img2 * self.scale
        return torch.mean((img1 - img2) ** 2)
