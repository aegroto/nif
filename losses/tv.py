import torch

class TotalVariationLoss:
    def __init__(self, weight=1.0):
        self.weight = weight

    def tv(self, img):
        tv_h = ((img[:,1:,:] - img[:,:-1,:]).pow(2)).mean()
        tv_w = ((img[:,:,1:] - img[:,:,:-1]).pow(2)).mean()    
        return self.weight * (tv_h + tv_w)

    def __call__(self, reconstructed, original):
        return torch.abs(self.tv(original) - self.tv(reconstructed))
