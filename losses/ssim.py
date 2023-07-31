from pytorch_msssim import SSIM, MS_SSIM

def reshape(tensor):
    return tensor.unsqueeze(0).div(2.0).add(0.5).mul(255.0)

class DSSIMLoss:
    def __init__(self, data_range=255, win_size=11):
        self.__ssim = SSIM(data_range=data_range, win_size=win_size)

    def __call__(self, original, distorted):
        ssim = self.__ssim(reshape(original), reshape(distorted))
        return (1.0 - ssim)

class DMS_SSIMLoss:
    def __init__(self, data_range=255, win_size=11):
        self.__ms_ssim = MS_SSIM(data_range=data_range, win_size=win_size)

    def __call__(self, original, distorted):
        ms_ssim = self.__ms_ssim(reshape(original), reshape(distorted))
        return (1.0 - ms_ssim)
