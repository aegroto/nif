from torch import nn
from losses.composite import CompositeLoss
from losses.log_cosh import LogCoshLoss
from losses.ssim import DMS_SSIMLoss, DSSIMLoss

def loss_from_id(id):
    if id == "l1":
        return nn.L1Loss()
    if id == "mse":
        return nn.MSELoss()
    if id == "log_cosh":
        return LogCoshLoss()
    if id == "ssim":
        return DSSIMLoss()
    if id == "ms_ssim":
        return DMS_SSIMLoss()

def build_loss_fn(config):
    losses = list()
    for loss_config in config["components"]:
        losses.append((loss_from_id(loss_config["type"]), loss_config["weight"]))

    return CompositeLoss(losses)