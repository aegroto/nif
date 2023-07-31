import numpy
import os
import json
import torch
import sys

from skimage import io
from losses.psnr import psnr

from pytorch_msssim import ms_ssim, ssim

from utils import calculate_state_dict_size, load_device

from torch._C import dtype
from typing import Dict

def ms_ssim_reshape(tensor):
    return tensor.movedim(-1, 0).unsqueeze(0)

def main():
    print("Loading device...")
    device = load_device()

    print("Loading parameters...")
    original_file_path = sys.argv[1]
    reconstructed_file_path = sys.argv[2]
    stats_path = sys.argv[3]
    compressed_state_path = sys.argv[4]

    print("Calculating compressed state size...")
    state_dict = torch.load(compressed_state_path)

    print("Loading images...")
    original_image_tensor = torch.from_numpy(io.imread(original_file_path)).to(device).to(torch.float32)
    reconstructed_image_tensor = torch.from_numpy(io.imread(reconstructed_file_path)).to(device).to(torch.float32)

    print("Calculating stats...")
    stats = {
        "psnr": psnr(original_image_tensor, reconstructed_image_tensor).item(),
        "ms-ssim": None,
        "ssim": ssim(ms_ssim_reshape(original_image_tensor), ms_ssim_reshape(reconstructed_image_tensor)).item(),
        "bpp": None
    }

    print(stats)

    json.dump(stats, open(stats_path, "w"))

if __name__ == "__main__":
    main()

