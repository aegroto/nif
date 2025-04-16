import os
import json
import torch
import sys

from skimage import io
from losses.psnr import psnr

from pytorch_msssim import ms_ssim, ssim
from serialization import deserialize_state_dict

from utils import calculate_state_dict_size, load_device

def ms_ssim_reshape(tensor):
    return tensor.movedim(-1, 0).unsqueeze(0)

def main():
    print("Loading device...")
    device = load_device(True)

    print("Loading parameters...")
    original_file_path = sys.argv[1]
    reconstructed_file_path = sys.argv[2]
    stats_path = sys.argv[3]
    compressed_file_path = sys.argv[4]

    print("Calculating compressed state size...")
    compressed_file_size = os.stat(compressed_file_path).st_size


    print("Loading images...")
    original_image_tensor = torch.from_numpy(io.imread(original_file_path)).to(device).to(torch.float32)
    reconstructed_image_tensor = torch.from_numpy(io.imread(reconstructed_file_path)).to(device).to(torch.float32)

    pixels = original_image_tensor.nelement() / 3.0

    print("Calculating stats...")

    try:
        ssim_value = ssim(ms_ssim_reshape(original_image_tensor), ms_ssim_reshape(reconstructed_image_tensor)).item()
    except Exception as e:
        print(f"Cannot calculate SSIM: {e}")
        ssim_value = None

    try:
        ms_ssim_value = ms_ssim(ms_ssim_reshape(original_image_tensor), ms_ssim_reshape(reconstructed_image_tensor)).item()
    except Exception as e:
        print(f"Cannot calculate MS-SSIM: {e}")
        ms_ssim_value = None

    try:
        compressed_state_dict = deserialize_state_dict(compressed_file_path)
    except Exception as e:
        print(f"WARNING: Cannot deserialize compressed state dict: {e}")
        compressed_state_dict = None

    try:
        if compressed_state_dict is None:
            compressed_state_dict = torch.load(compressed_file_path)

        compressed_state_size = calculate_state_dict_size(compressed_state_dict)
    except Exception as e:
        print(f"WARNING: Cannot calculate state-only bpp: {e}")
        compressed_state_size = 0 

    stats = {
        "psnr": psnr(original_image_tensor, reconstructed_image_tensor).item(),
        "ms-ssim": ms_ssim_value,
        "ssim": ssim_value,
        "bpp": (compressed_file_size * 8) / pixels,
        "state_bpp": (compressed_state_size * 8) / pixels
    }

    try:
        stats["state_size"] = calculate_state_dict_size(torch.load(compressed_file_path))
    except Exception as e:
        pass

    print(stats)

    json.dump(stats, open(stats_path, "w"), indent=4)

if __name__ == "__main__":
    main()

