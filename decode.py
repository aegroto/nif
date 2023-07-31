from skimage import io
import torch
import sys
from decompress import decompress_state_dict
from infer import infer

from serialization import deserialize_state_dict
from utils import load_device

def main():
    torch.random.manual_seed(1337)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)

    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)

    config_path = sys.argv[1] 
    compressed_path = sys.argv[2] 
    decoded_path = sys.argv[3] 

    compressed_state_dict = deserialize_state_dict(compressed_path)
    decompressed_state_dict = decompress_state_dict(compressed_state_dict, device=load_device())

    reconstructed_image = infer(config_path, decompressed_state_dict)
    io.imsave(decoded_path, reconstructed_image)

if __name__ == "__main__":
    main()
