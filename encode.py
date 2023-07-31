import torch
import sys
from compress import compress_state_dict

from fit import fit
from quantize import quantize
from serialization import serialize_state_dict

def main():
    torch.random.manual_seed(1337)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)

    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)

    config_path = sys.argv[1] 
    uncompressed_path = sys.argv[2] 
    compressed_path = sys.argv[3] 

    uncompressed_state_dict = fit(config_path, uncompressed_path)
    fp_quantized_state_dict = quantize(config_path, uncompressed_path, uncompressed_state_dict)
    compressed_state_dict = compress_state_dict(fp_quantized_state_dict)

    serialize_state_dict(compressed_state_dict, compressed_path)

if __name__ == "__main__":
    main()
