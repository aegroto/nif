import copy
import brotli

import torch
import sys

from compression import quantize_tensor
from serialization import serialize_state_dict

def size(tensor):
    return tensor.element_size() * tensor.nelement()

def compress_state_dict(state_dict):
    quantization_config = state_dict["quantization_config"]
    del state_dict["quantization_config"]

    metadata = copy.deepcopy(state_dict["__meta"])
    del state_dict["__meta"]

    print("Compressing...")
    compressed_state_dict = dict()
    compressed_state_dict["quantization_config"] = quantization_config

    total_uncompressed = 0
    total_compressed = 0

    for (key, tensor) in state_dict.items():
        tensor_config = {
            "bits": quantization_config[key]["bits"],
            "bound": quantization_config[key]["bound"]
        }

        quantized_tensor = quantize_tensor(tensor, tensor_config["bits"], tensor_config["bound"])

        array = quantized_tensor.cpu().numpy()
        buffer = array.tobytes()
        compressed = brotli.compress(buffer, lgwin=10)
        compressed_state_dict[key] = compressed

        uncompressed_size = size(tensor)
        compressed_size = len(compressed)

        total_uncompressed += uncompressed_size
        total_compressed += compressed_size

    print(f"Total uncompressed: {total_uncompressed}")
    print(f"Total compressed: {total_compressed}")
    print(f"Ratio: {total_uncompressed / total_compressed}")

    compressed_state_dict["__meta"] = metadata
    return compressed_state_dict

if __name__ == "__main__":
    model_dump_path = sys.argv[1]
    compressed_model_dump_path = sys.argv[2]

    print("Loading state dict...")
    state_dict = torch.load(model_dump_path)
    compressed_state_dict = compress_state_dict(state_dict)

    serialize_state_dict(compressed_state_dict, compressed_model_dump_path)

