import copy
import numpy
import brotli

import torch
import sys

from compression import dequantize_tensor
from compression.utils import numpy_type_for_bits
from serialization import deserialize_state_dict
from utils import load_device

def decompress_state_dict(compressed_state_dict, device=None):
    print("Decompressing...")
    decompressed_state_dict = dict()

    quantization_config = compressed_state_dict["quantization_config"]
    del compressed_state_dict["quantization_config"]

    metadata = copy.deepcopy(compressed_state_dict["__meta"])
    del compressed_state_dict["__meta"]

    for (key, compressed) in compressed_state_dict.items():

        bound = quantization_config[key]["bound"]
        bits = quantization_config[key]["bits"]

        buffer = brotli.decompress(compressed)
        array = numpy.frombuffer(buffer, numpy_type_for_bits(bits)).copy()
        quantized_tensor = torch.from_numpy(array).to(device)
        dequantized_tensor = dequantize_tensor(quantized_tensor, bits, bound)

        decompressed_state_dict[key] = dequantized_tensor

    decompressed_state_dict["__meta"] = metadata

    return decompressed_state_dict

if __name__ == "__main__":
    compressed_state_dict_path = sys.argv[1]
    decompressed_state_dict_path = sys.argv[2]

    print("Loading compressed state dict...")
    # compressed_state_dict = torch.load(compressed_state_dict_path)
    compressed_state_dict = deserialize_state_dict(compressed_state_dict_path)

    decompressed_state_dict = decompress_state_dict(compressed_state_dict, device=load_device())

    torch.save(decompressed_state_dict, decompressed_state_dict_path)

