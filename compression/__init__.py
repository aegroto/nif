import torch

from compression.utils import torch_type_for_bits

class Calibration():
    def __init__(self, bits, bound, device):
        self.max_symbol = torch.Tensor([2**(bits-1) - 1]).to(device).to(torch.float16)
        self.bound = torch.Tensor([bound]).to(device).to(torch.float16)

    def __repr__(self):
        return f"({self.max_symbol}, {self.bound})"

def quantize_tensor(tensor, bits, bound):
    calibration = Calibration(bits, bound, tensor.device)

    normalized_tensor = tensor.clone()
    normalized_tensor.clamp_(-calibration.bound, calibration.bound)
    normalized_tensor.div_(calibration.bound)

    quantized_tensor = normalized_tensor.to(torch.float16).mul(calibration.max_symbol)
    quantized_tensor = quantized_tensor.to(torch_type_for_bits(bits))

    return quantized_tensor

def dequantize_tensor(tensor, bits, bound):
    calibration = Calibration(bits, bound, tensor.device)

    quantized_tensor = tensor.clone()
    quantized_tensor = quantized_tensor.to(torch.float16)
    
    normalized_tensor = quantized_tensor.div(calibration.max_symbol)
    dequantized_tensor = normalized_tensor.mul(calibration.bound)

    dequantized_tensor = dequantized_tensor.to(torch.float32)

    return dequantized_tensor
