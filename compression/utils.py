import torch
import numpy

def numpy_type_for_bits(bits):
    if bits <= 8:
        return numpy.int8
    if bits <= 16:
        return numpy.int16
    else:
        return numpy.int32

def torch_type_for_bits(bits):
    if bits <= 8:
        return torch.int8
    if bits <= 16:
        return torch.int16
    else:
        return torch.int32

