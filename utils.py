import yaml
from scipy.stats import entropy
import numpy
import sys
import torch

PI = torch.acos(torch.zeros(1)).item() * 2

def load_configuration(path):
    config = yaml.safe_load(open(path, "r"))
    fill_configurations(config)

    return config

# from https://stackoverflow.com/questions/45335445/how-to-recursively-replace-dictionary-values-with-a-matching-key
def replace_config(config, default_config):
    for key, default_value in default_config.items():
        if key not in config:
            config[key] = default_value
        else:
            if isinstance(default_value, dict):
                replace_config(config[key], default_value)

def fill_configurations(config):
    default_config = yaml.safe_load(open("configurations/__default.yaml", "r"))
    replace_config(config, default_config)

def load_device(force_cpu=False):
    if not force_cpu and torch.cuda.is_available():
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")

def shannon_entropy(labels):
    _, counts = numpy.unique(labels, return_counts=True)
    return entropy(counts, base=2)

def state_dict_stats(state_dict):
    for (key, tensor) in state_dict.items():
        print(f"{key}: {shannon_entropy(tensor.clone().cpu())}")

def clearlines(n):
    for _ in range(n):
        sys.stdout.write("\033[F")

def printdots(n):
    for _ in range(n):
        sys.stdout.write("#\n")

def fill_config(config, default):
    for key, value in default.items():
        if key not in config:
            config[key] = value

def calculate_model_size(model, verbose=True):
    total_param_size = 0
    for name, param in model.named_parameters():
        param_size = param.nelement() * param.element_size()
        if verbose:
            print(f"{name}: {param_size}")
        total_param_size += param_size

    total_buffer_size = 0
    for name, buffer in model.named_buffers():
        buffer_size = buffer.nelement() * buffer.element_size()
        if verbose:
            print(f"{name}: {buffer_size}")
        total_buffer_size += buffer_size

    size_all = (total_param_size + total_buffer_size)
    return size_all

def calculate_state_dict_size(state_dict):
    total_param_size = 0
    for (name, param) in state_dict.items():
        if type(param) is dict:
            param_size = calculate_state_dict_size(param)
        elif type(param) is torch.Tensor: 
            param_size = param.nelement() * param.element_size()
        elif type(param) is bytes: 
            param_size = len(param)
        elif type(param) is int: 
            param_size = 4
        elif type(param) is float: 
            param_size = 4
        elif type(param) is numpy.float16: 
            param_size = 2
        elif type(param) is numpy.uint8: 
            param_size = 1
        elif type(param) is bool: 
            param_size = 1
        else:
            print(f"WARNING: Unsupported param format: {name}, {type(param)}")
            param_size = 0

        print(f"{name}: {param_size}")

        total_param_size += param_size

    size_all = (total_param_size)
    return size_all

def cast_for_dump(tensor):
    return rescale_img(tensor.detach()).mul(255.0).to(torch.uint8).cpu().numpy()

def replace_macros(config, macros):
    if not isinstance(config, dict):
        return

    for (key, value) in config.items():
        if isinstance(value, dict):
            replace_macros(value, macros)
        elif isinstance(value, list):
            for i in range(0, len(value)):
                item = value[i]
                if isinstance(item, str) and item in macros:
                    value[i] = macros[item]
                else:
                    replace_macros(item, macros)
        else:
            if isinstance(value, str) and value in macros:
                config[key] = macros[value]

def get_parameters(module, pruning_list):
    for submodule in module:
        try:
            _ = iter(submodule)
            get_parameters(submodule, pruning_list)
        except TypeError as e:
            for submodule in submodule.modules():
                if hasattr(submodule, "weight") and submodule.weight is not None:
                    pruning_list.append((submodule, "weight"))
                if hasattr(submodule, "bias") and submodule.bias is not None:
                    pruning_list.append((submodule, "bias"))


def dump_model_stats(model, width, height, writer):
    model_size = calculate_model_size(model)

    writer.add_text("model_size", f"{model_size:,} bytes")
    writer.add_text("models", "```\n" + str(model).replace('\n', '\n\n') + "\n```")

    pixels_count = width * height 

    bpp = (model_size * 8) / pixels_count
    writer.add_text("bpp", f"{bpp:.3f} bits per pixel")

    print(f"model size: {model_size}")
    print(f"bpp: {bpp}")

def lin2img(tensor, image_resolution=None):
    batch_size, num_samples, channels = tensor.shape
    if image_resolution is None:
        width = numpy.sqrt(num_samples).astype(int)
        height = width
    else:
        height = image_resolution[0]
        width = image_resolution[1]

    return tensor.permute(0, 2, 1).view(batch_size, channels, height, width)

def rescale_img(x, mode='scale', tmax=1.0, tmin=0.0):
    if (mode == 'scale'):
        xmax = torch.max(x)
        xmin = torch.min(x)
        if xmin == xmax:
            return 0.5 * torch.ones_like(x) * (tmax - tmin) + tmin
        x = ((x - xmin) / (xmax - xmin)) * (tmax - tmin) + tmin
    elif (mode == 'clamp'):
        x = torch.clamp(x, 0, 1)
    return x


def linear_reduction(start, end, x):
    return start + (end - start) * x

# def pad_measure(measure, patching):
#     if measure % patching == 0:
#         return 0

#     target_measure = (measure // patching + 1) * patching
#     padding = target_measure - measure
#     return padding

def pad_measure(measure, patching):
    return ((((measure + patching - 1) // patching) * patching) - measure)

def pad_for_patching(tensor, patching):
    height = tensor.size(-2)
    width = tensor.size(-1)

    height_padding = pad_measure(height, patching)
    width_padding = pad_measure(width, patching)

    padding = (
        width_padding // 2, 
        width_padding // 2 + width_padding % 2, 
        height_padding // 2, 
        height_padding // 2 + width_padding % 2
    )

    padded_tensor = torch.nn.functional.pad(tensor, padding, "replicate")

    return padded_tensor