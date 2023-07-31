import sys
import numpy as np
import math
import torch

from utils import PI 

def generate_grid(config, width, height, device):
    grid = generate_coordinates_tensor(width, height, device)
    return grid

def generate_coordinates_tensor(width, height, device):
    coordinates = torch.cartesian_prod(
        torch.arange(0.0, float(height)) / (height - 1.0),
        torch.arange(0.0, float(width)) / (width - 1.0),
    ).to(device)
    coordinates = coordinates.sub(0.5).mul(2.0)

    coordinates = coordinates.unflatten(0, (height, width))

    return coordinates

def one_hot_index(index, size, device):
    tensor = torch.zeros(size, device=device)
    tensor[index] = 1.0
    return tensor

def generate_components_tensor(scale, full_width, full_height, device):
    quads = scale ** 2
    width = full_width // scale
    height = full_height // scale
    
    q_indices = list()
    for q in range(0, quads):
        # index = one_hot_index(q, quads, device)
        index = torch.tensor([float(q) / float(quads)], device=device)
        q_indices.append(index)

    w_indices = list()
    for w in range(0, width):
        index = torch.tensor([float(w) / float(width)], device=device)
        w_indices.append(index)

    h_indices = list()
    for h in range(0, height):
        index = torch.tensor([float(h) / float(height)], device=device)
        h_indices.append(index)

    components_list = list()
    for q_index in q_indices:
        for w_index in w_indices:
            for h_index in h_indices:
                component = torch.cat([q_index, w_index, h_index]).unsqueeze(0)
                components_list.append(component)

    components = torch.cat(components_list)

    components = components.unflatten(0, (quads, width, height))

    print(components.shape)

    return components

def positional_encode(input_features, num_frequencies, scale):
    features = input_features
    for i in range(0, num_frequencies):
        w = (scale ** i) * PI

        sin_features = torch.sin(input_features * w)
        cos_features = torch.cos(input_features * w)

        if features is None:
            features = torch.cat((sin_features, cos_features), -1)
        else:
            features = torch.cat((features, sin_features, cos_features), -1)
        
    return features

def gaussian_encode(input_features, size, dev, scale):
    gaussian_matrix = torch.normal(0, dev, (size, input_features.size(-1)), device=input_features.device)
    gaussian_matrix *= scale

    pi_features = 2.0 * PI * input_features
    sampled_features = pi_features.matmul(gaussian_matrix.transpose(0, 1))
    features = torch.hstack([sampled_features.sin(), sampled_features.cos()])
    return features

def grid_to_vector(grid):
    return grid.flatten(-2, -1).transpose(0, 1)

def vector_to_grid(vector, width, height):
    return vector.transpose(0, 1).unflatten(-1, (width, height))

def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.'''
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if dim == 2:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (sidelen[0] - 1)
        pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / (sidelen[1] - 1)
    elif dim == 3:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1], :sidelen[2]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)

    pixel_coords -= 0.5
    pixel_coords *= 2.
    pixel_coords = torch.Tensor(pixel_coords).view(-1, dim)
    return pixel_coords

