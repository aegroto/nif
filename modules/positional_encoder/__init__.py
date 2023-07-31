import numpy as np
import math
import torch

class PositionalEncoder(torch.nn.Module):
    '''Module to add positional encoding as in NeRF [Mildenhall et al. 2020].'''

    def __init__(self, 
                 in_features, 
                 sidelength=None, 
                 use_nyquist=True, 
                 use_cache=True,
                 num_frequencies=None, 
                 scale=1.4):
        super().__init__()

        self.in_features = in_features
        self.scale = scale
        self.sidelength = sidelength
        if num_frequencies == None:
            if self.in_features == 3:
                self.num_frequencies = 10
            elif self.in_features == 2:
                assert sidelength is not None
                if isinstance(sidelength, int):
                    sidelength = (sidelength, sidelength)
                self.num_frequencies = 4
                if use_nyquist:
                    self.num_frequencies = self.get_num_frequencies_nyquist(min(sidelength[0], sidelength[1]))
        else:
            self.num_frequencies = num_frequencies

        self.out_dim = in_features + in_features * 2 * self.num_frequencies  # (sum(self.frequencies_per_axis))

        if use_cache:
            self.coords_cache = dict()
        else:
            self.coords_cache = None

    def get_num_frequencies_nyquist(self, samples):
        nyquist_rate = 1 / (2 * (2 * 1 / samples))
        return int(math.floor(math.log(nyquist_rate, 2)))

    def __encode(self, coords):
        coords_pos_enc = coords
        for i in range(self.num_frequencies):

            for j in range(self.in_features):
                c = coords[..., j]

                sin = torch.unsqueeze(torch.sin((self.scale ** i) * np.pi * c), -1)
                cos = torch.unsqueeze(torch.cos((self.scale ** i) * np.pi * c), -1)

                coords_pos_enc = torch.cat((coords_pos_enc, sin, cos), axis=-1)

        return coords_pos_enc

    def forward(self, coords):
        if self.coords_cache:
            tensor_id = coords.data_ptr()
            if not (tensor_id in self.coords_cache):
                self.coords_cache[tensor_id] = self.__encode(coords)

            return self.coords_cache[tensor_id]
        else:
            return self.__encode(coords)


