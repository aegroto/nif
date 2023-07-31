import torch
import yaml
import random
import copy
import hashlib

from models.nif import NIF
from utils import calculate_model_size, load_configuration

def sizes_sequence(first_size, amount, min_bound=16):
    sizes = list()
    sizes.append(first_size)
    for i in range(amount):
        min_size = int(sizes[i] * 0.4)
        max_size = int(sizes[i] * 0.8)
        sizes.append(max(random.randint(min_size, max_size), min_bound))
    return sizes

def main():
    base_conf = load_configuration("configurations/default.yaml")

    for pos_features in range(4, 15):
        conf = copy.deepcopy(base_conf)

        conf["model"]["encoder_params"]["num_frequencies"] = pos_features

        conf_name = hashlib.sha1(str(conf).encode("UTF-8")).hexdigest()[:8]
        yaml.safe_dump(conf, open(f"configurations/.tuning/{pos_features}.yaml", "w"))

    return

    valid_configurations = 0
    while valid_configurations < 100:
        conf = copy.deepcopy(base_conf)

        # conf["model"]["hidden_sizes"] = sizes_sequence(random.randint(20, 150), random.randint(2, 5))

        conf["model"]["modulator_params"]["hidden_sizes"] = \
            sizes_sequence(random.randint(12, 64), random.randint(1, 4), min_bound=12)

        bpp = estimated_uncompressed_bpp(conf) / 4.5
        if bpp < 2.7 or bpp > 3.1:
            print(f"invalid bpp: {bpp}")
            continue

        valid_configurations += 1
        print(f"bpp: {bpp} ({valid_configurations})")

        conf_name = hashlib.sha1(str(conf).encode("UTF-8")).hexdigest()[:8]
        yaml.safe_dump(conf, open(f"configurations/.tuning/{conf_name}.yaml", "w"))

def estimated_uncompressed_bpp(conf): 
    device = torch.device("cuda")
    model = NIF(input_features=2, **conf["model"], device=device)
    model_size = calculate_model_size(model, verbose=False)
    pixels_count = 178 * 218
    bpp = (model_size * 8) / pixels_count
    return bpp

if __name__ == "__main__":
    main()