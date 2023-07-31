import copy
import torch

import yaml
import torch
import sys

from torch.utils.tensorboard import SummaryWriter
from input_encoding import generate_grid
from models.nif import NIF
from phases.fitting import fit_with_config

from utils import dump_model_stats, load_configuration, load_device
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
from PIL import Image


def main():
    torch.random.manual_seed(1337)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)

    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)

    config_path = sys.argv[1]
    file_path = sys.argv[2]
    model_dump_path = sys.argv[3]

    fit(config_path, file_path, model_dump_path)

def fit(config_path, file_path, model_dump_path=None):
    print("Loading configuration...")
    device = load_device()

    config = load_configuration(config_path)

    writer = SummaryWriter(log_dir = f"runs/{config_path}_{file_path}_fitting")

    writer.add_text("config", "```\n" + str(config).replace('\n', '\n\n') + "\n```")

    print("Loading images...")
    image = Image.open(file_path)

    (height, width) = (image.size[1], image.size[0])

    transform = Compose([
        Resize((height, width)),
        ToTensor(),
        Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
    ])

    image_tensor = transform(image)
    image_tensor = image_tensor.to(device)

    print("Generating grid...")
    grid = generate_grid(config, width, height, device)
    
    print("Loading model...")
    params = config["model"]
    params["input_features"] = grid.size(-1)
    model = NIF(**params, device=device, writer=writer).to(device)
    model.initialize_weights()

    print(model)
    dump_model_stats(model, width, height, writer)

    final_psnr = fit_with_config(config["fitting"], model, grid, image_tensor, 
                                             verbose=True, writer=writer)
    print(f"Final PSNR: {final_psnr}")

    final_state_dict = copy.deepcopy(model.state_dict())
    final_state_dict["__meta"] = {
        "width": width,
        "height": height
    }

    if model_dump_path:
        print("Model weights dump...")
        model.eval()
        torch.save(final_state_dict, model_dump_path)

    return final_state_dict

if __name__ == "__main__":
    main()
