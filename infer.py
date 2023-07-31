import torchvision
from phases.training import patched_forward
import utils
import yaml
import torch
from skimage import io
import sys
from input_encoding import generate_grid

from models.nif import NIF
from utils import load_configuration, load_device, pad_for_patching

def load_flattened_state(model, state_dict):
    for key in list(state_dict.keys()):
        if key not in model.state_dict():
            del state_dict[key]
            continue

        state_dict[key] = state_dict[key].reshape(model.state_dict()[key].shape)

    model.load_state_dict(state_dict, strict=True)

def main():
    torch.random.manual_seed(1337)

    config_path = sys.argv[1]
    state_dict_path = sys.argv[2]
    reconstructed_image_path = sys.argv[3]

    state_dict = torch.load(state_dict_path)
    rescaled_reconstructed_image = infer(config_path, state_dict)
    io.imsave(reconstructed_image_path, rescaled_reconstructed_image)

def infer(config_path, state_dict):
    device = load_device()

    print("Loading configuration...")
    config = load_configuration(config_path)

    metadata = state_dict["__meta"]
    width = metadata["width"]
    height = metadata["height"]

    print("Generating grid...")
    grid = generate_grid(config, width, height, device)

    print("Loading model...")

    params = config["model"]
    params["input_features"] = grid.size(-1)
    model = NIF(**params, device=device).to(device)
    load_flattened_state(model, state_dict)
    model.eval()

    with torch.no_grad():
        patching = config["fitting"]["tuning"]["patching"]
        batched_grid = grid.permute(2, 0, 1)
        batched_grid = pad_for_patching(batched_grid, patching)
        batched_grid = batched_grid.unsqueeze(1)
        batched_grid = torch.pixel_unshuffle(batched_grid, patching)
        batched_grid = batched_grid.permute(1, 2, 3, 0)

        grid_patches = batched_grid.unbind(0)

        uncropped_reconstructed_image = patched_forward(model, grid_patches, patching)
        width_padding = (uncropped_reconstructed_image.size(-1) - width) // 2
        height_padding = (uncropped_reconstructed_image.size(-2) - height) // 2
        reconstructed_image = torchvision.transforms.functional.crop(
            uncropped_reconstructed_image,
            height_padding, width_padding,
            height, width 
        )

    rescaled_reconstructed_image = utils.rescale_img((reconstructed_image + 1) / 2, mode='clamp') \
        .detach().mul(255.0).to(torch.uint8).permute(1, 2, 0).cpu().numpy()

    return rescaled_reconstructed_image

if __name__ == "__main__":
    main()
