import copy
import torch

import yaml
import torch
import sys

from PIL import Image

from torch.utils.tensorboard import SummaryWriter
from compression import dequantize_tensor, quantize_tensor
from input_encoding import generate_grid
from losses.psnr import psnr
from models.nif import NIF
from phases.fitting import batch_grid, batch_image
from phases.training import TrainingContext, eval_psnr
from phases.qat import quantization_aware_train_model
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
from phases.reset import perform_restart_step

from utils import load_configuration, load_device, pad_for_patching

def main():
    torch.random.manual_seed(1337)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)

    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)

    config_path = sys.argv[1]
    file_path = sys.argv[2]
    unquantized_model_path = sys.argv[3]
    quantized_model_dump_path = sys.argv[4]
    full_precision_quantized_model_dump_path = sys.argv[5]

    unquantized_state_dict = torch.load(unquantized_model_path)

    quantize(config_path, file_path, unquantized_state_dict, quantized_model_dump_path, full_precision_quantized_model_dump_path)

def quantize(config_path, file_path, 
             unquantized_state_dict, 
             quantized_model_dump_path=None, 
             full_precision_quantized_model_dump_path=None):
    device = load_device()

    print("Loading configuration...")
    config = load_configuration(config_path)

    metadata = copy.deepcopy(unquantized_state_dict["__meta"])
    del unquantized_state_dict["__meta"]

    writer = SummaryWriter(log_dir = f"runs/{config_path}_{file_path}_quantization")

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
    model.load_state_dict(unquantized_state_dict)

    context = TrainingContext()

    patching = config["quantization"]["tuning"]["patching"]

    context.grid_patches = batch_grid(grid, patching).unbind(0)
    context.image_patches = batch_image(image_tensor, patching).unbind(0)
    context.padded_image = pad_for_patching(image_tensor, patching)

    total_steps = config["quantization"]["steps"]
    restart_config = config["quantization"]["restart"]

    best_psnr = 0.0
    best_full_precision_state_dict = None
    best_quantization_config = None

    for step in range(1, total_steps+1):
        print(f"Step #{step}/{total_steps}")
        last_step = step==total_steps

        quantization_config, state_dict, psnr = \
            quantization_aware_train_model(context, model, config["quantization"], writer)

        if psnr > best_psnr:
            best_psnr = psnr
            best_full_precision_state_dict = copy.deepcopy(state_dict)
            best_quantization_config = copy.deepcopy(quantization_config)
        else:
            model.load_state_dict(best_full_precision_state_dict)

        if restart_config and not last_step:
            perform_restart_step(model, restart_config, step / total_steps, True)

    print(f"Final PSNR: {best_psnr}")

    print("Loading best state...")

    model.load_state_dict(best_full_precision_state_dict)

    best_full_precision_state_dict = copy.deepcopy(model.state_dict())

    print("Model weights dump...")

    if quantized_model_dump_path:
        with torch.no_grad():
            quantized_dict = copy.deepcopy(model.state_dict())
            for (name, param) in quantized_dict.items():
                tensor_config = best_quantization_config[name]

                bits = tensor_config["bits"]
                bound = tensor_config["bound"]

                quantized_param = quantize_tensor(param, bits, bound)
                dequantized_param = dequantize_tensor(quantized_param, bits, bound)
                quantized_dict[name] = dequantized_param

            model.load_state_dict(quantized_dict)
            quantized_dict["quantization_config"] = best_quantization_config
            torch.save(quantized_dict, quantized_model_dump_path)

    best_full_precision_state_dict["quantization_config"] = best_quantization_config
    if full_precision_quantized_model_dump_path:
        torch.save(best_full_precision_state_dict, full_precision_quantized_model_dump_path)

    best_full_precision_state_dict["__meta"] = metadata

    return best_full_precision_state_dict

if __name__ == "__main__":
    main()
