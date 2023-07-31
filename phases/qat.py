import re
import math
import copy
import time

import torch
from compression import dequantize_tensor, quantize_tensor
from losses import build_loss_fn
from losses.composite import CompositeLoss
from losses.log_cosh import LogCoshLoss

from losses.psnr import PSNREvaluationMetric
from phases.training import eval_psnr, patched_forward
from phases.scheduler import CustomScheduler

from utils import cast_for_dump, clearlines, pad_for_patching, printdots

def generate_quantization_config(model, config):
    quantization_config = dict()
    with torch.no_grad():
        for (name, _) in model.named_parameters():
            tensor_config = config["default"]

            if "groups" in config:
                for group_config in config["groups"]:
                    if re.search(group_config["regex"], name):
                        tensor_config = group_config

            tensor_config["bound"] = 1.0
            quantization_config[name] = tensor_config

    return quantization_config

def simulate_quantization(model, quantization_config):
    with torch.no_grad():
        updated_quantization_config = copy.deepcopy(quantization_config)

        full_precision_dict = copy.deepcopy(model.state_dict())
        quantized_dict = copy.deepcopy(model.state_dict())
        for (name, param) in quantized_dict.items():
            if name.endswith("_mask"):
                continue

            tensor_config = copy.deepcopy(updated_quantization_config[name])
            bits = tensor_config["bits"]

            bound = torch.max(param.max().abs(), param.min().abs())
            quantized_param = quantize_tensor(param, bits, bound)
            dequantized_param = dequantize_tensor(quantized_param, bits, bound)
            quantized_dict[name] = dequantized_param

            tensor_config["bound"] = bound.clone().item()
            updated_quantization_config[name] = tensor_config

        model.load_state_dict(quantized_dict)

        return full_precision_dict, updated_quantization_config

def restore_state(model, full_precision_dict):
    with torch.no_grad():
        model.load_state_dict(full_precision_dict)

def quantization_aware_train_model(context, model, config, writer, optimizer_state_dict=None):
    loss_fn = build_loss_fn(config["tuning"]["loss"])

    if not context.optimizer:
        context.optimizer = torch.optim.AdamW(model.parameters(), **config["tuning"]["optimizer"])
    optimizer = context.optimizer

    if optimizer_state_dict:
        optimizer.load_state_dict(optimizer_state_dict)

    if not context.scheduler:
        context.scheduler = CustomScheduler(optimizer, **config["tuning"]["scheduler"])
    scheduler = context.scheduler

    iterations = config["tuning"]["iterations"]
    patching = config["tuning"]["patching"]

    grid_patches = context.grid_patches
    image_patches = context.image_patches
    padded_image = context.padded_image

    quantization_config = generate_quantization_config(model, config)

    best_psnr = 0.0
    best_state_dict = None
    best_full_precision_state_dict = None
    best_quantization_config = None
    
    printdots(5)

    last_log_time = time.time()
    iteration = 0
    while iteration < iterations:
        iteration += 1
        context.iteration += 1

        optimizer.zero_grad()

        reconstructed_patches = list()
        for (grid_patch, image_patch) in zip(grid_patches, image_patches):
            optimizer.zero_grad()

            full_precision_dict, updated_quantization_config = simulate_quantization(model, quantization_config)

            reconstructed_image_patch = model(grid_patch)
            loss = loss_fn(reconstructed_image_patch, image_patch)

            loss.backward()

            quantization_config = updated_quantization_config

            restore_state(model, full_precision_dict)

            optimizer.step()

            reconstructed_patches.append(reconstructed_image_patch.unsqueeze(0))

        reconstructed_image = torch.cat(reconstructed_patches, 0).permute(1, 0, 2, 3)
        reconstructed_image = torch.pixel_shuffle(reconstructed_image, patching).squeeze()
        psnr = eval_psnr(reconstructed_image, padded_image)

        if psnr > best_psnr:
            best_psnr = psnr
            best_full_precision_state_dict = copy.deepcopy(full_precision_dict)
            best_state_dict = copy.deepcopy(model.state_dict())
            best_quantization_config = copy.deepcopy(quantization_config)
            best_reconstructed = reconstructed_image

        scheduler.step()

        if context.iteration % config["tuning"]["log_interval"] == 0:
            clearlines(5)
            log_time = time.time()
            past_time = log_time - last_log_time
            last_log_time = log_time
            iterations_per_second = config["tuning"]["log_interval"] / past_time
            estimated_time_remaining = (iterations - iteration) / (iterations_per_second * 60)
            estimated_minutes_remaining = math.floor(estimated_time_remaining)
            estimated_seconds_remaining = math.floor((estimated_time_remaining % 1) * 60)

            with torch.no_grad():
                writer.add_scalar("loss", loss, context.iteration)

                learning_rate = scheduler._last_lr
                writer.add_scalar("learning_rate", learning_rate[0], context.iteration)

                print(f"# Iteration {iteration}/{iterations}, Learning rate: {learning_rate[0]}")
                print(f"# Iterations/s: {iterations_per_second:.2f}, estimated time remaining: {estimated_minutes_remaining}m{estimated_seconds_remaining}s")
                print(f"# Loss: {loss}")
                print(f"# PSNR: {psnr}")
                print(f"# Best PSNR: {best_psnr}")

                writer.add_scalar("loss", loss, context.iteration)
                writer.add_scalar("PSNR", psnr, context.iteration)
                writer.add_scalar("PSNR_best", best_psnr, context.iteration)
                writer.flush()

        if context.iteration % config["tuning"]["image_dump_interval"] == 0 and best_reconstructed is not None:
            writer.add_images("reconstructed_plane", cast_for_dump(best_reconstructed), context.iteration, dataformats="CHW")

    with torch.no_grad():
        model.load_state_dict(best_state_dict)
        model.eval()

        return best_quantization_config, best_full_precision_state_dict, best_psnr
