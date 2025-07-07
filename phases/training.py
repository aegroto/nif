import math
import copy
import time
from losses import build_loss_fn


import torch

from losses.psnr import psnr
from phases.scheduler import CustomScheduler

from utils import cast_for_dump, clearlines, printdots

class TrainingContext:
    def __init__(self):
        self.iteration = 0
        self.optimizer = None
        self.scheduler = None

def eval_psnr(reconstructed_image, image):
    """
    Evaluates the Peak Signal-to-Noise Ratio (PSNR) between the reconstructed image and the original image.

    Args:
        reconstructed_image (torch.Tensor): The reconstructed image.
        image (torch.Tensor): The original image.

    Returns:
        float: The PSNR value.
    """
    return psnr(
        reconstructed_image.div(2.0).add(0.5), 
        image.div(2.0).add(0.5), 
    1.0)

def patched_forward(model, grid_patches, patching):
    """
    Performs a forward pass through the model that takes into account patching (Section 3.1.7).

    Args:
        model (torch.nn.Module): The model to be used for reconstruction.
        grid_patches (list): List of grid patches.
        patching (int): The patching factor.

    Returns:
        torch.Tensor: The reconstructed image.
    """
    reconstructed_patches = list()
    for grid_patch in grid_patches:
        reconstructed_image_patch = model(grid_patch)
        reconstructed_patches.append(reconstructed_image_patch.unsqueeze(0))

    reconstructed_image = torch.cat(reconstructed_patches, 0).permute(1, 0, 2, 3)
    reconstructed_image = torch.pixel_shuffle(reconstructed_image, patching).squeeze()
    return reconstructed_image

def train_model(context, model, config, verbose=True, writer=None, overwrite_state=False):
    loss_fn = build_loss_fn(config["loss"])

    if not context.optimizer:
        context.optimizer = torch.optim.AdamW(model.parameters(), **config["optimizer"])
    optimizer = context.optimizer

    if not context.scheduler:
        context.scheduler = CustomScheduler(optimizer, **config["scheduler"])
    scheduler = context.scheduler

    iterations = config["iterations"]
    patching = config["patching"]

    grid_patches = context.grid_patches
    image_patches = context.image_patches
    padded_image = context.padded_image

    best_psnr = 0.0
    best_reconstructed = None
    best_state_dict = None

    model.train()

    if verbose:
        printdots(5)

    last_log_time = time.time()
    iteration = 0
    while iteration < iterations:
        iteration += 1
        context.iteration += 1

        reconstructed_patches = list()
        for (grid_patch, image_patch) in zip(grid_patches, image_patches):
            optimizer.zero_grad()

            reconstructed_image_patch = model(grid_patch)
            loss = loss_fn(reconstructed_image_patch, image_patch)

            loss.backward()
            optimizer.step()

            reconstructed_patches.append(reconstructed_image_patch.unsqueeze(0))

        scheduler.step()

        reconstructed_image = torch.cat(reconstructed_patches, 0).permute(1, 0, 2, 3)
        reconstructed_image = torch.pixel_shuffle(reconstructed_image, patching).squeeze()
        psnr = eval_psnr(reconstructed_image, padded_image)

        if psnr > best_psnr:
            best_psnr = psnr
            best_state_dict = copy.deepcopy(model.state_dict())
            best_reconstructed = reconstructed_image

        if context.iteration % config["log_interval"] == 0:
            clearlines(5)
            log_time = time.time()
            past_time = log_time - last_log_time
            last_log_time = log_time
            iterations_per_second = config["log_interval"] / past_time
            estimated_time_remaining = (iterations - iteration) / (iterations_per_second * 60)
            estimated_minutes_remaining = math.floor(estimated_time_remaining)
            estimated_seconds_remaining = math.floor((estimated_time_remaining % 1) * 60)

            with torch.no_grad():
                learning_rate = scheduler._last_lr

                if verbose:
                    print(f"# Iteration {iteration}/{iterations}, Learning rate: {learning_rate[0]}")
                    print(f"# Iterations/s: {iterations_per_second:.2f}, estimated time remaining: {estimated_minutes_remaining}m{estimated_seconds_remaining}s")
                    print(f"# Loss: {loss}")
                    print(f"# PSNR: {psnr}")
                    print(f"# Best PSNR: {best_psnr}")

                if writer is not None:
                    writer.add_scalar("learning_rate", learning_rate[0], context.iteration)
                    writer.add_scalar("loss", loss, context.iteration)
                    writer.add_scalar("loss", loss, context.iteration)
                    writer.add_scalar("PSNR", psnr, context.iteration)
                    writer.add_scalar("PSNR_best", best_psnr, context.iteration)
                    writer.flush()

        if writer is not None:
            if context.iteration % config["image_dump_interval"] == 0 and best_reconstructed is not None:
                writer.add_images("reconstructed_plane", cast_for_dump(best_reconstructed), context.iteration, dataformats="CHW")
                writer.flush()

    if overwrite_state:
        with torch.no_grad():
            model.load_state_dict(best_state_dict)

    return best_psnr
