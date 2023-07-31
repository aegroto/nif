import copy
import torch

from phases.training import TrainingContext, train_model
from phases.reset import perform_restart_step
from utils import pad_for_patching

def batch_grid(grid, patching):
    batched_grid = grid.permute(2, 0, 1)
    batched_grid = pad_for_patching(batched_grid, patching)
    batched_grid = batched_grid.unsqueeze(1)
    batched_grid = torch.pixel_unshuffle(batched_grid, patching)
    batched_grid = batched_grid.permute(1, 2, 3, 0)
    return batched_grid

def batch_image(image, patching):
    batched_image = image.unsqueeze(1)
    batched_image = pad_for_patching(batched_image, patching)
    batched_image = torch.pixel_unshuffle(batched_image, patching)
    batched_image = batched_image.permute(1, 0, 2, 3)
    return batched_image

def fit_with_config(config, model, grid, image, verbose = False, writer = None):
    context = TrainingContext()

    patching = config["tuning"]["patching"]

    context.grid_patches = batch_grid(grid, patching).unbind(0)
    context.image_patches = batch_image(image, patching).unbind(0)
    context.padded_image = pad_for_patching(image, patching)

    restart_config = config["restart"]

    total_steps = config["steps"]

    best_psnr = 0.0
    best_state_dict = None

    for step in range(1, total_steps+1):
        last_step = step==total_steps

        if verbose:
            print(f"Step #{step}/{total_steps}")

        step_psnr = train_model(context, model, config["tuning"],
                        verbose = verbose,
                        writer = writer,
                        overwrite_state = True)
        
        if step_psnr > best_psnr:
            best_state_dict = copy.deepcopy(model.state_dict())
            best_psnr = step_psnr

        if restart_config and not last_step:
            perform_restart_step(model, restart_config, step / total_steps, verbose)

    model.load_state_dict(best_state_dict)

    return float(best_psnr)

