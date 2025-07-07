import math
import torch
from utils import linear_reduction

def restart_weights(model, amount, range):
    """
    Perform the weights restart (perturbation) of the NIF model as described in Section 3.1.5.

    Args:
        model (torch.nn.Module): The model whose weights are to be restarted.
        amount (float): The amount of weight restarting.
        range (float): The range of weight restarting.
    """
    for (name, module) in model.named_modules():
        if "head" in name:
            continue

        if hasattr(module, "weight"):
            current_weight = module.weight.clone()
            mean = current_weight.abs().mean()
            variation = mean * range
            torch.nn.init.uniform_(module.weight, -variation, variation)
            torch.nn.functional.dropout(module.weight, 1.0 - amount, inplace=True)
            module.weight.add_(current_weight)

def perform_restart_step(model, restart_config, progress, verbose=False):
    """
    Performs a restart step for the model based on the restart configuration and progress.

    Args:
        model (torch.nn.Module): The model to be restarted.
        restart_config (dict): Configuration dictionary for restarting.
        progress (float): The progress of training.
        verbose (bool, optional): If True, prints restart details. Defaults to False.
    """
    amount_vars = restart_config["amount"]
    range_vars = restart_config["range"]
    restart_amount = linear_reduction(amount_vars["start"], amount_vars["end"], math.pow(progress, amount_vars["smoothing"]))
    restart_range = linear_reduction(range_vars["start"], range_vars["end"], math.pow(progress, range_vars["smoothing"]))

    if verbose:
        print(f"Restarting weights, amount: {restart_amount}, range: {restart_range}")

    with torch.no_grad():
        restart_weights(model, restart_amount, restart_range)