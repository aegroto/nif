import math
import torch
from modules.siren.utils import build_linear_initializer, initialize_first_layer
from utils import linear_reduction

def restart_weights(model, amount, range):
    # initializer = build_linear_initializer(model.c, model.omega)
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
    amount_vars = restart_config["amount"]
    range_vars = restart_config["range"]
    restart_amount = linear_reduction(amount_vars["start"], amount_vars["end"], math.pow(progress, amount_vars["smoothing"]))
    restart_range = linear_reduction(range_vars["start"], range_vars["end"], math.pow(progress, range_vars["smoothing"]))

    if verbose:
        print(f"Restarting weights, amount: {restart_amount}, range: {restart_range}")

    with torch.no_grad():
        restart_weights(model, restart_amount, restart_range)
