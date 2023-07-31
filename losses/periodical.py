import math
import torch

HALF_PI = torch.acos(torch.zeros(1)).item()

class PeriodicLoss:
    def __init__(self, first_loss_fn, second_loss_fn, period):
        self.first_loss_fn = first_loss_fn
        self.second_loss_fn = second_loss_fn

        self.T = period / 3
        self.iteration = 0

    def __call__(self, original, distorted):
        alpha = (1.0 + math.sin(HALF_PI * (self.iteration / self.T))) / 2.0
        self.iteration += 1

        first_loss = self.first_loss_fn(original, distorted)
        second_loss = self.second_loss_fn(original, distorted)

        total_loss = alpha * first_loss + (1.0 - alpha) * second_loss
        return total_loss, {
            type(self.first_loss_fn).__name__: first_loss, 
            type(self.second_loss_fn).__name__: second_loss
        }



