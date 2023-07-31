import kornia
import torch
import torchvision

class SobelMSELoss:
    def __init__(self, device=None, writer=None):
        self.writer = writer
        self.__iterations = 0

    def clear_cache(self):
        self.original_grad_cache = None

    def __grad(self, image):
        return kornia.filters.spatial_gradient(image.unsqueeze(0))

    def dump_image(self, tag, image):
        self.writer.add_images(tag, image.squeeze().transpose(0, 1).transpose(-1, -2))

    def __call__(self, reconstructed, original):
        reconstructed_grad = self.__grad(reconstructed)
        if self.original_grad_cache is None:
            self.original_grad_cache = self.__grad(original)

        if self.writer:
            self.__iterations += 1
            if self.__iterations == 100:
                self.dump_image("reconstructed_grad", reconstructed_grad)
                self.dump_image("original_grad", self.original_grad_cache)
                self.writer.flush()
                self.__iterations = 0

        return torch.mean((reconstructed_grad - self.original_grad_cache) ** 2)
