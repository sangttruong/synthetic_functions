import os

import torch
from torchvision.io import read_image


class NightLight(torch.nn.Module):
    """Semi-synthetic NightLight function."""

    def __init__(self, dim=2, noise_std=0.0):
        self.bounds = torch.tensor([[-1, 1]] * dim).T
        self.dim = 2
        self.noise_std = noise_std
        self.dtype = torch.float64
        self.device = torch.device("cpu")

        # Read image "images/nightlight.jpg" into 2D tensor
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.f = read_image(os.path.join(dir_path, "images/nightlight.jpg"))
        self.f = self.f[0].float() / 255.0
        self.width = self.f.shape[1]
        self.height = self.f.shape[0]

    def __call__(self, x):
        assert x.shape[-1] == self.dim

        # Normalize x to [0, 1]
        x = (x - self.bounds[0, :]) / (self.bounds[1, :] - self.bounds[0, :])
        x = x * torch.tensor(
            [self.width, self.height], dtype=self.dtype, device=self.device
        )
        int_x = torch.round(x).long()
        y = self.f[int_x[..., 0], int_x[..., 1]]
        y += self.noise_std * torch.randn_like(y)
        return y

    def to(self, dtype, device):
        self.dtype = dtype
        self.device = device
        self.bounds = self.bounds.to(dtype=dtype, device=device)
        self.f = self.f.to(dtype=dtype, device=device)
        return self
