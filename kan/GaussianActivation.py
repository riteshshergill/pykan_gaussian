import torch
import torch.nn as nn


class GaussianActivation(nn.Module):
    def __init__(self, mu=0.0, sigma=1.0, amplitude=1.0):
        super(GaussianActivation, self).__init__()
        self.mu = nn.Parameter(torch.tensor(mu, dtype=torch.float32))
        self.sigma = nn.Parameter(torch.tensor(sigma, dtype=torch.float32))
        self.amplitude = nn.Parameter(torch.tensor(amplitude, dtype=torch.float32))

    def forward(self, x):
        return self.amplitude * torch.exp(-0.5 * ((x - self.mu) / self.sigma) ** 2)