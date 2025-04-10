import torch
import torch.nn as nn
import math

class FourierFeatureLayer(nn.Module):
    def __init__(self, input_dim, mapping_size, scale=10.0):
        """
        input_dim: dimension of the original input (e.g., 1 for time t)
        mapping_size: number of Fourier features
        scale: std. deviation of the random matrix B ~ N(0, scale^2)
        """
        super(FourierFeatureLayer, self).__init__()
        self.B = nn.Parameter(
            torch.randn(mapping_size, input_dim) * scale,
            requires_grad=False  # fixed random projection
        )

    def forward(self, x):
        # x: shape (batch_size, input_dim)
        x_proj = 2 * math.pi * x @ self.B.T  # shape: (batch_size, mapping_size)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
