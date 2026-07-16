# models/pinn.py - PINN, HC-PINN, and FF-PINN architectures

import torch
import torch.nn as nn
import numpy as np


class MLP(nn.Module):
    def __init__(self, input_dim=2, output_dim=2, hidden_layers=6, hidden_dim=64):
        super().__init__()
        layers = []
        prev = input_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(prev, hidden_dim))
            layers.append(nn.Tanh())
            prev = hidden_dim
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class PINN(nn.Module):
    def __init__(self, L=1.0, hidden_layers=6, hidden_dim=64):
        super().__init__()
        self.L = L
        self.net = MLP(input_dim=2, output_dim=2,
                       hidden_layers=hidden_layers, hidden_dim=hidden_dim)
        self.register_buffer("x_scale", torch.tensor(2.0 / L))
        self.register_buffer("x_offset", torch.tensor(-1.0))

    def forward(self, x, t):
        xn = self.x_scale * x + self.x_offset
        inp = torch.cat([xn, t], dim=1)
        out = self.net(inp)
        return out[:, 0:1], out[:, 1:2]


class HCPINN(nn.Module):
    def __init__(self, L=1.0, hidden_layers=6, hidden_dim=64):
        super().__init__()
        self.L = L
        self.net = MLP(input_dim=2, output_dim=2,
                       hidden_layers=hidden_layers, hidden_dim=hidden_dim)

    def forward(self, x, t):
        xi = torch.cos(np.pi * x / self.L)
        inp = torch.cat([xi, t], dim=1)
        out = self.net(inp)
        return out[:, 0:1], out[:, 1:2]


class FFPINN(nn.Module):
    def __init__(self, L=1.0, hidden_layers=6, hidden_dim=64,
                 n_fourier=64, sigma=10.0):
        super().__init__()
        self.L = L
        self.n_fourier = n_fourier
        input_dim = 2 * n_fourier
        self.net = MLP(input_dim=input_dim, output_dim=2,
                       hidden_layers=hidden_layers, hidden_dim=hidden_dim)
        self.register_buffer("B", torch.randn(2, n_fourier) * sigma)

    def forward(self, x, t):
        v = torch.cat([x, t], dim=1)
        proj = v @ self.B
        ff = torch.cat([torch.sin(proj), torch.cos(proj)], dim=1)
        out = self.net(ff)
        return out[:, 0:1], out[:, 1:2]
