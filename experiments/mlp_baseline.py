"""#22 Data-driven MLP baseline.
Trains an MLP (same architecture, no physics) to regress FDM reference solution.
Distinguishes spectral bias from PDE-constraint-induced smoothing.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time

from models.pinn import MLP
from physics.schnakenberg import get_parameter_sets

os.makedirs("results/mlp_baseline", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

p = get_parameter_sets()["mid_k"]
L = p["L"]

# Load FDM data (final time slice)
fdm = np.load("results/fdm/fdm_mid_k.npz")
u_fdm = fdm["u_history"][-1]
x_fdm = fdm["x"]
t_fdm = fdm["t"]

# Training data: uniformly sample x from [0,1], target = FDM solution
N_train = 8000
rng = np.random.RandomState(2026)
x_data = L * rng.rand(N_train, 1).astype(np.float32)
u_target = np.interp(x_data.flatten(), x_fdm, u_fdm).astype(np.float32).reshape(-1, 1)

x_t = torch.tensor(x_data, device=device)
u_t = torch.tensor(u_target, device=device)

# MLP: 1D input, 2D output (u, v) — trains on u only
class DataMLP(nn.Module):
    def __init__(self, hidden_layers=6, hidden_dim=64):
        super().__init__()
        layers = []
        prev = 1
        for _ in range(hidden_layers):
            layers.append(nn.Linear(prev, hidden_dim))
            layers.append(nn.Tanh())
            prev = hidden_dim
        layers.append(nn.Linear(prev, 2))
        self.net = nn.Sequential(*layers)
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

model = DataMLP().to(device)
opt = torch.optim.Adam(model.parameters(), lr=5e-4)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=12000, eta_min=1e-5)

print("Training data-driven MLP...")
history = {"epoch": [], "loss": []}
t0 = time.time()

for epoch in range(1, 12001):
    out = model(x_t)
    loss = torch.mean((out[:, 0:1] - u_t)**2)

    opt.zero_grad()
    loss.backward()
    opt.step()
    sch.step()

    if epoch % 2000 == 0:
        e = time.time() - t0
        history["epoch"].append(epoch)
        history["loss"].append(loss.item())
        print(f"  [{epoch:5d}] MSE={loss.item():.4e} [{e:.0f}s]")

# Evaluate on uniform grid
Nx = 256
x_eval = torch.linspace(0, L, Nx, device=device).unsqueeze(1)
with torch.no_grad():
    u_pred = model(x_eval)[:, 0].cpu().numpy()

# FFT
window = np.hanning(Nx)
u_w = (u_pred - u_pred.mean()) * window
u_fft = np.abs(np.fft.fft(u_w))
kv = 2 * np.pi * np.fft.fftfreq(Nx, L/Nx)
pos = kv > 0
k_dom = kv[pos][np.argmax(u_fft[pos])]

print(f"\n  Data MLP k_dom:    {k_dom:.3f}")
print(f"  FDM k_dom:         12.52")
print(f"  PINN k_dom:        6.28")

# Plot comparison
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5))

# FDM truth vs MLP prediction
ax1.plot(x_eval.cpu(), u_pred, "C0-", lw=2, label="Data MLP")
ax1.plot(x_fdm, u_fdm, "k--", lw=1.5, alpha=0.6, label="FDM")
ax1.set_xlabel("x"); ax1.set_ylabel("u")
ax1.set_title("Data-driven MLP vs FDM"); ax1.legend(); ax1.grid(True)

# FFT
ax2.semilogy(kv[pos], u_fft[pos], "C0-", lw=1.5)
ax2.axvline(k_dom, color="C3", ls="--", label=f"MLP k={k_dom:.2f}")
ax2.axvline(12.52, color="C2", ls=":", label="FDM k=12.52")
ax2.set_xlim(0, 60); ax2.set_xlabel("k"); ax2.set_ylabel("|FFT|")
ax2.set_title("FFT: data-driven MLP"); ax2.legend(fontsize=9); ax2.grid(True)

# Loss curve
ax3.semilogy(history["epoch"], history["loss"], "C0-", lw=1.5)
ax3.set_xlabel("Epoch"); ax3.set_ylabel("MSE")
ax3.set_title("Training loss"); ax3.grid(True)

plt.suptitle("Data-Driven MLP Baseline (mid_k, γ=900)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("results/mlp_baseline/mlp_baseline.png", dpi=200)
plt.savefig("results/mlp_baseline/mlp_baseline.pdf")
plt.close()

np.savez("results/mlp_baseline/mlp_results.npz", k_dom=k_dom, history=history)
print(f"\nDone. Results in results/mlp_baseline/")
