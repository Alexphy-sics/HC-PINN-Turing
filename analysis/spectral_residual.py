"""#13 Spectral decomposition of PDE residual.
For trained PINN on mid_k, decomposes PDE residual into frequency bands
to prove that residual power concentrates at high k.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from models.pinn import PINN
from physics.schnakenberg import get_parameter_sets
from train import pde_residual

os.makedirs("results/spectral_residual", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

p = get_parameter_sets()["mid_k"]
a, b = p["a"], p["b"]
d_u, d_v = p["d_u"], p["d_v"]
gamma, L = p["gamma"], p["L"]

Nx = 256
x_pts = torch.linspace(0, L, Nx, device=device).unsqueeze(1)
t_eval = 5.0

# Load a trained PINN model (use multi_seed output if available, else init new)
model = PINN(L=L).to(device)
ckpt_path = "results/multi_seed/PINN_900_s0.pt"
if os.path.exists(ckpt_path):
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"Loaded PINN from {ckpt_path}")
else:
    print("WARNING: No trained model found, using random init")

# Compute prediction and PDE residual at final time on uniform grid
with torch.no_grad():
    t_batch = torch.full((Nx, 1), t_eval, device=device)
    u_pred, v_pred = model(x_pts, t_batch)
    u = u_pred.cpu().numpy().flatten()

# Compute PDE residual with grad
x_pts.requires_grad_(True)
t_batch = torch.full((Nx, 1), t_eval, device=device, requires_grad=True)
R_u, R_v = pde_residual(model, x_pts, t_batch, a, b, d_u, d_v, gamma)
R_u_np = R_u.detach().cpu().numpy().flatten()
R_v_np = R_v.detach().cpu().numpy().flatten()
R_total = R_u_np**2 + R_v_np**2

# FFT of solution and residual
window = np.hanning(Nx)
k_vals = 2 * np.pi * np.fft.fftfreq(Nx, L/Nx)
pos = k_vals > 0
kv = k_vals[pos]

# Solution spectrum
u_fft = np.abs(np.fft.fft((u - u.mean()) * window))[pos]

# Residual spectrum
R_fft = np.abs(np.fft.fft((R_total - R_total.mean()) * window))[pos]

# Compute energy fractions
k_star = 12.57
low_mask = kv < k_star
high_mask = kv >= k_star
low_energy = np.trapezoid(R_fft[low_mask], kv[low_mask])
high_energy = np.trapezoid(R_fft[high_mask], kv[high_mask])
total_energy = low_energy + high_energy
low_frac = low_energy / total_energy * 100
high_frac = high_energy / total_energy * 100

print(f"\n=== Spectral Residual Decomposition ===")
print(f"  Solution k_dom:      {kv[np.argmax(u_fft)]:.2f}")
print(f"  Low-k (<{k_star:.1f}):   {low_frac:.1f}% of residual power")
print(f"  High-k (≥{k_star:.1f}):  {high_frac:.1f}% of residual power")
print(f"  Peak residual k:     {kv[np.argmax(R_fft)]:.2f}")

# ---- Plot ----
fig, axes = plt.subplots(3, 1, figsize=(12, 11))

# Top: spatial profile
axes[0].plot(x_pts.detach().cpu().numpy(), u, "b-", lw=1.5)
axes[0].set_xlabel("x"); axes[0].set_ylabel("u(x)")
axes[0].set_title("PINN solution at t=T_final (mid_k, γ=900)")
axes[0].grid(True, alpha=0.3)

# Middle: solution FFT
k_max = 60
k_mask = kv <= k_max
axes[1].semilogy(kv[k_mask], u_fft[k_mask], "b-", lw=1.5, label="|FFT[û]|")
axes[1].axvline(kv[np.argmax(u_fft)], color="C3", ls="--", lw=1.5,
                label=f"k_dom={kv[np.argmax(u_fft)]:.2f}")
axes[1].axvline(12.57, color="C2", ls=":", lw=1.5, label="k*_theory=12.57")
axes[1].set_xlabel("k"); axes[1].set_ylabel("|FFT|")
axes[1].set_title("Solution spectrum")
axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

# Bottom: residual FFT
axes[2].semilogy(kv[k_mask], R_fft[k_mask], "C3-", lw=1.5, label="|FFT[PDE residual]|")
axes[2].axvline(k_star, color="k", ls="--", lw=1.5, alpha=0.5, label=f"k*={k_star}")
ymin, ymax = axes[2].get_ylim()
axes[2].fill_between([0, k_star], ymin, ymax, alpha=0.1, color="blue", label=f"low-k: {low_frac:.0f}%")
axes[2].fill_between([k_star, k_max], ymin, ymax, alpha=0.1, color="red", label=f"high-k: {high_frac:.0f}%")
axes[2].set_xlabel("k"); axes[2].set_ylabel("|FFT[R]|")
axes[2].set_title("PDE residual spectrum")
axes[2].legend(fontsize=8); axes[2].grid(True, alpha=0.3)

plt.suptitle("Spectral Residual Decomposition: PINN mid_k (γ=900)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("results/spectral_residual/spectral_residual.png", dpi=200)
plt.savefig("results/spectral_residual/spectral_residual.pdf")
plt.close()

np.savez("results/spectral_residual/residual_data.npz",
         k_vals=kv, u_fft=u_fft, R_fft=R_fft,
         low_frac=low_frac, high_frac=high_frac)

print(f"\nDone. Results in results/spectral_residual/")
