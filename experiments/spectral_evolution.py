"""#5 Spectral evolution diagnostic.
Tracks spatial FFT spectra of PINN prediction at log-spaced epoch intervals
during mid_k (γ=900) training to visualize spectral bias dynamics.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time

from models.pinn import PINN
from physics.schnakenberg import get_parameter_sets
from train import generate_fixed_points, pde_residual

os.makedirs("results/spectral_evolution", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

p = get_parameter_sets()["mid_k"]
a, b = p["a"], p["b"]
d_u, d_v = p["d_u"], p["d_v"]
gamma, L = p["gamma"], p["L"]
Nx_fft = 256

T_train = 5.0
pts = generate_fixed_points(L, T_train, 8000, 500, 400, device)
x_int, t_int = pts["x_int"], pts["t_int"]
x_ic, t_ic = pts["x_ic"], pts["t_ic"]
x_bc, t_bc = pts["x_bc"], pts["t_bc"]

# IC targets
fdm = np.load("results/fdm/fdm_mid_k.npz")
u_ic_full = fdm["u_history"][0]
x_fdm_grid = np.linspace(0, L, p["Nx"])
v_ic_const = b / (a + b)**2
u_ic_target = torch.tensor(
    np.interp(x_ic.cpu().detach().numpy().flatten(), x_fdm_grid, u_ic_full),
    dtype=torch.float32, device=device).unsqueeze(1)

# x grid for FFT evaluation
x_eval = torch.linspace(0, L, Nx_fft, device=device).unsqueeze(1)

model = PINN(L=L).to(device)
opt = torch.optim.Adam(model.parameters(), lr=5e-4)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=12000, eta_min=1e-5)

# Log-spaced epoch checkpoints (plus start)
checkpoints = [0, 100, 300, 1000, 2000, 3000, 5000, 8000, 12000]
spectral_history = []

def record_spectrum(model, epoch):
    """Evaluate FFT of prediction at t=T_train."""
    with torch.no_grad():
        t_batch = torch.full((Nx_fft, 1), T_train, device=device)
        u_pred, _ = model(x_eval, t_batch)
        u = u_pred.cpu().numpy().flatten()
    window = np.hanning(Nx_fft)
    u_w = (u - u.mean()) * window
    amp = np.abs(np.fft.fft(u_w))
    k = 2 * np.pi * np.fft.fftfreq(Nx_fft, L/Nx_fft)
    pos = k > 0
    return epoch, k[pos], amp[pos]

# Record initial
ep, kv, av = record_spectrum(model, 0)
spectral_history.append((ep, kv, av))

print("Training with spectral logging...")
t0 = time.time()

for epoch in range(1, 12001):
    # PDE
    R_u, R_v = pde_residual(model, x_int, t_int, a, b, d_u, d_v, gamma)
    lpde = torch.mean(R_u**2) + torch.mean(R_v**2)

    # IC
    up_ic, vp_ic = model(x_ic, t_ic)
    lic = torch.mean((up_ic - u_ic_target)**2) + torch.mean((vp_ic - v_ic_const)**2)

    # BC
    ub, vb = model(x_bc, t_bc)
    uxb = torch.autograd.grad(ub, x_bc, torch.ones_like(ub), create_graph=True, retain_graph=True)[0]
    vxb = torch.autograd.grad(vb, x_bc, torch.ones_like(vb), create_graph=True, retain_graph=True)[0]
    lbc = torch.mean(uxb**2) + torch.mean(vxb**2)

    loss = lpde + 10.0 * lic + 10.0 * lbc
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sch.step()

    if epoch in checkpoints:
        ep, kv, av = record_spectrum(model, epoch)
        spectral_history.append((ep, kv, av))
        elapsed = time.time() - t0
        print(f"  Epoch {epoch:5d}: PDE={lpde.item():.2e}, IC={lic.item():.2e}, [{elapsed:.0f}s]")

    if epoch % 2000 == 0:
        elapsed = time.time() - t0
        print(f"  [{epoch}/12000] PDE={lpde.item():.2e} [{elapsed:.0f}s]")

# ---- Save spectral data ----
np.savez("results/spectral_evolution/spectral_history.npz",
         checkpoints=np.array(checkpoints),
         k_vals=spectral_history[0][1],
         amplitudes=np.array([av for _, _, av in spectral_history]))

# ---- Plot: heatmap + slice grid ----
fig = plt.figure(figsize=(14, 12))
gs = fig.add_gridspec(3, 3, height_ratios=[1.2, 0.8, 0.8])

# Heatmap: epoch vs k, color = log10(|FFT|)
ax_heat = fig.add_subplot(gs[0, :])
all_k = spectral_history[0][1]
k_max_plot = 40
k_mask = all_k <= k_max_plot
k_plot = all_k[k_mask]

spec_matrix = np.zeros((len(spectral_history), k_mask.sum()))
for i, (ep, kv, av) in enumerate(spectral_history):
    spec_matrix[i] = av[k_mask]

im = ax_heat.pcolormesh(k_plot, [ep for ep, _, _ in spectral_history],
                         np.log10(np.maximum(spec_matrix, 1e-8)),
                         shading="auto", cmap="inferno")
ax_heat.axvline(12.57, color="cyan", ls="--", lw=2, alpha=0.8, label=r"k* (theory)")
ax_heat.axvline(6.28, color="lime", ls=":", lw=2, alpha=0.8, label=r"k≈6.28")
ax_heat.set_xlabel("k"); ax_heat.set_ylabel("Epoch")
ax_heat.set_title("Spectral Evolution: log₁₀|FFT(û)|")
ax_heat.legend(fontsize=8)
plt.colorbar(im, ax=ax_heat, label="log₁₀|FFT|")

# Slice plots at selected epochs
slice_epochs = [0, 1000, 3000, 5000, 8000, 12000]
slice_map = {ep: (kv, av) for ep, kv, av in spectral_history}
ax_slices = [fig.add_subplot(gs[1, i]) for i in range(3)]

for idx, (ep_slice, ax) in enumerate(zip(slice_epochs[:3], ax_slices)):
    if ep_slice in slice_map:
        kv, av = slice_map[ep_slice]
        mask = kv <= k_max_plot
        ax.semilogy(kv[mask], av[mask], lw=1)
        ax.axvline(12.57, color="cyan", ls="--", lw=1, alpha=0.6)
        ax.axvline(6.28, color="lime", ls=":", lw=1, alpha=0.6)
        ax.set_title(f"Epoch {ep_slice}"); ax.set_xlabel("k")
        if idx == 0: ax.set_ylabel("|FFT|")

ax_slices2 = [fig.add_subplot(gs[2, i]) for i in range(3)]
for idx, (ep_slice, ax) in enumerate(zip(slice_epochs[3:], ax_slices2)):
    if ep_slice in slice_map:
        kv, av = slice_map[ep_slice]
        mask = kv <= k_max_plot
        ax.semilogy(kv[mask], av[mask], lw=1)
        ax.axvline(12.57, color="cyan", ls="--", lw=1, alpha=0.6)
        ax.axvline(6.28, color="lime", ls=":", lw=1, alpha=0.6)
        ax.set_title(f"Epoch {ep_slice}"); ax.set_xlabel("k")

plt.suptitle("Spectral Evolution: PINN on mid_k (γ=900)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("results/spectral_evolution/spectral_evolution.png", dpi=200)
plt.savefig("results/spectral_evolution/spectral_evolution.pdf")
plt.close()

# Final report
final_ep, final_kv, final_av = spectral_history[-1]
k_dom = final_kv[np.argmax(final_av)]
print(f"\nTraining complete. Final k_dom = {k_dom:.3f}")
print(f"Expected target: 12.57, observed collapse: {k_dom:.3f}")
print(f"Results in results/spectral_evolution/")
