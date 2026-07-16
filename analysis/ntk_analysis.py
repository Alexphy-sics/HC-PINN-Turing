"""#6 NTK eigenvalue analysis for FF-PINN models at different sigma.
Computes gradient norm statistics as proxy for NTK conditioning.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.pinn import FFPINN
from physics.schnakenberg import get_parameter_sets

os.makedirs("results/ntk_analysis", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

p = get_parameter_sets()["mid_k"]
a, b = p["a"], p["b"]
d_u, d_v = p["d_u"], p["d_v"]
gamma, L = p["gamma"], p["L"]

sigmas = [5.0, 12.5, 25.0]
N_ntk = 200

def compute_pde_residual_1d(model, x, t):
    """Compute Schnakenberg PDE residual manually with autograd support."""
    u, v = model(x, t)
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    v_t = torch.autograd.grad(v, t, torch.ones_like(v), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u), create_graph=True)[0]
    v_x = torch.autograd.grad(v, x, torch.ones_like(v), create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x, torch.ones_like(v), create_graph=True)[0]
    R_u = u_t - d_u * u_xx - gamma * (a - u + u**2 * v)
    R_v = v_t - d_v * v_xx - gamma * (b - u**2 * v)
    return R_u, R_v

results = {}

for sigma in sigmas:
    print(f"\n{'='*50}")
    print(f"NTK analysis: FF-PINN sigma={sigma}")

    model = FFPINN(L=L, n_fourier=64, sigma=sigma).to(device)
    ckpt_path = f"results/multi_seed/FFPINN_s{str(sigma).replace('.0','')}_s0.pt"
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"  Loaded trained weights from {ckpt_path}")
    else:
        print(f"  WARNING: No trained weights at {ckpt_path}, using random init")

    model.train()

    # Subsample collocation points at t=T_final for gradient analysis
    rng = np.random.RandomState(2026)
    x_ntk = L * rng.rand(N_ntk, 1).astype(np.float32)
    t_ntk = 5.0 * np.ones((N_ntk, 1), dtype=np.float32)

    grad_norms = []
    N_sample = 100

    for i in range(N_sample):
        xi = torch.tensor(x_ntk[i:i+1], device=device, requires_grad=True)
        ti = torch.tensor(t_ntk[i:i+1], device=device, requires_grad=True)

        R_u, R_v = compute_pde_residual_1d(model, xi, ti)
        loss_i = torch.mean(R_u**2) + torch.mean(R_v**2)

        grads = torch.autograd.grad(loss_i, model.parameters(), retain_graph=False)
        gn = sum(g.norm().item()**2 for g in grads if g is not None)**0.5
        grad_norms.append(gn)

    gn_arr = np.array(grad_norms)

    # Diagnostic: PDE residual on a grid
    model.eval()
    with torch.no_grad():
        x_grid = torch.linspace(0, L, 200, device=device).unsqueeze(1)
        t_grid = 5.0 * torch.ones(200, 1, device=device)
        u_pred, v_pred = model(x_grid, t_grid)
        # No grad needed — just a forward pass diagnostic
        ux = (u_pred[2:] - u_pred[:-2]) / (2 * (L / 199))
        uxx = (u_pred[2:] + u_pred[:-2] - 2 * u_pred[1:-1]) / ((L / 199)**2)
        vx = (v_pred[2:] - v_pred[:-2]) / (2 * (L / 199))
        vxx = (v_pred[2:] + v_pred[:-2] - 2 * v_pred[1:-1]) / ((L / 199)**2)
        u_mid = u_pred[1:-1]; v_mid = v_pred[1:-1]
        # The PDE residual at steady-state (u_t=v_t=0):
        R_u_diag = -d_u * uxx - gamma * (a - u_mid + u_mid**2 * v_mid)
        R_v_diag = -d_v * vxx - gamma * (b - u_mid**2 * v_mid)
        pde_val = (R_u_diag**2 + R_v_diag**2).mean().item()

    results[sigma] = {
        "grad_norm_mean": gn_arr.mean(), "grad_norm_std": gn_arr.std(),
        "grad_norm_max": gn_arr.max(), "grad_norm_min": gn_arr.min(),
        "condition_estimate": gn_arr.max() / max(gn_arr.min(), 1e-12),
        "pde_residual": pde_val
    }

    print(f"  PDE residual:     {pde_val:.4e}")
    print(f"  Grad norm mean:   {gn_arr.mean():.4f} ± {gn_arr.std():.4f}")
    print(f"  Condition est:    {gn_arr.max()/max(gn_arr.min(),1e-12):.2e}")

# ---- Plot ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

sigma_list = list(results.keys())
ax1.bar(range(len(sigma_list)), [results[s]["pde_residual"] for s in sigma_list],
       color=["C0", "C1", "C2"], tick_label=[f"σ={s}" for s in sigma_list])
ax1.set_yscale("log"); ax1.set_ylabel("PDE residual")
ax1.set_title("PDE residual at t=T_final")

ax2.bar(range(len(sigma_list)), [results[s]["condition_estimate"] for s in sigma_list],
       color=["C0", "C1", "C2"], tick_label=[f"σ={s}" for s in sigma_list])
ax2.set_yscale("log"); ax2.set_ylabel("Condition number estimate")
ax2.set_title("NTK condition estimate (max/min grad norm)")
ax2.axhline(1e3, color="gray", ls="--", alpha=0.5, label="ill-conditioned")
ax2.legend()

plt.suptitle("FF-PINN NTK Analysis: mid_k (γ=900)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("results/ntk_analysis/ntk_condition.png", dpi=200)
plt.savefig("results/ntk_analysis/ntk_condition.pdf")
plt.close()

np.savez("results/ntk_analysis/ntk_results.npz", results=results)
print("\nDone. Results in results/ntk_analysis/")
