"""
Diagnostic training for PINN vs HC-PINN on Schnakenberg.
Fixed collocation points, lower LR, longer training.

Run: python train.py
"""

import torch
import torch.nn as nn
import numpy as np
import os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.pinn import PINN, HCPINN
from physics.schnakenberg import get_parameter_sets, steady_state


# ============================================================
# Fixed collocation points (pre-sample once)
# ============================================================

def generate_fixed_points(L, T, N_int, N_ic, N_bc, device="cpu"):
    """Generate fixed collocation points, sampled once."""
    rng = np.random.RandomState(2026)

    x_int = L * rng.rand(N_int, 1).astype(np.float32)
    t_int = T * rng.rand(N_int, 1).astype(np.float32)

    x_ic = L * rng.rand(N_ic, 1).astype(np.float32)
    t_ic = np.zeros((N_ic, 1), dtype=np.float32)

    # BC: x=0 and x=L
    t_bc = T * rng.rand(N_bc, 1).astype(np.float32)
    x_bc0 = np.zeros((N_bc // 2, 1), dtype=np.float32)
    x_bcL = L * np.ones((N_bc // 2, 1), dtype=np.float32)
    x_bc = np.vstack([x_bc0, x_bcL])

    return {
        "x_int": torch.tensor(x_int, device=device, requires_grad=True),
        "t_int": torch.tensor(t_int, device=device, requires_grad=True),
        "x_ic": torch.tensor(x_ic, device=device, requires_grad=True),
        "t_ic": torch.tensor(t_ic, device=device, requires_grad=True),
        "x_bc": torch.tensor(x_bc, device=device, requires_grad=True),
        "t_bc": torch.tensor(t_bc, device=device, requires_grad=True),
    }


# ============================================================
# PDE residual
# ============================================================

def pde_residual(model, x, t, a, b, d_u, d_v, gamma):
    u, v = model(x, t)
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    v_t = torch.autograd.grad(v, t, torch.ones_like(v), create_graph=True, retain_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    v_x = torch.autograd.grad(v, x, torch.ones_like(v), create_graph=True, retain_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x, torch.ones_like(v), create_graph=True, retain_graph=True)[0]

    R_u = u_t - d_u * u_xx - gamma * (a - u + u**2 * v)
    R_v = v_t - d_v * v_xx - gamma * (b - u**2 * v)
    return R_u, R_v


# ============================================================
# Training (Adam → LBFGS)
# ============================================================

def train_model(model, model_type, params, fixed_pts, device="cpu",
                adam_epochs=12000, lr=5e-4,
                bc_weight=10.0, ic_weight=10.0,
                report_every=500, label=""):
    """Train with Adam + cosine annealing."""

    a, b = params["a"], params["b"]
    d_u, d_v = params["d_u"], params["d_v"]
    gamma, L = params["gamma"], params["L"]

    history = {"epoch": [], "loss_pde": [], "loss_ic": [], "loss_bc": [], "loss_total": []}
    t0 = time.time()

    # ---- Phase 1: Adam ----
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=adam_epochs, eta_min=1e-5)

    x_int, t_int = fixed_pts["x_int"], fixed_pts["t_int"]
    x_ic, t_ic = fixed_pts["x_ic"], fixed_pts["t_ic"]
    x_bc, t_bc = fixed_pts["x_bc"], fixed_pts["t_bc"]

    # IC targets
    ic_path = f"results/fdm/fdm_{label}.npz"
    if os.path.exists(ic_path):
        fdm = np.load(ic_path)
        u_ic_full = fdm["u_history"][0]
        x_fdm_grid = np.linspace(0, L, params["Nx"])
    else:
        u_s, _ = steady_state(a, b)
        u_ic_full = np.full(params["Nx"], u_s)
        x_fdm_grid = np.linspace(0, L, params["Nx"])
    v_ic_const = b / (a + b)**2

    u_ic_target = torch.tensor(
        np.interp(x_ic.cpu().detach().numpy().flatten(), x_fdm_grid, u_ic_full),
        dtype=torch.float32, device=device
    ).unsqueeze(1)

    print(f"  Adam: {adam_epochs} epochs, lr={lr}")

    for epoch in range(adam_epochs):
        # PDE
        R_u, R_v = pde_residual(model, x_int, t_int, a, b, d_u, d_v, gamma)
        loss_pde = torch.mean(R_u**2) + torch.mean(R_v**2)

        # IC
        u_pred_ic, v_pred_ic = model(x_ic, t_ic)
        loss_ic = torch.mean((u_pred_ic - u_ic_target)**2) + torch.mean((v_pred_ic - v_ic_const)**2)

        # BC
        if model_type == "pinn":
            u_pred_bc, v_pred_bc = model(x_bc, t_bc)
            u_x_bc = torch.autograd.grad(u_pred_bc, x_bc, torch.ones_like(u_pred_bc),
                                         create_graph=True, retain_graph=True)[0]
            v_x_bc = torch.autograd.grad(v_pred_bc, x_bc, torch.ones_like(v_pred_bc),
                                         create_graph=True, retain_graph=True)[0]
            loss_bc = torch.mean(u_x_bc**2) + torch.mean(v_x_bc**2)
        else:
            loss_bc = torch.tensor(0.0, device=device)

        loss = loss_pde + ic_weight * loss_ic + bc_weight * loss_bc

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if epoch % report_every == 0 and epoch > 0:
            history["epoch"].append(epoch)
            history["loss_pde"].append(loss_pde.item())
            history["loss_ic"].append(loss_ic.item())
            history["loss_bc"].append(loss_bc.item() if model_type == "pinn" else 0.0)
            history["loss_total"].append(loss.item())
            elapsed = time.time() - t0
            print(f"  [{label}] Adam {epoch:5d} | PDE={loss_pde.item():.2e} "
                  f"IC={loss_ic.item():.2e} BC={loss_bc.item():.2e} "
                  f"Total={loss.item():.2e} [{elapsed:.0f}s]")

    # Final log
    elapsed = time.time() - t0
    print(f"  [{label}] Done in {elapsed:.0f}s | Final PDE={loss_pde.item():.2e}")
    history["epoch"].append(adam_epochs)
    history["loss_pde"].append(loss_pde.item())
    history["loss_ic"].append(loss_ic.item())
    history["loss_total"].append(loss.item())

    return history


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate_model(model, params, device="cpu", Nx=256, T_eval=5.0):
    a, b = params["a"], params["b"]
    L = params["L"]
    x = torch.linspace(0, L, Nx, device=device).unsqueeze(1)
    Nt = 51
    t_eval = torch.linspace(0, T_eval, Nt, device=device).unsqueeze(1)

    u_grid = torch.zeros(Nt, Nx, device=device)
    v_grid = torch.zeros(Nt, Nx, device=device)

    for i, ti in enumerate(t_eval):
        t_batch = ti.expand(Nx, 1)
        u, v = model(x, t_batch)
        u_grid[i] = u.squeeze()
        v_grid[i] = v.squeeze()

    u_final = u_grid[-1].cpu().numpy()
    window = np.hanning(Nx)
    u_windowed = (u_final - u_final.mean()) * window
    fft_amp = np.abs(np.fft.fft(u_windowed))
    k_vals = 2 * np.pi * np.fft.fftfreq(Nx, L / Nx)
    pos = k_vals > 0
    k_dom = k_vals[pos][np.argmax(fft_amp[pos])]

    return {
        "x": x.cpu().numpy().flatten(),
        "t": t_eval.cpu().numpy().flatten(),
        "u_grid": u_grid.cpu().numpy(),
        "v_grid": v_grid.cpu().numpy(),
        "u_final": u_final,
        "k_fft": k_vals[pos],
        "fft_amplitude": fft_amp[pos],
        "k_dominant": k_dom
    }


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    param_sets = get_parameter_sets()
    labels_to_run = ["low_k", "mid_k"]

    for label in labels_to_run:
        p = param_sets[label]
        print(f"{'='*60}")
        print(f"{label}  gamma={p['gamma']}")
        print(f"{'='*60}")

        T_train = 5.0
        L = p["L"]

        # Generate fixed collocation points
        fixed_pts = generate_fixed_points(L, T_train,
                                          N_int=8000, N_ic=500, N_bc=400, device=device)

        # ---- PINN ----
        print("\n  [PINN (soft BC)]")
        pinn = PINN(L=L).to(device)
        hist_pinn = train_model(pinn, "pinn", p, fixed_pts, device=device,
                                adam_epochs=12000, lr=5e-4,
                                bc_weight=10.0, ic_weight=10.0,
                                report_every=500, label=label)

        # ---- HC-PINN ----
        print("\n  [HC-PINN (hard BC)]")
        hc = HCPINN(L=L).to(device)
        hist_hc = train_model(hc, "hcpinn", p, fixed_pts, device=device,
                              adam_epochs=12000, lr=5e-4,
                              bc_weight=0.0, ic_weight=10.0,
                              report_every=500, label=label)

        # ---- Evaluate ----
        eval_pinn = evaluate_model(pinn, p, device=device, T_eval=T_train)
        eval_hc = evaluate_model(hc, p, device=device, T_eval=T_train)

        fdm_data = np.load(f"results/fdm/fdm_{label}.npz")
        k_fdm = fdm_data["k_dominant"]

        # ---- Plot comparison ----
        fig, axes = plt.subplots(3, 3, figsize=(16, 14))

        for col, (name, eval_data, hist) in enumerate([
            ("PINN (soft BC)", eval_pinn, hist_pinn),
            ("HC-PINN (hard BC)", eval_hc, hist_hc),
            ("FDM (ref)", None, None)
        ]):
            if col < 2:
                im = axes[0, col].pcolormesh(eval_data["x"], eval_data["t"],
                                              eval_data["u_grid"], shading="auto", cmap="RdBu_r")
                axes[0, col].set_title(f"{name}: u(x,t)")
                axes[0, col].set_xlabel("x"); axes[0, col].set_ylabel("t")
                plt.colorbar(im, ax=axes[0, col])

                axes[1, col].plot(eval_data["x"], eval_data["u_final"], "b-", lw=1.5)
                axes[1, col].set_title(f"{name}: final u(x)")
                axes[1, col].set_xlabel("x"); axes[1, col].set_ylabel("u")
                axes[1, col].grid(True, alpha=0.3)

                axes[2, col].semilogy(eval_data["k_fft"], eval_data["fft_amplitude"], "b-", lw=1.5)
                axes[2, col].axvline(eval_data["k_dominant"], color="C3", ls="--", lw=1.5,
                                     label=f"k_dom={eval_data['k_dominant']:.2f}")
                axes[2, col].axvline(k_fdm, color="C2", ls=":", lw=1.5,
                                     label=f"FDM k={k_fdm:.2f}")
                axes[2, col].set_title(f"{name}: FFT")
                axes[2, col].set_xlabel("k"); axes[2, col].set_ylabel("|FFT|")
                axes[2, col].legend(fontsize=8); axes[2, col].set_xlim(0, 60)
                axes[2, col].grid(True, alpha=0.3)
            else:
                u_fdm = fdm_data["u_history"]
                t_fdm = fdm_data["t"]
                x_fdm = fdm_data["x"]
                t_idx = t_fdm <= T_train
                im = axes[0, col].pcolormesh(x_fdm, t_fdm[t_idx], u_fdm[t_idx],
                                              shading="auto", cmap="RdBu_r")
                axes[0, col].set_title("FDM: u(x,t)")
                axes[0, col].set_xlabel("x"); axes[0, col].set_ylabel("t")
                plt.colorbar(im, ax=axes[0, col])

                axes[1, col].plot(x_fdm, u_fdm[-1], "b-", lw=1.5)
                axes[1, col].set_title("FDM: final u(x)")
                axes[1, col].set_xlabel("x"); axes[1, col].set_ylabel("u")
                axes[1, col].grid(True, alpha=0.3)

                axes[2, col].semilogy(fdm_data["k_fft"], fdm_data["fft_amplitude"], "b-", lw=1.5)
                axes[2, col].axvline(k_fdm, color="C2", ls="--", lw=1.5,
                                     label=f"k_dom={k_fdm:.2f}")
                axes[2, col].set_title("FDM: FFT")
                axes[2, col].set_xlabel("k"); axes[2, col].set_ylabel("|FFT|")
                axes[2, col].legend(fontsize=8); axes[2, col].set_xlim(0, 60)
                axes[2, col].grid(True, alpha=0.3)

        plt.suptitle(f"Diagnostic: {label} (γ={p['gamma']})  —  8000 Adam + 500 LBFGS",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"results/{label}_diagnostic.png", dpi=200, bbox_inches="tight")
        plt.close()

        # ---- Loss curves ----
        fig, ax = plt.subplots(figsize=(8, 4))
        for name, hist, color in [("PINN", hist_pinn, "C0"), ("HC-PINN", hist_hc, "C1")]:
            ax.semilogy(hist["epoch"], hist["loss_total"], color=color, lw=1.5, label=f"{name} total")
            ax.semilogy(hist["epoch"], hist["loss_pde"], color=color, ls="--", lw=1, alpha=0.7,
                        label=f"{name} PDE")
        ax.set_title(f"Training Loss: {label}")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"results/{label}_loss.png", dpi=150, bbox_inches="tight")
        plt.close()

        # Print
        print(f"\n  === {label} Summary ===")
        print(f"  FDM k_dom:      {k_fdm:.2f}")
        print(f"  PINN k_dom:     {eval_pinn['k_dominant']:.2f}  Δk={abs(eval_pinn['k_dominant']-float(k_fdm)):.3f}")
        print(f"  HC-PINN k_dom:  {eval_hc['k_dominant']:.2f}  Δk={abs(eval_hc['k_dominant']-float(k_fdm)):.3f}")

    print(f"\n{'='*60}")
    print("Diagnostic complete.")
    print("Check results/diagnostic_{low_k,mid_k}.png for visual comparison.")
    print(f"{'='*60}")
