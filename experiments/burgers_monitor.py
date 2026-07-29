"""Burgers equation PINN + Spectral Monitor verification.

Implements the cross-validation experiment described in §4.2 of the paper:
the spectral monitor is applied to a standard PINN solving the 1D Burgers
equation (nu=0.01/pi, 6x64 MLP, 20000 Adam epochs).  Because Burgers has
no Turing-type pattern selection failure, the monitor should register zero
false positives across all checkpoints.

Results from the paper: 0/40 out-of-band alarms (0 false positives).

Usage:
    python experiments/burgers_monitor.py           # full training (20000 epochs)
    python experiments/burgers_monitor.py --quick    # 2000 epochs for smoke test
"""

import sys, os, time, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np

# Import the spectral monitor
from experiments.spectral_monitor import SpectralMonitor

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("BurgersMonitor")

# ============================================================
# Burgers PINN
# ============================================================

class BurgersPINN(nn.Module):
    """Standard PINN for 1D Burgers equation with 6x64 tanh MLP.

    Architecture identical to the Schnakenberg PINN (models/pinn.py)
    except the output dimension is 1 (scalar u).
    """

    def __init__(self, hidden_layers=6, hidden_dim=64, x_min=-1.0, x_max=1.0):
        super().__init__()
        self.x_min = x_min
        self.x_max = x_max
        layers = []
        prev = 2  # (x, t)
        for _ in range(hidden_layers):
            layers.append(nn.Linear(prev, hidden_dim))
            layers.append(nn.Tanh())
            prev = hidden_dim
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, t):
        inp = torch.cat([x, t], dim=1)
        return self.net(inp)


def burgers_pde_residual(model, x, t, nu):
    """Compute Burgers PDE residual: u_t + u*u_x - nu*u_xx."""
    u = model(x, t)
    u_t = torch.autograd.grad(
        u, t, torch.ones_like(u), create_graph=True, retain_graph=True
    )[0]
    u_x = torch.autograd.grad(
        u, x, torch.ones_like(u), create_graph=True, retain_graph=True
    )[0]
    u_xx = torch.autograd.grad(
        u_x, x, torch.ones_like(u_x), create_graph=True, retain_graph=True
    )[0]
    R = u_t + u * u_x - nu * u_xx
    return R


# ============================================================
# Training
# ============================================================

def train_burgers(
    nu=0.01 / np.pi,
    x_min=-1.0,
    x_max=1.0,
    T=1.0,
    N_int=10000,
    N_ic=500,
    N_bc=400,
    hidden_layers=6,
    hidden_dim=64,
    epochs=20000,
    lr=1e-3,
    ic_weight=10.0,
    bc_weight=10.0,
    monitor_interval=500,
    report_every=1000,
    seed=42,
    device="cpu",
):
    """Train a Burgers PINN with spectral monitoring.

    Args:
        nu: Viscosity parameter (default 0.01/pi).
        epochs: Total Adam epochs.
        monitor_interval: SpectralMonitor check interval.

    Returns:
        (model, monitor, history_dict)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.RandomState(seed)

    model = BurgersPINN(
        hidden_layers=hidden_layers,
        hidden_dim=hidden_dim,
        x_min=x_min,
        x_max=x_max,
    ).to(device)

    # ---- Collocation points ----
    x_int = torch.tensor(
        (x_max - x_min) * rng.rand(N_int, 1).astype(np.float32) + x_min,
        device=device, requires_grad=True,
    )
    t_int = torch.tensor(
        T * rng.rand(N_int, 1).astype(np.float32),
        device=device, requires_grad=True,
    )

    # IC: u(x, 0) = -sin(pi * x)
    x_ic = torch.tensor(
        (x_max - x_min) * rng.rand(N_ic, 1).astype(np.float32) + x_min,
        device=device, requires_grad=True,
    )
    t_ic = torch.zeros(N_ic, 1, device=device)
    u_ic_target = -torch.sin(np.pi * x_ic)

    # BC: u(-1, t) = u(1, t) = 0 (Dirichlet from -sin)
    t_bc_tensor = torch.tensor(
        T * rng.rand(N_bc // 2, 1).astype(np.float32),
        device=device,
    )
    x_bc0 = torch.full((N_bc // 2, 1), x_min, device=device)
    x_bcL = torch.full((N_bc // 2, 1), x_max, device=device)
    x_bc = torch.cat([x_bc0, x_bcL], dim=0)
    t_bc = torch.cat([t_bc_tensor, t_bc_tensor], dim=0)
    u_bc_target = torch.zeros(N_bc, 1, device=device)

    # ---- Monitor setup ----
    # Burgers with u(x,0) = -sin(pi*x) has fundamental mode k = pi.
    # The shock develops high-frequency content but the fundamental
    # remains dominant.  Admissible band: pi +/- 1.0 (generous).
    k_target = np.pi
    Nx_monitor = 256
    x_grid = np.linspace(x_min, x_max, Nx_monitor)
    t_eval = T

    monitor = SpectralMonitor(
        k_target=k_target,
        delta=2.0,  # wide enough to accommodate shock harmonics
        interval=monitor_interval,
        L=x_max - x_min,
        Nx=Nx_monitor,
        t_eval=t_eval,
    )

    # ---- Adam training ----
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    history = {
        "epoch": [],
        "loss_pde": [],
        "loss_ic": [],
        "loss_bc": [],
        "loss_total": [],
    }
    t_start = time.time()

    logger.info(f"Burgers PINN training: {epochs} Adam epochs, nu={nu:.4f}")
    logger.info(f"  Monitor interval={monitor_interval}, k_target={k_target:.3f} +/- {monitor.delta:.1f}")
    logger.info(f"  Device: {device}")

    for epoch in range(1, epochs + 1):
        # PDE residual
        R = burgers_pde_residual(model, x_int, t_int, nu)
        loss_pde = torch.mean(R ** 2)

        # IC
        u_pred_ic = model(x_ic, t_ic)
        loss_ic = torch.mean((u_pred_ic - u_ic_target) ** 2)

        # BC
        u_pred_bc = model(x_bc, t_bc)
        u_x_bc = torch.autograd.grad(
            u_pred_bc,
            x_bc,
            torch.ones_like(u_pred_bc),
            create_graph=True,
            retain_graph=True,
        )[0]
        loss_bc = torch.mean(u_pred_bc ** 2) + torch.mean(u_x_bc ** 2)

        loss = loss_pde + ic_weight * loss_ic + bc_weight * loss_bc

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Spectral monitor check
        monitor.report(epoch, model, x_grid, t_eval)

        if epoch % report_every == 0:
            history["epoch"].append(epoch)
            history["loss_pde"].append(loss_pde.item())
            history["loss_ic"].append(loss_ic.item())
            history["loss_bc"].append(loss_bc.item())
            history["loss_total"].append(loss.item())
            elapsed = time.time() - t_start
            logger.info(
                f"  Epoch {epoch:5d}/{epochs} | PDE={loss_pde.item():.2e} "
                f"IC={loss_ic.item():.2e} BC={loss_bc.item():.2e} "
                f"Total={loss.item():.2e} [{elapsed:.0f}s]"
            )

    elapsed = time.time() - t_start
    logger.info(f"  Done in {elapsed:.0f}s")
    logger.info(
        f"  Final PDE={loss_pde.item():.2e}, IC={loss_ic.item():.2e}, "
        f"BC={loss_bc.item():.2e}"
    )

    return model, monitor, history


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate_burgers(
    model, x_min=-1.0, x_max=1.0, T=1.0, Nx=256, Nt=101, device="cpu"
):
    """Evaluate the trained Burgers PINN on a uniform grid."""
    x = torch.linspace(x_min, x_max, Nx, device=device).unsqueeze(1)
    t = torch.linspace(0, T, Nt, device=device).unsqueeze(1)

    u_grid = torch.zeros(Nt, Nx, device=device)
    for i, ti in enumerate(t):
        u_grid[i] = model(x, ti.expand(Nx, 1)).squeeze()

    u_final = u_grid[-1].cpu().numpy()
    window = np.hanning(Nx)
    u_w = (u_final - u_final.mean()) * window
    fft_amp = np.abs(np.fft.fft(u_w))
    k_vals = 2 * np.pi * np.fft.fftfreq(Nx, (x_max - x_min) / Nx)
    pos = k_vals > 0
    k_dom = k_vals[pos][np.argmax(fft_amp[pos])]

    return {
        "x": x.cpu().numpy().flatten(),
        "t": t.cpu().numpy().flatten(),
        "u_grid": u_grid.cpu().numpy(),
        "u_final": u_final,
        "k_fft": k_vals[pos],
        "fft_amplitude": fft_amp[pos],
        "k_dominant": k_dom,
    }


# ============================================================
# Main
# ============================================================

def main(quick=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs("results/burgers_monitor", exist_ok=True)

    config = dict(
        nu=0.01 / np.pi,
        x_min=-1.0,
        x_max=1.0,
        T=1.0,
        N_int=10000,
        N_ic=500,
        N_bc=400,
        hidden_layers=6,
        hidden_dim=64,
        epochs=2000 if quick else 20000,
        lr=1e-3,
        ic_weight=10.0,
        bc_weight=10.0,
        monitor_interval=50 if quick else 500,
        report_every=500 if quick else 2000,
        seed=42,
        device=device,
    )

    print("=" * 60)
    print("Burgers PINN + Spectral Monitor")
    print(f"  Mode: {'QUICK (2k epochs)' if quick else 'FULL (20k epochs)'}")
    print("=" * 60)

    model, monitor, history = train_burgers(**config)

    # ---- Results ----
    summary = monitor.summary()
    print("\n" + "=" * 60)
    print("SPECTRAL MONITOR RESULTS")
    print("=" * 60)
    print(f"  Total checks:       {summary['n_checks']}")
    print(f"  Total alarms:       {summary['n_alarms']}")
    print(f"  Alarm rate:         {summary['alarm_rate']*100:.1f}%")
    print(f"  Last k_dom:         {summary['last_k_dom']:.3f}")
    print(f"  Target band:        [{summary['k_target']-summary['delta']:.3f}, "
          f"{summary['k_target']+summary['delta']:.3f}]")
    print(f"                  k* = {summary['k_target']:.3f} +/- {summary['delta']:.1f}")

    if summary["n_alarms"] == 0:
        print("\n  VERIFICATION PASSED: zero false positives in Burgers training.")
        print("  (Consistent with paper: 0/40 out-of-band alarms)")
    else:
        print(f"\n  WARNING: {summary['n_alarms']} false positives detected.")

    # ---- Evaluate final solution ----
    ev = evaluate_burgers(model, device=device)

    print(f"\n  Final k_dom (FFT):  {ev['k_dominant']:.3f}")

    # ---- Save data ----
    np.savez(
        f"results/burgers_monitor/burgers_monitor_{'quick' if quick else 'full'}.npz",
        x=ev["x"],
        t=ev["t"],
        u_grid=ev["u_grid"],
        u_final=ev["u_final"],
        k_fft=ev["k_fft"],
        fft_amplitude=ev["fft_amplitude"],
        k_dominant=ev["k_dominant"],
        monitor_checks=np.array(monitor.epochs),
        monitor_k_dom=np.array(monitor.k_dom_history),
        monitor_alarms=np.array(monitor.alarm_history),
        monitor_target=summary["k_target"],
        monitor_delta=summary["delta"],
        n_checks=summary["n_checks"],
        n_alarms=summary["n_alarms"],
    )

    # ---- Plot ----
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # u(x,t)
    im = axes[0, 0].pcolormesh(ev["x"], ev["t"], ev["u_grid"], shading="auto", cmap="RdBu_r")
    axes[0, 0].set_title("Burgers PINN: u(x,t)")
    axes[0, 0].set_xlabel("x"); axes[0, 0].set_ylabel("t")
    plt.colorbar(im, ax=axes[0, 0])

    # final u(x)
    axes[0, 1].plot(ev["x"], ev["u_final"], "b-", lw=1.5)
    axes[0, 1].set_title(f"Final u(x) at t=1.0  (nu=0.01/pi)")
    axes[0, 1].set_xlabel("x"); axes[0, 1].set_ylabel("u")
    axes[0, 1].grid(True, alpha=0.3)

    # FFT
    axes[0, 2].semilogy(ev["k_fft"], ev["fft_amplitude"], "b-", lw=1.5)
    axes[0, 2].axvline(ev["k_dominant"], color="C3", ls="--", lw=1.5,
                        label=f"k_dom={ev['k_dominant']:.2f}")
    axes[0, 2].axvline(np.pi, color="C2", ls=":", lw=1.5, label="k=pi (IC)")
    axes[0, 2].set_title("FFT of final u")
    axes[0, 2].set_xlabel("k"); axes[0, 2].set_ylabel("|FFT|")
    axes[0, 2].legend(fontsize=8); axes[0, 2].set_xlim(0, 60)
    axes[0, 2].grid(True, alpha=0.3)

    # k_dom trajectory
    axes[1, 0].plot(monitor.epochs, monitor.k_dom_history, "C0-o", markersize=4, lw=1.5)
    axes[1, 0].axhline(summary["k_target"], color="C2", ls="--", lw=1.5, label=f"k*={summary['k_target']:.2f}")
    axes[1, 0].fill_between(
        monitor.epochs,
        summary["k_target"] - summary["delta"],
        summary["k_target"] + summary["delta"],
        alpha=0.1, color="C2",
    )
    axes[1, 0].set_title("Spectral Monitor: k_dom trajectory")
    axes[1, 0].set_xlabel("Epoch"); axes[1, 0].set_ylabel("k_dom")
    axes[1, 0].legend(fontsize=8); axes[1, 0].grid(True, alpha=0.3)

    # Spectrogram
    spec = np.array(monitor.full_spectra).T
    vmax = np.percentile(spec, 95)
    im2 = axes[1, 1].pcolormesh(
        monitor.epochs, monitor.k_pos, spec,
        shading="auto", cmap="inferno", vmin=0, vmax=vmax,
    )
    axes[1, 1].axhline(summary["k_target"], color="C2", ls="--", lw=1.5)
    axes[1, 1].set_title("Spectral heatmap")
    axes[1, 1].set_xlabel("Epoch"); axes[1, 1].set_ylabel("k")
    plt.colorbar(im2, ax=axes[1, 1])

    # Loss
    axes[1, 2].semilogy(history["epoch"], history["loss_total"], "C0-", lw=1.5, label="total")
    axes[1, 2].semilogy(history["epoch"], history["loss_pde"], "C0--", lw=1, alpha=0.7, label="PDE")
    axes[1, 2].set_title("Training loss")
    axes[1, 2].set_xlabel("Epoch"); axes[1, 2].set_ylabel("Loss")
    axes[1, 2].legend(fontsize=8); axes[1, 2].grid(True, alpha=0.3)

    plt.suptitle(
        f"Burgers PINN + Spectral Monitor  ({'Quick' if quick else 'Full'} run, "
        f"alarms={summary['n_alarms']}/{summary['n_checks']})",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    png_path = f"results/burgers_monitor/burgers_monitor_{'quick' if quick else 'full'}.png"
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved: {png_path}")

    # ---- Verification report ----
    assert summary["n_alarms"] == 0, (
        f"Burgers verification FAILED: {summary['n_alarms']} false positives."
    )
    print("\n  ==> Burgers verification: PASS (zero false positives)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Burgers PINN + Spectral Monitor verification"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick smoke test (2000 epochs instead of 20000)"
    )
    args = parser.parse_args()
    main(quick=args.quick)
