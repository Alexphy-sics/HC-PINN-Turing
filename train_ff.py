# train_ff.py - Fourier Features PINN experiment
# Trains FF-PINN on mid_k (gamma=900) with three sigma values.
# Compares with FDM reference.

import torch
import numpy as np
import os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.pinn import FFPINN
from physics.schnakenberg import get_parameter_sets, steady_state


def generate_fixed_points(L, T, N_int, N_ic, N_bc, device="cpu"):
    rng = np.random.RandomState(2026)
    x_int = L * rng.rand(N_int, 1).astype(np.float32)
    t_int = T * rng.rand(N_int, 1).astype(np.float32)
    x_ic = L * rng.rand(N_ic, 1).astype(np.float32)
    t_ic = np.zeros((N_ic, 1), dtype=np.float32)
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


def train_ffpinn(model, params, pts, device, epochs, lr, bc_w, ic_w, report_every, label):
    a, b = params["a"], params["b"]
    d_u, d_v = params["d_u"], params["d_v"]
    gamma, L = params["gamma"], params["L"]

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

    x_int, t_int = pts["x_int"], pts["t_int"]
    x_ic, t_ic = pts["x_ic"], pts["t_ic"]
    x_bc, t_bc = pts["x_bc"], pts["t_bc"]

    path = f"results/fdm/fdm_{label}.npz"
    if os.path.exists(path):
        fd = np.load(path)
        u_ic_full = fd["u_history"][0]
        x_fdm_grid = np.linspace(0, L, params["Nx"])
    else:
        u_s, _ = steady_state(a, b)
        u_ic_full = np.full(params["Nx"], u_s)
        x_fdm_grid = np.linspace(0, L, params["Nx"])
    v_ic_const = b / (a + b)**2
    u_ic_target = torch.tensor(
        np.interp(x_ic.cpu().detach().numpy().flatten(), x_fdm_grid, u_ic_full),
        dtype=torch.float32, device=device).unsqueeze(1)

    history = {"epoch": [], "pde": [], "ic": [], "bc": [], "total": []}
    t0 = time.time()

    for ep in range(epochs):
        R_u, R_v = pde_residual(model, x_int, t_int, a, b, d_u, d_v, gamma)
        lpde = torch.mean(R_u**2) + torch.mean(R_v**2)
        up, vp = model(x_ic, t_ic)
        lic = torch.mean((up - u_ic_target)**2) + torch.mean((vp - v_ic_const)**2)
        ub, vb = model(x_bc, t_bc)
        uxb = torch.autograd.grad(ub, x_bc, torch.ones_like(ub), create_graph=True, retain_graph=True)[0]
        vxb = torch.autograd.grad(vb, x_bc, torch.ones_like(vb), create_graph=True, retain_graph=True)[0]
        lbc = torch.mean(uxb**2) + torch.mean(vxb**2)
        loss = lpde + ic_w * lic + bc_w * lbc
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sch.step()
        if ep % report_every == 0 and ep > 0:
            e = time.time() - t0
            history["epoch"].append(ep)
            history["pde"].append(lpde.item())
            history["ic"].append(lic.item())
            history["bc"].append(lbc.item())
            history["total"].append(loss.item())
            print(f"  [{label}] {ep:5d} | PDE={lpde.item():.2e} IC={lic.item():.2e} BC={lbc.item():.2e} Total={loss.item():.2e} [{e:.0f}s]")

    e = time.time() - t0
    print(f"  [{label}] Done in {e:.0f}s | Final PDE={lpde.item():.2e}")
    return history


@torch.no_grad()
def evaluate(model, params, device="cpu", Nx=256, Te=5.0):
    L = params["L"]
    x = torch.linspace(0, L, Nx, device=device).unsqueeze(1)
    Nt = 51
    te = torch.linspace(0, Te, Nt, device=device).unsqueeze(1)
    ug = torch.zeros(Nt, Nx, device=device)
    for i, ti in enumerate(te):
        u, _ = model(x, ti.expand(Nx, 1))
        ug[i] = u.squeeze()
    uf = ug[-1].cpu().numpy()
    window = np.hanning(Nx)
    uf_windowed = (uf - uf.mean()) * window
    fa = np.abs(np.fft.fft(uf_windowed))
    kv = 2 * np.pi * np.fft.fftfreq(Nx, L / Nx)
    pos = kv > 0
    kd = kv[pos][np.argmax(fa[pos])]
    return {"x": x.cpu().numpy().flatten(), "t": te.cpu().numpy().flatten(),
            "u_grid": ug.cpu().numpy(), "u_final": uf,
            "k_fft": kv[pos], "fft_amplitude": fa[pos], "k_dominant": kd}


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    p = get_parameter_sets()["mid_k"]
    label = "mid_k"
    T_train = 5.0
    L = p["L"]

    pts = generate_fixed_points(L, T_train, N_int=8000, N_ic=500, N_bc=400, device=device)

    sigmas = [5.0, 12.5, 25.0]

    for sigma in sigmas:
        print(f"\n{'='*60}")
        print(f"FF-PINN: sigma={sigma}")
        print(f"{'='*60}")

        model = FFPINN(L=L, n_fourier=64, sigma=sigma).to(device)
        hist = train_ffpinn(model, p, pts, device, epochs=12000, lr=5e-4,
                            bc_w=10.0, ic_w=10.0, report_every=500,
                            label=f"mid_k_s{sigma}")

        ev = evaluate(model, p, device=device, Te=T_train)
        fdm = np.load(f"results/fdm/fdm_{label}.npz")
        k_fdm = fdm["k_dominant"]

        print(f"\n  === FF-PINN sigma={sigma} ===")
        print(f"  FDM k_dom:     {k_fdm:.2f}")
        print(f"  FF-PINN k_dom: {ev['k_dominant']:.2f}  dk={abs(ev['k_dominant']-float(k_fdm)):.3f}")

        np.savez(f"results/ffpinn_mid_k_s{sigma}.npz", **ev, k_dom=ev['k_dominant'])

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].pcolormesh(ev["x"], ev["t"], ev["u_grid"], shading="auto", cmap="RdBu_r")
        axes[0].set_title(f"FF-PINN (s={sigma}): u(x,t)")
        axes[0].set_xlabel("x"); axes[0].set_ylabel("t")
        axes[1].plot(ev["x"], ev["u_final"], "b-", lw=1.5)
        axes[1].set_title("final u(x)"); axes[1].set_xlabel("x"); axes[1].set_ylabel("u")
        axes[1].grid(True, alpha=0.3)
        axes[2].semilogy(ev["k_fft"], ev["fft_amplitude"], "b-", lw=1.5)
        axes[2].axvline(ev["k_dominant"], color="C3", ls="--", lw=1.5, label=f"FF k={ev['k_dominant']:.2f}")
        axes[2].axvline(k_fdm, color="C2", ls=":", lw=1.5, label=f"FDM k={k_fdm:.2f}")
        axes[2].set_title(f"FFT (s={sigma})"); axes[2].set_xlabel("k"); axes[2].set_ylabel("|FFT|")
        axes[2].legend(fontsize=8); axes[2].set_xlim(0, 60); axes[2].grid(True, alpha=0.3)
        plt.suptitle(f"FF-PINN mid_k (g=900): sigma={sigma}", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"results/ffpinn_s{sigma}.png", dpi=200, bbox_inches="tight")
        plt.close()

        # Loss curves (semilog)
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.semilogy(hist["epoch"], hist["total"], "C0-", lw=1.5, label="total")
        ax2.semilogy(hist["epoch"], hist["pde"], "C0--", lw=1, alpha=0.7, label="PDE")
        ax2.set_title(f"FF-PINN $\\sigma={sigma}$: Training Loss")
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
        ax2.legend(); ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"results/ffpinn_s{sigma}_loss.png", dpi=150, bbox_inches="tight")
        plt.close()

    print(f"\n{'='*60}")
    print("FF-PINN experiment complete.")
    print(f"{'='*60}")
