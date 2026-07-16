"""
1D Finite Difference solver for Schnakenberg reaction-diffusion system.

Semi-implicit (IMEX) scheme:
    - Diffusion: Crank-Nicolson (implicit, 2nd order)
    - Reaction: forward Euler (explicit, 1st order)

Boundary conditions: Neumann (zero-flux, du/dx = dv/dx = 0)

Run: python numerical/fdm_solver.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from physics.schnakenberg import get_parameter_sets, turing_analysis, steady_state
import time


def solve_schnakenberg_fdm(
    a: float, b: float,
    d_u: float, d_v: float, gamma: float,
    L: float = 1.0, Nx: int = 256,
    T: float = 30.0, dt: float = 0.0005,
    noise_amp: float = 0.05,
    save_every: int = 200,
    progress: bool = True
) -> dict:
    """Solve 1D Schnakenberg model using Crank-Nicolson + explicit reaction.

    Spatial scheme:
        (I - D*dt/2 * Lap) * u^{n+1} = (I + D*dt/2 * Lap) * u^n + dt * gamma * R(u^n)

    Returns:
        dict: x, t, u_history, v_history, params
    """
    dx = L / (Nx - 1)
    x = np.linspace(0, L, Nx)

    # Check stability
    cfl_u = d_u * dt / dx**2
    cfl_v = d_v * dt / dx**2
    if progress:
        print(f"  dx={dx:.5f}, dt={dt:.5f}, CFL_u={cfl_u:.3f}, CFL_v={cfl_v:.3f}")

    # ---- Laplacian matrix (3-point, Neumann BC) ----
    main_diag = -2 * np.ones(Nx)
    off_diag = np.ones(Nx - 1)
    Lap = (np.diag(main_diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)) / dx**2

    # Neumann BC correction: du/dx=0 means ghost points mirror interior
    Lap[0, 0] = -2 / dx**2; Lap[0, 1] = 2 / dx**2
    Lap[-1, -1] = -2 / dx**2; Lap[-1, -2] = 2 / dx**2

    I = np.eye(Nx)

    # Crank-Nicolson operators
    A_u_lhs = I - 0.5 * d_u * dt * Lap
    A_u_rhs = I + 0.5 * d_u * dt * Lap
    A_v_lhs = I - 0.5 * d_v * dt * Lap
    A_v_rhs = I + 0.5 * d_v * dt * Lap

    # Pre-compute inverses
    from numpy.linalg import inv
    A_u_lhs_inv = inv(A_u_lhs)
    A_v_lhs_inv = inv(A_v_lhs)

    # ---- Initial conditions: steady state + noise ----
    u_s, v_s = steady_state(a, b)
    rng = np.random.RandomState(42)
    u = u_s + noise_amp * (rng.rand(Nx) - 0.5)
    rng = np.random.RandomState(43)
    v = v_s + noise_amp * (rng.rand(Nx) - 0.5)

    # ---- Time stepping ----
    Nt = int(T / dt)
    Nt_save = Nt // save_every + 1

    t_save = np.zeros(Nt_save)
    u_history = np.zeros((Nt_save, Nx))
    v_history = np.zeros((Nt_save, Nx))
    u_history[0] = u; v_history[0] = v; t_save[0] = 0.0

    save_idx = 1
    t_start = time.time()

    for n in range(1, Nt + 1):
        # Reaction at current state
        r_u = gamma * (a - u + u**2 * v)
        r_v = gamma * (b - u**2 * v)

        u_new = A_u_lhs_inv @ (A_u_rhs @ u + dt * r_u)
        v_new = A_v_lhs_inv @ (A_v_rhs @ v + dt * r_v)

        # Physical bounds
        u_new = np.maximum(u_new, 0)
        v_new = np.maximum(v_new, 0)

        u, v = u_new, v_new

        if n % save_every == 0 and save_idx < Nt_save:
            u_history[save_idx] = u
            v_history[save_idx] = v
            t_save[save_idx] = n * dt
            save_idx += 1

        if progress and n % max(Nt // 10, 1) == 0:
            print(f"    t={n*dt:.1f}/{T}  ({100*n/Nt:.0f}%)  [{time.time()-t_start:.0f}s]")

    elapsed = time.time() - t_start
    if progress:
        print(f"    Done in {elapsed:.0f}s")

    return {
        "x": x, "t": t_save,
        "u_history": u_history, "v_history": v_history,
        "dx": dx, "dt": dt,
        "params": {"a": a, "b": b, "d_u": d_u, "d_v": d_v, "gamma": gamma, "L": L, "Nx": Nx}
    }


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from numpy.fft import fft, fftfreq

    os.makedirs("results/fdm", exist_ok=True)

    param_sets = get_parameter_sets()

    for label, p in param_sets.items():
        print(f"\n{'='*60}")
        print(f"FDM: {label}  (gamma={p['gamma']})")
        print(f"{'='*60}")

        # Turing theory
        ta = turing_analysis(p["a"], p["b"], p["d_u"], p["d_v"], p["gamma"], p["L"])

        # FDM solve
        result = solve_schnakenberg_fdm(
            a=p["a"], b=p["b"],
            d_u=p["d_u"], d_v=p["d_v"],
            gamma=p["gamma"],
            L=p["L"], Nx=p["Nx"],
            T=30.0, dt=0.0005,
            noise_amp=0.05,
            save_every=200
        )

        # ---- FFT of final u ----
        u_final = result["u_history"][-1]
        u_detrend = u_final - np.mean(u_final)
        window = np.hanning(p["Nx"])
        u_windowed = u_detrend * window
        fft_amp = np.abs(fft(u_windowed))
        k_vals = 2 * np.pi * fftfreq(p["Nx"], result["dx"])
        pos = k_vals > 0
        k_pos = k_vals[pos]; fft_pos = fft_amp[pos]
        k_dom = k_pos[np.argmax(fft_pos)]

        # ---- Plot: 4 panels ----
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # u(x,t)
        im0 = axes[0, 0].pcolormesh(result["x"], result["t"], result["u_history"],
                                     shading="auto", cmap="RdBu_r")
        axes[0, 0].set_title(f"u(x,t) — {label}")
        axes[0, 0].set_xlabel("x"); axes[0, 0].set_ylabel("t")
        plt.colorbar(im0, ax=axes[0, 0])

        # v(x,t)
        im1 = axes[0, 1].pcolormesh(result["x"], result["t"], result["v_history"],
                                     shading="auto", cmap="RdBu_r")
        axes[0, 1].set_title(f"v(x,t) — {label}")
        axes[0, 1].set_xlabel("x"); axes[0, 1].set_ylabel("t")
        plt.colorbar(im1, ax=axes[0, 1])

        # Final u(x)
        axes[1, 0].plot(result["x"], u_final, "b-", lw=1.5)
        axes[1, 0].set_title(f"Final u(x) — {label}")
        axes[1, 0].set_xlabel("x"); axes[1, 0].set_ylabel("u")
        axes[1, 0].grid(True, alpha=0.3)

        # FFT
        axes[1, 1].semilogy(k_pos, fft_pos, "b-", lw=1.5)
        axes[1, 1].axvline(k_dom, color="C3", ls="--", lw=1.5,
                           label=f"FDM k_dom = {k_dom:.2f}")
        if ta["is_turing"]:
            k_th = ta["k_most_unstable"]
            axes[1, 1].axvline(k_th, color="C2", ls=":", lw=1.5,
                               label=f"Theory k* = {k_th:.2f}")
        axes[1, 1].set_title(f"FFT of final u — {label}")
        axes[1, 1].set_xlabel("k"); axes[1, 1].set_ylabel("|FFT(u)|")
        axes[1, 1].legend(fontsize=9)
        axes[1, 1].set_xlim(0, 80)
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle(f"Schnakenberg FDM: {label}  (γ={p['gamma']}, a={p['a']}, b={p['b']})",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"results/fdm/fdm_{label}.png", dpi=200, bbox_inches="tight")
        plt.close()

        # Save data
        np.savez(
            f"results/fdm/fdm_{label}.npz",
            x=result["x"], t=result["t"],
            u_history=result["u_history"], v_history=result["v_history"],
            u_final=u_final,
            k_fft=k_pos, fft_amplitude=fft_pos,
            k_dominant=k_dom,
            k_theory=ta.get("k_most_unstable", None),
            wavelength_theory=ta.get("wavelength_theory", None)
        )

        # Print results
        print(f"\n  --- {label} Summary ---")
        print(f"  k_dominant (FFT):     {k_dom:.2f}")
        print(f"  k_theory (LSA):       {ta['k_most_unstable']:.2f}" if ta["is_turing"] else "  k_theory: N/A")
        print(f"  λ_theory:             {ta['wavelength_theory']:.4f}" if ta["is_turing"] else "")
        print(f"  u range (final):      [{u_final.min():.4f}, {u_final.max():.4f}]")
        print(f"  v range (final):      [{result['v_history'][-1].min():.4f}, {result['v_history'][-1].max():.4f}]")

    print("\n" + "="*60)
    print("All FDM simulations complete. Results in results/fdm/")
    print("="*60)
