"""#24 FDM grid convergence study.
Verify that Nx=256 is sufficient by comparing FDM solutions at Nx = [64, 128, 256, 512].
Reports k_dominant and L2 error (relative to Nx=512 reference) for each resolution.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from numerical.fdm_solver import solve_schnakenberg_fdm
from physics.schnakenberg import get_parameter_sets, turing_analysis

os.makedirs("results/fdm_convergence", exist_ok=True)

p = get_parameter_sets()["mid_k"]  # gamma=900, the demanding case
Nx_list = [64, 128, 256, 512]
results = {}

for Nx in Nx_list:
    print(f"\n{'='*50}")
    print(f"FDM Nx={Nx}  (gamma={p['gamma']})")
    print(f"{'='*50}")
    t0 = time.time()

    p_mod = dict(p)
    p_mod["Nx"] = Nx

    sol = solve_schnakenberg_fdm(
        a=p_mod["a"], b=p_mod["b"],
        d_u=p_mod["d_u"], d_v=p_mod["d_v"],
        gamma=p_mod["gamma"],
        L=p_mod["L"], Nx=Nx,
        T=30.0, dt=0.0005, noise_amp=0.05,
        save_every=200, progress=True
    )

    # FFT with Hann window
    u_final = sol["u_history"][-1]
    u_detrend = u_final - np.mean(u_final)
    window = np.hanning(Nx)
    u_windowed = u_detrend * window
    fft_amp = np.abs(np.fft.fft(u_windowed))
    k_vals = 2 * np.pi * np.fft.fftfreq(Nx, sol["dx"])
    pos = k_vals > 0
    k_dom = k_vals[pos][np.argmax(fft_amp[pos])]

    results[Nx] = {
        "x": sol["x"], "u_final": u_final,
        "k_dom": k_dom, "k_vals": k_vals[pos], "fft_amp": fft_amp[pos],
        "dx": sol["dx"], "elapsed": time.time() - t0
    }

    print(f"  k_dom = {k_dom:.4f}, elapsed = {results[Nx]['elapsed']:.0f}s")

# ---- Richardson extrapolation estimate ----
# Using Nx=128,256,512 for order-2 extrapolation
k_dom_256 = results[256]["k_dom"]
k_dom_512 = results[512]["k_dom"]
k_dom_richardson = (4 * k_dom_512 - k_dom_256) / 3  # 2nd-order extrapolation
k_rel_diff = abs(k_dom_256 - k_dom_512) / k_dom_512 * 100

# L2 errors relative to Nx=512
u512 = results[512]["u_final"]
x512 = results[512]["x"]
l2_errors = {}
for Nx in Nx_list[:-1]:
    u_ref = np.interp(x512, results[Nx]["x"], results[Nx]["u_final"])
    l2_err = np.sqrt(np.trapz((u_ref - u512)**2, x512)) / np.sqrt(np.trapz(u512**2, x512))
    l2_errors[Nx] = l2_err

# ---- Save ----
np.savez("results/fdm_convergence/convergence_data.npz",
         Nx_list=Nx_list, l2_errors=l2_errors,
         k_dom_richardson=k_dom_richardson,
         results={str(k): v for k, v in results.items()})

# ---- Figure ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left: k_dom vs Nx
ax1.plot(Nx_list, [results[nx]["k_dom"] for nx in Nx_list], "ko-", ms=6)
ax1.axhline(k_dom_512, color="gray", ls="--", alpha=0.5, label=f"Nx=512: {k_dom_512:.4f}")
ax1.set_xlabel("Nx"); ax1.set_ylabel("k_dom")
ax1.set_title("FDM grid convergence: dominant wavenumber")
ax1.legend(); ax1.grid(True, alpha=0.3)

# Right: L2 error vs Nx
nx_plot = Nx_list[:-1]
ax2.loglog(nx_plot, [l2_errors[nx] for nx in nx_plot], "s-", ms=6)
# 2nd-order reference line
ax2.loglog(nx_plot, [l2_errors[128] * (128/nx)**2 for nx in nx_plot],
           "k--", alpha=0.5, label=r"$\mathcal{O}(N_x^{-2})$")
ax2.set_xlabel("Nx"); ax2.set_ylabel("Relative L² error")
ax2.set_title("L² error (reference: Nx=512)")
ax2.legend(); ax2.grid(True, alpha=0.3)

plt.suptitle(f"FDM Grid Convergence (γ={p['gamma']})", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("results/fdm_convergence/convergence.png", dpi=200, bbox_inches="tight")
plt.savefig("results/fdm_convergence/convergence.pdf", bbox_inches="tight")
plt.close()

# ---- Print summary ----
print(f"\n{'='*60}")
print("FDM GRID CONVERGENCE SUMMARY")
print(f"{'='*60}")
print(f"  Nx    k_dom        Δk vs 512    L² error")
print(f"  ---   ------       ----------    --------")
for Nx in Nx_list:
    dk = abs(results[Nx]["k_dom"] - k_dom_512)
    le = l2_errors.get(Nx, 0)
    print(f"  {Nx:4d}  {results[Nx]['k_dom']:.4f}       {dk:.4f}         {le:.6f}")
print(f"\n  Richardson extrap: k_dom ≈ {k_dom_richardson:.4f}")
print(f"  Rel diff 256→512:  {k_rel_diff:.3f}%")
print(f"\n  Nx=256 is {'SUFFICIENT' if k_rel_diff < 1.0 else 'MARGINAL'} for ground truth.")
