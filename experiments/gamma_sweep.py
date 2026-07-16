"""#23 Gamma sweep: test PINN across γ ∈ [220, 350, 500, 650, 900].
Produces k_dom vs γ comparison (FDM vs PINN vs theory).
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.pinn import PINN
from physics.schnakenberg import get_parameter_sets, turing_analysis, steady_state
from numerical.fdm_solver import solve_schnakenberg_fdm
from train import generate_fixed_points, train_model, evaluate_model

os.makedirs("results/gamma_sweep", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

base_p = get_parameter_sets()["low_k"]
a, b = base_p["a"], base_p["b"]
d_u, d_v = base_p["d_u"], base_p["d_v"]
L = base_p["L"]

gamma_list = [220, 350, 500, 650, 900]
T_train = 5.0

fdm_k = []
pinn_k = []
theory_k = []

for gamma in gamma_list:
    ckpt_file = f"results/gamma_sweep/gamma_{gamma}.npz"

    # -- Resume: skip completed --
    if os.path.exists(ckpt_file):
        data = np.load(ckpt_file)
        fdm_k.append(float(data["fdm_k"]))
        pinn_k.append(float(data["pinn_k"]))
        theory_k.append(float(data["theory_k"]) if data["theory_k"] != "None" else None)
        print(f"\n  [γ={gamma}] SKIP — already done (k_dom={data['pinn_k']:.3f})")
        continue

    print(f"\n{'='*50}")
    print(f"γ = {gamma}")
    p_g = {"a": a, "b": b, "d_u": d_u, "d_v": d_v, "gamma": gamma, "L": L, "Nx": 256}

    # Theory
    ta = turing_analysis(a, b, d_u, d_v, gamma, L)
    k_th = ta["k_most_unstable"] if ta["is_turing"] else None
    theory_k.append(k_th)
    print(f"  Theory k*: {k_th:.2f}" if k_th else "  Theory: no Turing")

    # FDM
    print("  Running FDM...")
    t0 = time.time()
    sol = solve_schnakenberg_fdm(a, b, d_u, d_v, gamma, L=L, Nx=256, T=30.0,
                                 dt=0.0005, progress=False)
    u_final = sol["u_history"][-1]
    window = np.hanning(256)
    u_w = (u_final - u_final.mean()) * window
    fft_amp = np.abs(np.fft.fft(u_w))
    kv = 2 * np.pi * np.fft.fftfreq(256, sol["dx"])
    pos = kv > 0
    k_fdm = kv[pos][np.argmax(fft_amp[pos])]
    fdm_k.append(k_fdm)
    print(f"  FDM k_dom: {k_fdm:.2f}  [{time.time()-t0:.0f}s]")

    # PINN (1 run per gamma — multi-seed stats from #4)
    print("  Training PINN...")
    t0 = time.time()
    torch.manual_seed(0); np.random.seed(0)
    pts = generate_fixed_points(L, T_train, 8000, 500, 400, device)
    model = PINN(L=L).to(device)
    hist = train_model(model, "pinn", p_g, pts, device=device,
                      adam_epochs=12000, lr=5e-4,
                      bc_weight=10.0, ic_weight=10.0,
                      report_every=2000, label=f"gamma{gamma}")
    ev = evaluate_model(model, p_g, device=device, T_eval=T_train)
    pinn_k.append(ev["k_dominant"])
    print(f"  PINN k_dom: {ev['k_dominant']:.2f}  [{time.time()-t0:.0f}s]")

    # save per-gamma checkpoint
    np.savez(ckpt_file, fdm_k=k_fdm, pinn_k=ev["k_dominant"],
             theory_k=k_th if k_th is not None else "None")

# ---- Save ----
np.savez("results/gamma_sweep/gamma_sweep.npz",
         gamma_list=gamma_list, fdm_k=fdm_k, pinn_k=pinn_k, theory_k=theory_k)

# ---- Plot ----
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(gamma_list, fdm_k, "s-", color="C0", ms=8, lw=1.5, label="FDM k_dom")
ax.plot(gamma_list, pinn_k, "o-", color="C3", ms=8, lw=1.5, label="PINN k_dom")
k_th_arr = np.array([t for t in theory_k if t is not None])
gamma_th = [gamma_list[i] for i, t in enumerate(theory_k) if t is not None]
ax.plot(gamma_th, k_th_arr, "--", color="gray", lw=1.5, alpha=0.7, label="Theory k*")

# Horizontal line at lowest mode
ax.axhline(6.28, color="lime", ls=":", lw=1.5, alpha=0.6, label="Lowest mode k≈6.28")

ax.set_xlabel("γ"); ax.set_ylabel("k_dom")
ax.set_title("PINN Wavenumber Recovery vs γ (Schnakenberg)")
ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("results/gamma_sweep/gamma_sweep.png", dpi=200)
plt.savefig("results/gamma_sweep/gamma_sweep.pdf")
plt.close()

print(f"\n{'='*60}")
print("GAMMA SWEEP SUMMARY")
print(f"{'='*60}")
print(f"{'γ':>6s}  {'Theory':>8s}  {'FDM':>8s}  {'PINN':>8s}  {'Match?':>8s}")
print("-" * 50)
for i, gamma in enumerate(gamma_list):
    th = theory_k[i]
    fd = fdm_k[i]
    pn = pinn_k[i]
    match = "✓" if abs(pn - fd) < 1.0 else "✗"
    th_s = f"{th:.2f}" if th is not None else "N/A"
    print(f"{gamma:6.0f}  {th_s:>8s}  {fd:8.2f}  {pn:8.2f}  {match:>8s}")

print(f"\nDone. Results in results/gamma_sweep/")
