"""Regenerate all publication-quality figures with unified styling.
Run AFTER all experiments are complete.
Outputs PDF (vector) + PNG to results/figures/

Revision 2026-07-16: External AI review fixes
- All axes: x → "x / L", k → "k (rad/unit length)", colorbar → "u (a.u.)"
- Fig 3/4: GridSpec width_ratios, shared colorbar, FDM cropped to t=5
- Fig 5: y-label "PDE residual loss", shared y-range
- Fig 6: Fixed k_dom y-range, proper math formatting
- Fig 7: Shared colorbar, sigma as row labels
- Fig 8: Bar value labels, "NTK condition number estimate"
- Fig 9: Reduced ms=6, distinguishable reference lines
- Fig 10: Simplified legend, fixed percentage labels
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Unified rcParams
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.format": "pdf",
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "text.usetex": False,
    "figure.constrained_layout.use": False,  # manual GridSpec for multi-panel
})

from models.pinn import PINN, HCPINN, FFPINN
from physics.schnakenberg import get_parameter_sets, turing_analysis
from train import evaluate_model

os.makedirs("results/figures", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

p_sets = get_parameter_sets()
L = 1.0

def save_both(fig, name):
    fig.savefig(f"results/figures/{name}.pdf")
    fig.savefig(f"results/figures/{name}.png", dpi=300)
    print(f"  Saved {name}")

# Shared style helpers
LABEL_X = "x / L"
LABEL_K = "k (rad / unit length)"
LABEL_U = "u (a.u.)"
LABEL_FFT = "|FFT(u(x, t = T_final))| (a.u.)"

# ============================================================
# Fig 1: FDM low_k benchmark (γ=220)
# ============================================================
print("\n=== Fig 1: FDM low_k ===")
fdm_low = np.load("results/fdm/fdm_low_k.npz")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

im = axes[0].pcolormesh(fdm_low["x"], fdm_low["t"], fdm_low["u_history"],
                         shading="auto", cmap="RdBu_r")
axes[0].set_xlabel(LABEL_X); axes[0].set_ylabel("t")
axes[0].set_title("u(x, t)")
plt.colorbar(im, ax=axes[0], label=LABEL_U)

axes[1].plot(fdm_low["x"], fdm_low["u_history"][-1], "b-", lw=1.5)
axes[1].set_xlabel(LABEL_X); axes[1].set_ylabel(LABEL_U)
axes[1].set_title("Final u(x)")
axes[1].grid(True, alpha=0.3)

axes[2].semilogy(fdm_low["k_fft"], fdm_low["fft_amplitude"], "b-", lw=1.5)
axes[2].axvline(fdm_low["k_dominant"], color="C3", ls="--", lw=1.5,
                label=f"$k_\\mathrm{{dom}}$={fdm_low['k_dominant']:.2f}")
axes[2].axvline(fdm_low["k_theory"], color="C2", ls=":", lw=1.5,
                label=f"$k^*$={fdm_low['k_theory']:.2f}")
axes[2].set_xlabel(LABEL_K)
axes[2].set_ylabel(LABEL_FFT)
axes[2].set_title("FFT spectrum")
axes[2].legend(fontsize=9, frameon=False); axes[2].grid(True, alpha=0.3)

fig.suptitle("Fig. 1: FDM reference — low-$k$ regime ($\\gamma=220$, $k^*\\approx 6.24$)",
             fontsize=14, fontweight="bold")
save_both(fig, "fig1_fdm_low_k")

# ============================================================
# Fig 2: FDM mid_k benchmark (γ=900)
# ============================================================
print("=== Fig 2: FDM mid_k ===")
fdm_mid = np.load("results/fdm/fdm_mid_k.npz")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

im = axes[0].pcolormesh(fdm_mid["x"], fdm_mid["t"], fdm_mid["u_history"],
                         shading="auto", cmap="RdBu_r")
axes[0].set_xlabel(LABEL_X); axes[0].set_ylabel("t")
axes[0].set_title("u(x, t)")
plt.colorbar(im, ax=axes[0], label=LABEL_U)

axes[1].plot(fdm_mid["x"], fdm_mid["u_history"][-1], "b-", lw=1.5)
axes[1].set_xlabel(LABEL_X); axes[1].set_ylabel(LABEL_U)
axes[1].set_title("Final u(x)")
axes[1].grid(True, alpha=0.3)

axes[2].semilogy(fdm_mid["k_fft"], fdm_mid["fft_amplitude"], "b-", lw=1.5)
axes[2].axvline(fdm_mid["k_dominant"], color="C3", ls="--", lw=1.5,
                label=f"$k_\\mathrm{{dom}}$={fdm_mid['k_dominant']:.2f}")
axes[2].axvline(fdm_mid["k_theory"], color="C2", ls=":", lw=1.5,
                label=f"$k^*$={fdm_mid['k_theory']:.2f}")
axes[2].set_xlabel(LABEL_K)
axes[2].set_ylabel(LABEL_FFT)
axes[2].set_title("FFT spectrum")
axes[2].legend(fontsize=9, frameon=False); axes[2].grid(True, alpha=0.3)

fig.suptitle("Fig. 2: FDM reference — mid-$k$ regime ($\\gamma=900$, $k^*\\approx 12.61$)",
             fontsize=14, fontweight="bold")
save_both(fig, "fig2_fdm_mid_k")

# ============================================================
# Fig 3: low_k 3-way comparison (PINN vs HC-PINN vs FDM)
# GridSpec with width_ratios, shared colorbar per row, FDM cropped to t=5
# ============================================================
print("=== Fig 3: low_k 3-way ===")
p_low = p_sets["low_k"]

# Load multi-seed models (use seed=0 as representative)
pinn_low = PINN(L=L).to(device)
pinn_low.load_state_dict(torch.load("results/multi_seed/PINN_220_s0.pt", map_location=device))
hc_low = HCPINN(L=L).to(device)
hc_low.load_state_dict(torch.load("results/multi_seed/HCPINN_220_s0.pt", map_location=device))
ev_pinn_low = evaluate_model(pinn_low, p_low, device)
ev_hc_low = evaluate_model(hc_low, p_low, device)

T_eval = 5.0
fdm_low_data = np.load("results/fdm/fdm_low_k.npz")
k_fdm_low = fdm_low_data["k_dominant"]
# Crop FDM to t <= T_eval for consistency
fdm_t_mask = fdm_low_data["t"] <= T_eval

fig = plt.figure(figsize=(16, 13))
gs = GridSpec(3, 4, figure=fig, width_ratios=[1, 1, 1, 0.05],
              hspace=0.35, wspace=0.35)

models_info = [
    ("PINN (soft BC)", ev_pinn_low),
    ("HC-PINN (hard BC)", ev_hc_low),
    ("FDM (reference)", None),
]

for col in range(3):
    name, ev = models_info[col]

    # Row 0: u(x,t) heatmap
    ax0 = fig.add_subplot(gs[0, col])
    if col < 2:
        im = ax0.pcolormesh(ev["x"], ev["t"], ev["u_grid"],
                            shading="auto", cmap="RdBu_r")
        ax0.set_title(f"{name}\nu(x, t)")
    else:
        im = ax0.pcolormesh(fdm_low_data["x"], fdm_low_data["t"][fdm_t_mask],
                            fdm_low_data["u_history"][fdm_t_mask],
                            shading="auto", cmap="RdBu_r")
        ax0.set_title(f"{name}\nu(x, t)")
    ax0.set_xlabel(LABEL_X); ax0.set_ylabel("t")

    # Row 1: Final profile
    ax1 = fig.add_subplot(gs[1, col])
    if col < 2:
        ax1.plot(ev["x"], ev["u_final"], "C0-", lw=1.5)
        ax1.set_title(f"{name}\nu(x, t = {T_eval:.0f})")
    else:
        ax1.plot(fdm_low_data["x"], fdm_low_data["u_history"][-1], "C0-", lw=1.5)
        ax1.set_title(f"{name}\nu(x, t = {T_eval:.0f})")
    ax1.set_xlabel(LABEL_X); ax1.set_ylabel(LABEL_U)
    ax1.grid(True, alpha=0.3)

    # Row 2: FFT spectrum
    ax2 = fig.add_subplot(gs[2, col])
    if col < 2:
        ax2.semilogy(ev["k_fft"], ev["fft_amplitude"], "C0-", lw=1.5)
        ax2.axvline(ev["k_dominant"], color="C3", ls="--", lw=1.5,
                    label=f"$k_\\mathrm{{dom}}$={ev['k_dominant']:.2f}")
    else:
        ax2.semilogy(fdm_low_data["k_fft"], fdm_low_data["fft_amplitude"], "C0-", lw=1.5)
        ax2.axvline(k_fdm_low, color="C3", ls="--", lw=1.5,
                    label=f"$k_\\mathrm{{dom}}$={k_fdm_low:.2f}")
    ax2.axvline(fdm_low_data["k_theory"], color="C2", ls=":", lw=1.5,
                label=f"$k^*$={fdm_low_data['k_theory']:.2f}")
    ax2.set_xlabel(LABEL_K); ax2.set_ylabel(LABEL_FFT)
    ax2.set_title(f"{name}\nFFT")
    ax2.legend(fontsize=9, frameon=False)
    ax2.set_xlim(0, 60); ax2.grid(True, alpha=0.3)

# Shared colorbar (rightmost column, row 0)
cax = fig.add_subplot(gs[0, 3])
plt.colorbar(im, cax=cax, label=LABEL_U)
# Hide unused colorbar slots
for r in [1, 2]:
    ax_hide = fig.add_subplot(gs[r, 3])
    ax_hide.set_visible(False)

fig.suptitle("Fig. 3: Low-$k$ regime comparison ($\\gamma=220$, $k^*\\approx 6.24$)",
             fontsize=14, fontweight="bold")
save_both(fig, "fig3_low_k_comparison")

# ============================================================
# Fig 4: mid_k 3-way comparison (both fail)
# Same GridSpec layout as Fig 3
# ============================================================
print("=== Fig 4: mid_k 3-way ===")
p_mid = p_sets["mid_k"]

pinn_mid = PINN(L=L).to(device)
pinn_mid.load_state_dict(torch.load("results/multi_seed/PINN_900_s0.pt", map_location=device))
hc_mid = HCPINN(L=L).to(device)
hc_mid.load_state_dict(torch.load("results/multi_seed/HCPINN_900_s0.pt", map_location=device))
ev_pinn_mid = evaluate_model(pinn_mid, p_mid, device)
ev_hc_mid = evaluate_model(hc_mid, p_mid, device)

fdm_mid_data = np.load("results/fdm/fdm_mid_k.npz")
k_fdm_mid = fdm_mid_data["k_dominant"]
fdm_mid_t_mask = fdm_mid_data["t"] <= T_eval

fig = plt.figure(figsize=(16, 13))
gs = GridSpec(3, 4, figure=fig, width_ratios=[1, 1, 1, 0.05],
              hspace=0.35, wspace=0.35)

models_info_mid = [
    ("PINN (soft BC)", ev_pinn_mid),
    ("HC-PINN (hard BC)", ev_hc_mid),
    ("FDM (reference)", None),
]

for col in range(3):
    name, ev = models_info_mid[col]

    ax0 = fig.add_subplot(gs[0, col])
    if col < 2:
        im = ax0.pcolormesh(ev["x"], ev["t"], ev["u_grid"],
                            shading="auto", cmap="RdBu_r")
    else:
        im = ax0.pcolormesh(fdm_mid_data["x"], fdm_mid_data["t"][fdm_mid_t_mask],
                            fdm_mid_data["u_history"][fdm_mid_t_mask],
                            shading="auto", cmap="RdBu_r")
    ax0.set_title(f"{name}\nu(x, t)")
    ax0.set_xlabel(LABEL_X); ax0.set_ylabel("t")

    ax1 = fig.add_subplot(gs[1, col])
    if col < 2:
        ax1.plot(ev["x"], ev["u_final"], "C0-", lw=1.5)
    else:
        ax1.plot(fdm_mid_data["x"], fdm_mid_data["u_history"][-1], "C0-", lw=1.5)
    ax1.set_title(f"{name}\nu(x, t = {T_eval:.0f})")
    ax1.set_xlabel(LABEL_X); ax1.set_ylabel(LABEL_U)
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[2, col])
    if col < 2:
        ax2.semilogy(ev["k_fft"], ev["fft_amplitude"], "C0-", lw=1.5)
        ax2.axvline(ev["k_dominant"], color="C3", ls="--", lw=1.5,
                    label=f"$k_\\mathrm{{dom}}$={ev['k_dominant']:.2f}")
    else:
        ax2.semilogy(fdm_mid_data["k_fft"], fdm_mid_data["fft_amplitude"], "C0-", lw=1.5)
        ax2.axvline(k_fdm_mid, color="C3", ls="--", lw=1.5,
                    label=f"$k_\\mathrm{{dom}}$={k_fdm_mid:.2f}")
    ax2.axvline(fdm_mid_data["k_theory"], color="C2", ls=":", lw=1.5,
                label=f"$k^*$={fdm_mid_data['k_theory']:.2f}")
    ax2.set_xlabel(LABEL_K); ax2.set_ylabel(LABEL_FFT)
    ax2.set_title(f"{name}\nFFT")
    ax2.legend(fontsize=9, frameon=False)
    ax2.set_xlim(0, 60); ax2.grid(True, alpha=0.3)

cax = fig.add_subplot(gs[0, 3])
plt.colorbar(im, cax=cax, label=LABEL_U)
for r in [1, 2]:
    ax_hide = fig.add_subplot(gs[r, 3])
    ax_hide.set_visible(False)

fig.suptitle("Fig. 4: Mid-$k$ regime comparison ($\\gamma=900$, $k^*\\approx 12.61$)",
             fontsize=14, fontweight="bold")
save_both(fig, "fig4_mid_k_comparison")

# ============================================================
# Fig 5: Training loss curves (low_k + mid_k)
# Try loading cached loss histories first; retrain only if missing
# ============================================================
print("=== Fig 5: Loss curves ===")

histories = {}
cache_dir = "results/fig5_cache"
os.makedirs(cache_dir, exist_ok=True)
need_retrain = False

for label, pl in [("low_k", p_low), ("mid_k", p_mid)]:
    cache_path = f"{cache_dir}/loss_{label}.npz"
    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        histories[label] = ({"epoch": data["epoch_p"], "loss_pde": data["loss_pde_p"]},
                            {"epoch": data["epoch_h"], "loss_pde": data["loss_pde_h"]})
        print(f"  Loaded cached {label} loss history")
    else:
        need_retrain = True

if need_retrain:
    from train import generate_fixed_points, train_model
    T_train = 5.0

    for label, pl in [("low_k", p_low), ("mid_k", p_mid)]:
        cache_path = f"{cache_dir}/loss_{label}.npz"
        if label in histories:
            continue  # already loaded from cache
        print(f"  Retraining {label}... (this takes ~5 min)")
        pts = generate_fixed_points(L, T_train, 8000, 500, 400, device)

        torch.manual_seed(0); np.random.seed(0)
        model_p = PINN(L=L).to(device)
        hist_p = train_model(model_p, "pinn", pl, pts, device, adam_epochs=12000, lr=5e-4,
                             bc_weight=10.0, ic_weight=10.0, report_every=500, label=label)

        torch.manual_seed(0); np.random.seed(0)
        model_h = HCPINN(L=L).to(device)
        hist_h = train_model(model_h, "hcpinn", pl, pts, device, adam_epochs=12000, lr=5e-4,
                             bc_weight=0.0, ic_weight=10.0, report_every=500, label=label)

        # Cache loss histories
        np.savez(cache_path,
                 epoch_p=hist_p["epoch"], loss_pde_p=hist_p["loss_pde"],
                 epoch_h=hist_h["epoch"], loss_pde_h=hist_h["loss_pde"])
        histories[label] = (hist_p, hist_h)
        print(f"    Cached to {cache_path}")

# Collect all loss for shared y-range
all_loss = []
for _, (hp, hh) in histories.items():
    all_loss.extend(hp["loss_pde"])
    all_loss.extend(hh["loss_pde"])
all_loss = np.array(all_loss)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

y_min = max(float(np.min(all_loss[all_loss > 0])) * 0.5, 1e-8)
y_max = float(np.max(all_loss)) * 2.0

for ax, (label, pl) in [(ax1, ("low_k", p_low)), (ax2, ("mid_k", p_mid))]:
    hist_p, hist_h = histories[label]
    ax.semilogy(hist_p["epoch"], hist_p["loss_pde"], "C0-", lw=1.5, alpha=0.8, label="PINN PDE")
    ax.semilogy(hist_h["epoch"], hist_h["loss_pde"], "C1--", lw=1.5, alpha=0.8, label="HC-PINN PDE")
    ax.set_xlabel("Epoch"); ax.set_ylabel("PDE residual loss (a.u.)")
    ax.set_title(f"{label} ($\\gamma={pl['gamma']}$)")
    ax.legend(fontsize=9, frameon=False); ax.grid(True, alpha=0.3)
    ax.set_ylim(y_min, y_max)

fig.suptitle("Fig. 5: Training loss curves", fontsize=14, fontweight="bold")
save_both(fig, "fig5_loss_curves")

# ============================================================
# Fig 6: Spectral evolution heatmap
# Fixed k_dom y-range, distinguishable dashed lines, proper math
# ============================================================
print("=== Fig 6: Spectral evolution ===")
sev = np.load("results/spectral_evolution/spectral_history.npz")
chk = sev["checkpoints"]
kv_all = sev["k_vals"]
amps = sev["amplitudes"]

fig, (ax_heat, ax_kdom) = plt.subplots(1, 2, figsize=(14, 5.5),
                                        gridspec_kw={"width_ratios": [2, 1]})

k_max = 40
mask = kv_all <= k_max
kv = kv_all[mask]

im = ax_heat.pcolormesh(kv, chk, np.log10(np.maximum(amps[:, mask], 1e-8)),
                         shading="auto", cmap="inferno")
# Use distinct line styles for grayscale-distinguishable reference lines
ax_heat.axvline(12.57, color="white", ls=(0, (4, 2)), lw=2.5, alpha=0.9,
                label="$k^*=12.57$ (target)")
ax_heat.axvline(6.28, color="lime", ls=(0, (1, 3)), lw=2.5, alpha=0.8,
                label="$k\\approx 6.28$ (lowest)")
ax_heat.set_xlabel(LABEL_K); ax_heat.set_ylabel("Epoch")
ax_heat.set_title("$\\log_{10}\\,|\\mathrm{FFT}(\\hat{u})|$")
ax_heat.legend(fontsize=9, frameon=False, loc="lower right")
plt.colorbar(im, ax=ax_heat, label="$\\log_{10}\\,|\\mathrm{FFT}|$")

# Right: k_dom vs epoch — fixed y-range to show target k*
k_doms = [kv[np.argmax(amps[i, mask])] for i in range(len(chk))]
ax_kdom.plot(chk, k_doms, "o-", color="C3", ms=5, lw=1.5)
ax_kdom.axhline(12.57, color="gray", ls=(0, (4, 2)), lw=2, alpha=0.8,
                label="$k^*=12.57$ (target)")
ax_kdom.axhline(6.28, color="green", ls=(0, (1, 3)), lw=2, alpha=0.7,
                label="$k\\approx 6.28$ (lowest)")
ax_kdom.set_xlabel("Epoch"); ax_kdom.set_ylabel("$k_\\mathrm{dom}$")
ax_kdom.set_title("Dominant wavenumber")
# Fixed y-range so target k* is visible even if never reached
ax_kdom.set_ylim(0, max(max(k_doms) * 1.3, 20))
ax_kdom.legend(fontsize=9, frameon=False); ax_kdom.grid(True, alpha=0.3)

fig.suptitle("Fig. 6: Spectral evolution — PINN mid-$k$ ($\\gamma=900$)",
             fontsize=14, fontweight="bold")
save_both(fig, "fig6_spectral_evolution")

# ============================================================
# Fig 7: FF-PINN sigma sweep
# Shared colorbar, sigma as row labels, stronger k* reference
# ============================================================
print("=== Fig 7: FF-PINN sweep ===")
sigmas = [5.0, 12.5, 25.0]
sigma_strs = ["5.0", "12.5", "25.0"]

fig = plt.figure(figsize=(16, 13))
gs = GridSpec(3, 4, figure=fig, width_ratios=[1, 1, 1, 0.05],
              hspace=0.4, wspace=0.35)

for row, sigma in enumerate(sigmas):
    sigma_str = str(sigma).replace(".0", "")
    model = FFPINN(L=L, n_fourier=64, sigma=sigma).to(device)
    model.load_state_dict(torch.load(f"results/multi_seed/FFPINN_s{sigma_str}_s0.pt", map_location=device))
    ev = evaluate_model(model, p_mid, device)

    # u(x,t) — capture im for shared colorbar
    ax0 = fig.add_subplot(gs[row, 0])
    im = ax0.pcolormesh(ev["x"], ev["t"], ev["u_grid"], shading="auto", cmap="RdBu_r")
    ax0.set_title(f"$\\sigma={sigma_strs[row]}$: $u(x, t)$")
    ax0.set_xlabel(LABEL_X); ax0.set_ylabel("t")

    # Final profile
    ax1 = fig.add_subplot(gs[row, 1])
    ax1.plot(ev["x"], ev["u_final"], "C0-", lw=1.5)
    ax1.set_title(f"$\\sigma={sigma_strs[row]}$: $u(x, t = {T_eval:.0f})$")
    ax1.set_xlabel(LABEL_X); ax1.set_ylabel(LABEL_U)
    ax1.grid(True, alpha=0.3)

    # FFT
    ax2 = fig.add_subplot(gs[row, 2])
    ax2.semilogy(ev["k_fft"], ev["fft_amplitude"], "C0-", lw=1.5)
    ax2.axvline(ev["k_dominant"], color="C3", ls="--", lw=1.5,
                label=f"$k_\\mathrm{{dom}}$={ev['k_dominant']:.2f}")
    # Stronger k* reference line
    ax2.axvline(k_fdm_mid, color="black", ls=(0, (3, 3)), lw=2, alpha=0.7,
                label=f"$k^*$={k_fdm_mid:.2f}")
    ax2.set_xlabel(LABEL_K); ax2.set_ylabel(LABEL_FFT)
    ax2.set_title(f"$\\sigma={sigma_strs[row]}$: FFT")
    ax2.legend(fontsize=9, frameon=False)
    ax2.set_xlim(0, 60); ax2.grid(True, alpha=0.3)

# Shared colorbar
cax = fig.add_subplot(gs[0, 3])
plt.colorbar(im, cax=cax, label=LABEL_U)
for r in [1, 2]:
    ax_hide = fig.add_subplot(gs[r, 3])
    ax_hide.set_visible(False)

fig.suptitle("Fig. 7: FF-PINN $\\sigma$ sweep ($\\gamma=900$, $k^*\\approx 12.61$)",
             fontsize=14, fontweight="bold")
save_both(fig, "fig7_ffpinn_sweep")

# ============================================================
# Fig 8: NTK condition number bar chart
# Value labels on bars, "NTK condition number estimate"
# ============================================================
print("=== Fig 8: NTK analysis ===")
ntk = np.load("results/ntk_analysis/ntk_results.npz", allow_pickle=True)["results"].item()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

sigma_list = list(ntk.keys())
bar_colors = ["C0", "C1", "C2"]

# Left: PDE residual
pde_vals = [ntk[s]["pde_residual"] for s in sigma_list]
bars1 = ax1.bar(range(len(sigma_list)), pde_vals,
                color=bar_colors, tick_label=[f"σ={s}" for s in sigma_list])
for bar, val in zip(bars1, pde_vals):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.05,
             f"{val:.1e}", ha="center", va="bottom", fontsize=8)
ax1.set_yscale("log"); ax1.set_ylabel("PDE residual (a.u.)")
ax1.set_title("PDE residual at $t=T_\\mathrm{final}$")
ax1.grid(True, alpha=0.3, axis="y")

# Right: Condition number
cond_vals = [ntk[s]["condition_estimate"] for s in sigma_list]
bars2 = ax2.bar(range(len(sigma_list)), cond_vals,
                color=bar_colors, tick_label=[f"σ={s}" for s in sigma_list])
for bar, val in zip(bars2, cond_vals):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.05,
             f"{val:.1f}", ha="center", va="bottom", fontsize=8)
ax2.set_yscale("log"); ax2.set_ylabel("NTK condition number estimate")
ax2.set_title("NTK condition estimate")
ax2.grid(True, alpha=0.3, axis="y")

fig.suptitle("Fig. 8: NTK analysis — FF-PINN mid-$k$ ($\\gamma=900$)",
             fontsize=14, fontweight="bold")
save_both(fig, "fig8_ntk_analysis")

# ============================================================
# Fig 9: Gamma sweep (k_dom vs γ)
# Reduced markers, better line distinguishability
# ============================================================
print("=== Fig 9: Gamma sweep ===")
gs = np.load("results/gamma_sweep/gamma_sweep.npz")
gamma_list = gs["gamma_list"]
fdm_k = gs["fdm_k"]
pinn_k = gs["pinn_k"]

# Theory values
theory_k = []
for g in gamma_list:
    ta = turing_analysis(0.1, 0.9, 1.0, 40.0, g, 1.0)
    if ta["is_turing"]:
        theory_k.append(ta["k_most_unstable"])
    else:
        theory_k.append(None)

fig, ax = plt.subplots(figsize=(9, 6))

ax.plot(gamma_list, fdm_k, "s-", color="C0", ms=7, lw=1.5, label="FDM $k_\\mathrm{dom}$")
ax.plot(gamma_list, pinn_k, "D-", color="C3", ms=7, lw=1.5, label="PINN $k_\\mathrm{dom}$")
k_th_arr = np.array([t for t in theory_k if t is not None])
gamma_th = [gamma_list[i] for i, t in enumerate(theory_k) if t is not None]
ax.plot(gamma_th, k_th_arr, "-", color="gray", lw=1.5, alpha=0.8, label="Theory $k^*$")
# Lowest mode line: use a distinct dash-dot pattern
ax.axhline(6.28, color="green", ls=(0, (3, 5, 1, 5)), lw=1.5, alpha=0.7,
           label="Lowest mode $k\\approx 6.28$")

ax.set_xlabel("$\\gamma$ (dimensionless)"); ax.set_ylabel("$k_\\mathrm{dom}$")
ax.set_title("PINN wavenumber recovery vs $\\gamma$")
ax.legend(fontsize=9, frameon=False); ax.grid(True, alpha=0.3)

fig.suptitle("Fig. 9: Parameter sweep — $k_\\mathrm{dom}$ vs $\\gamma$",
             fontsize=14, fontweight="bold")
save_both(fig, "fig9_gamma_sweep")

# ============================================================
# Fig 10: Spectral residual decomposition
# Simplified legend, anchored percentage labels
# ============================================================
print("=== Fig 10: Spectral residual ===")
sr = np.load("results/spectral_residual/residual_data.npz")
kv_sr = sr["k_vals"]
u_fft_sr = sr["u_fft"]
R_fft_sr = sr["R_fft"]
low_frac = float(sr["low_frac"])
high_frac = float(sr["high_frac"])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

k_max = 60
km = kv_sr <= k_max

# Left: Solution spectrum — simplified 2-entry legend
ax1.semilogy(kv_sr[km], u_fft_sr[km], "C0-", lw=1.5)
ax1.axvline(kv_sr[np.argmax(u_fft_sr)], color="C0", ls="--", lw=1.5,
            label=f"$k_\\mathrm{{dom}}$={kv_sr[np.argmax(u_fft_sr)]:.2f}")
ax1.axvline(12.57, color="k", ls=":", lw=1.5, label="$k^*=12.57$")
ax1.set_xlabel(LABEL_K)
ax1.set_ylabel(LABEL_FFT)
ax1.set_title("Solution spectrum")
ax1.legend(fontsize=9, frameon=False); ax1.grid(True, alpha=0.3)

# Right: PDE residual spectrum
ax2.semilogy(kv_sr[km], R_fft_sr[km], "C3-", lw=1.5, label="$|\\mathrm{FFT}[R]|$")
ax2.axvline(12.57, color="k", ls="--", lw=1.5, alpha=0.5, label="$k^*=12.57$")
y_lo, y_hi = ax2.get_ylim()
ax2.fill_between([0, 12.57], float(y_lo), float(y_hi), alpha=0.15, color="C0")
ax2.fill_between([12.57, float(k_max)], float(y_lo), float(y_hi), alpha=0.15, color="C3")
# Place percentage text inside the shaded zones
mid_low = 6.3
mid_high = (12.57 + k_max) / 2
ax2.text(mid_low, y_hi * 0.85, f"$k<k^*$: {low_frac:.0f}%",
         ha="center", va="top", fontsize=9, color="C0", fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))
ax2.text(mid_high, y_hi * 0.85, f"$k\\geq k^*$: {high_frac:.0f}%",
         ha="center", va="top", fontsize=9, color="C3", fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))
ax2.set_ylim(float(y_lo), float(y_hi))
ax2.set_xlabel(LABEL_K)
ax2.set_ylabel("$|\\mathrm{FFT}(\\mathrm{PDE\\ residual})|$ (a.u.)")
ax2.set_title("PDE residual spectrum")
ax2.legend(fontsize=9, frameon=False); ax2.grid(True, alpha=0.3)

fig.suptitle("Fig. 10: Spectral residual decomposition — PINN mid-$k$ ($\\gamma=900$)",
             fontsize=14, fontweight="bold")
save_both(fig, "fig10_spectral_residual")

print(f"\n{'='*60}")
print("All figures saved to results/figures/")
print(f"{'='*60}")
