# ============================================================
# Fig 10: Spectral residual decomposition
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

ax1.semilogy(kv_sr[km], u_fft_sr[km], "C0-", lw=1.5, label="|FFT[û]|")
ax1.axvline(kv_sr[np.argmax(u_fft_sr)], color="C0", ls="--", lw=1.5,
            label=f"k_dom={kv_sr[np.argmax(u_fft_sr)]:.2f}")
ax1.axvline(12.57, color="C2", ls=":", lw=1.5, label="k* = 12.57")
ax1.set_xlabel("k");
ax1.set_ylabel("|FFT(u(x, t = 5))|")
ax1.set_title("Solution spectrum")
ax1.legend(fontsize=8, frameon=False); ax1.grid(True, alpha=0.3)

ax2.semilogy(kv_sr[km], R_fft_sr[km], "C3-", lw=1.5, label="|FFT[R]|")
ax2.axvline(12.57, color="k", ls="--", lw=1.5, alpha=0.5, label="k* = 12.57")
y_lo, y_hi = ax2.get_ylim()
ax2.fill_between([0, 12.57], float(y_lo), float(y_hi), alpha=0.15, color="C0",
                 label=f"k < k*: {low_frac:.0f}%")
ax2.fill_between([12.57, float(k_max)], float(y_lo), float(y_hi), alpha=0.15, color="C3",
                 label=f"k >= k*: {high_frac:.0f}%")
ax2.set_ylim(float(y_lo), float(y_hi))
ax2.set_xlabel("k");
ax2.set_ylabel("|FFT(PDE residual)|")
ax2.set_title("PDE residual spectrum")
ax2.legend(fontsize=8, frameon=False); ax2.grid(True, alpha=0.3)

fig.suptitle("Fig. 10: Spectral residual decomposition — PINN mid_k (γ=900)",
             fontsize=14, fontweight="bold")
save_both(fig, "fig10_spectral_residual")

print(f"\n{'='*60}")
print("All figures saved to results/figures/")
print(f"{'='*60}")