"""#9 Sensitivity of k_dom to collocation point count N_int.
Trains PINN on mid_k (gamma=900) with N_int ∈ [2000, 5000, 10000, 20000].
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.pinn import PINN
from physics.schnakenberg import get_parameter_sets

os.makedirs("results/sensitivity", exist_ok=True)

# Import our training/eval utilities from train.py
from train import generate_fixed_points, train_model, evaluate_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

p = get_parameter_sets()["mid_k"]
L, T_train = p["L"], 5.0
N_int_list = [2000, 5000, 10000, 20000]
seeds = [0, 1, 2]  # 3 seeds per N_int

results = {ni: {"k_dom": [], "pde_res": []} for ni in N_int_list}

for N_int in N_int_list:
    for seed in seeds:
        label = f"mid_k_N{N_int}_s{seed}"
        ckpt_file = f"results/sensitivity/{label}.npz"

        # -- Resume: skip completed --
        if os.path.exists(ckpt_file):
            data = np.load(ckpt_file)
            results[N_int]["k_dom"].append(float(data["k_dom"]))
            results[N_int]["pde_res"].append(float(data["pde_res"]))
            print(f"\n  [N_int={N_int}, seed={seed}] SKIP — k_dom={data['k_dom']:.3f}")
            continue

        print(f"\n{'='*50}")
        print(f"N_int={N_int}, seed={seed}")

        torch.manual_seed(seed)
        np.random.seed(seed)

        # Must call generate_fixed_points with the current seed context,
        # but it uses RandomState(2026) internally. Override with our seed.
        import train
        orig_rng = train.generate_fixed_points

        def seeded_gen(L, T, N_int_in, N_ic, N_bc, device="cpu"):
            rng = np.random.RandomState(2026 + seed)
            x_int = L * rng.rand(N_int_in, 1).astype(np.float32)
            t_int = T * rng.rand(N_int_in, 1).astype(np.float32)
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

        pts = seeded_gen(L, T_train, N_int, 500, 400, device)
        model = PINN(L=L).to(device)
        hist = train_model(model, "pinn", p, pts, device=device,
                          adam_epochs=12000, lr=5e-4,
                          bc_weight=10.0, ic_weight=10.0,
                          report_every=2000, label=label)
        ev = evaluate_model(model, p, device=device, T_eval=T_train)

        results[N_int]["k_dom"].append(ev["k_dominant"])
        results[N_int]["pde_res"].append(hist["loss_pde"][-1])

        print(f"  k_dom={ev['k_dominant']:.3f}, PDE={hist['loss_pde'][-1]:.2e}")

        # save per-run checkpoint
        np.savez(ckpt_file, k_dom=ev["k_dominant"], pde_res=hist["loss_pde"][-1])

# ---- Save ----
np.savez("results/sensitivity/sensitivity_Nint.npz",
         N_int_list=N_int_list, results=results)

# ---- Plot ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

for ni in N_int_list:
    kd = results[ni]["k_dom"]
    ax1.errorbar(ni, np.mean(kd), yerr=np.std(kd), fmt='o', capsize=5, ms=8,
                 label=f"N_int={ni}")
ax1.axhline(6.28, color="C3", ls="--", alpha=0.7, label="lowest mode k≈6.28")
ax1.set_xscale("log"); ax1.set_xlabel("N_int"); ax1.set_ylabel("k_dom")
ax1.set_title("Dominant wavenumber vs N_int"); ax1.legend(); ax1.grid(True)

for ni in N_int_list:
    pr = results[ni]["pde_res"]
    ax2.errorbar(ni, np.mean(pr), yerr=np.std(pr), fmt='s', capsize=5, ms=8)
ax2.set_xscale("log"); ax2.set_yscale("log")
ax2.set_xlabel("N_int"); ax2.set_ylabel("Final PDE residual")
ax2.set_title("PDE residual vs N_int"); ax2.grid(True)

plt.suptitle("Collocation Point Sensitivity (γ=900, PINN, 3 seeds)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("results/sensitivity/sensitivity_Nint.png", dpi=200)
plt.savefig("results/sensitivity/sensitivity_Nint.pdf")
plt.close()

print("\nDone. Results in results/sensitivity/")
