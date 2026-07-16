"""#4 Multi-seed statistics: 5 independent runs per model configuration.
Reports k_dom mean ± std for all PINN/HC-PINN/FF-PINN combinations.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.pinn import PINN, HCPINN, FFPINN
from physics.schnakenberg import get_parameter_sets
from train import generate_fixed_points, train_model, evaluate_model
from train_ff import train_ffpinn, evaluate as evaluate_ff

os.makedirs("results/multi_seed", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

p_sets = get_parameter_sets()
N_SEEDS = 5
T_train = 5.0

configs = [
    # (model_name, model_class, model_type, gamma_label, extra_kwargs)
    ("PINN_220", PINN, "pinn", "low_k", {}),
    ("PINN_900", PINN, "pinn", "mid_k", {}),
    ("HCPINN_220", HCPINN, "hcpinn", "low_k", {}),
    ("HCPINN_900", HCPINN, "hcpinn", "mid_k", {}),
    ("FFPINN_s5", FFPINN, "ffpinn", "mid_k", {"sigma": 5.0}),
    ("FFPINN_s12.5", FFPINN, "ffpinn", "mid_k", {"sigma": 12.5}),
    ("FFPINN_s25", FFPINN, "ffpinn", "mid_k", {"sigma": 25.0}),
]

all_results = {}

for cfg_name, model_cls, model_type, gamma_label, extra in configs:
    p = p_sets[gamma_label]
    L = p["L"]
    k_doms = []
    pde_residuals = []
    times = []

    for seed in range(N_SEEDS):
        label = f"{cfg_name}_s{seed}"
        ckpt_path = f"results/multi_seed/{label}.pt"

        # -- Resume: skip completed seeds but evaluate saved model --
        if os.path.exists(ckpt_path):
            print(f"\n  [{cfg_name} seed={seed}] checkpoint exists, evaluating saved model...")
            if model_type == "ffpinn":
                model = model_cls(L=L, n_fourier=64, sigma=extra["sigma"]).to(device)
                model.load_state_dict(torch.load(ckpt_path, map_location=device))
                ev = evaluate_ff(model, p, device=device, Te=T_train)
            else:
                model = model_cls(L=L).to(device)
                model.load_state_dict(torch.load(ckpt_path, map_location=device))
                ev = evaluate_model(model, p, device=device, T_eval=T_train)
            k_doms.append(ev["k_dominant"])
            pde_residuals.append(float("nan"))  # no loss history available
            times.append(float("nan"))
            print(f"  k_dom={ev['k_dominant']:.3f} (from saved checkpoint)")
            continue

        print(f"\n{'='*50}")
        print(f"{cfg_name}  seed={seed}")

        torch.manual_seed(seed)
        np.random.seed(seed)

        t0 = time.time()

        if model_type == "ffpinn":
            pts = generate_fixed_points(L, T_train, 8000, 500, 400, device)
            model = model_cls(L=L, n_fourier=64, sigma=extra["sigma"]).to(device)
            hist = train_ffpinn(model, p, pts, device, epochs=12000, lr=5e-4,
                               bc_w=10.0, ic_w=10.0, report_every=2000, label=label)
            ev = evaluate_ff(model, p, device=device, Te=T_train)
        else:
            pts = generate_fixed_points(L, T_train, 8000, 500, 400, device)
            model = model_cls(L=L).to(device)
            hist = train_model(model, model_type, p, pts, device=device,
                              adam_epochs=12000, lr=5e-4,
                              bc_weight=10.0, ic_weight=10.0,
                              report_every=2000, label=label)
            ev = evaluate_model(model, p, device=device, T_eval=T_train)

        elapsed = time.time() - t0
        k_doms.append(ev["k_dominant"])
        # FF-PINN uses "pde" key, PINN/HC-PINN use "loss_pde"
        pde_key = "pde" if model_type == "ffpinn" else "loss_pde"
        pde_residuals.append(hist[pde_key][-1])
        times.append(elapsed)

        print(f"  k_dom={ev['k_dominant']:.3f}, PDE={hist[pde_key][-1]:.2e}, time={elapsed:.0f}s")

        # Save individual model
        torch.save(model.state_dict(), f"results/multi_seed/{label}.pt")

    kd_arr = np.array([x for x in k_doms if not np.isnan(x)])
    pd_arr = np.array([x for x in pde_residuals if not np.isnan(x)])
    all_results[cfg_name] = {
        "k_dom_mean": kd_arr.mean() if len(kd_arr) else float("nan"),
        "k_dom_std": kd_arr.std() if len(kd_arr) else float("nan"),
        "pde_mean": pd_arr.mean() if len(pd_arr) else float("nan"),
        "pde_std": pd_arr.std() if len(pd_arr) else float("nan"),
        "time_mean": np.mean(times) if times else 0,
        "k_doms": kd_arr, "pde_residuals": pd_arr
    }

    print(f"\n--- {cfg_name}: k_dom = {kd_arr.mean():.3f} ± {kd_arr.std():.3f} ---")

# ---- Save ----
np.savez("results/multi_seed/multi_seed_results.npz", results=all_results)

# ---- Summary table ----
print(f"\n{'='*70}")
print(f"SUMMARY: k_dom mean ± std (N={N_SEEDS})")
print(f"{'='*70}")
print(f"{'Model':<20s} {'γ':>6s} {'k_dom':>18s} {'PDE residual':>18s} {'Time':>8s}")
print("-" * 70)
for cfg_name, r in all_results.items():
    parts = cfg_name.split("_")
    model_part = parts[0]
    if model_part.startswith("FF"):
        gamma_str = "900"
        model_str = f"FF-PINN σ={extra.get('sigma','?')}"
    else:
        gamma_str = parts[1] if len(parts) > 1 else "?"
        model_str = model_part
    print(f"{cfg_name:<20s} {gamma_str:>6s} {r['k_dom_mean']:>8.3f}±{r['k_dom_std']:.3f}  "
          f"{r['pde_mean']:>8.2e}±{r['pde_std']:.2e}  {r['time_mean']:>6.0f}s")

# Save CSV for Table 1
with open("results/multi_seed/summary.csv", "w") as f:
    f.write("model,gamma,k_dom_mean,k_dom_std,pde_mean,pde_std,time_mean\n")
    for cfg_name, r in all_results.items():
        f.write(f"{cfg_name},{gamma_label},{r['k_dom_mean']:.4f},{r['k_dom_std']:.4f},"
                f"{r['pde_mean']:.4e},{r['pde_std']:.4e},{r['time_mean']:.0f}\n")

print(f"\nResults saved to results/multi_seed/")
