# HC-PINN Turing Pattern Wavelength Preservation Study

## Overview
Investigation of whether boundary constraint encoding (soft vs hard) affects the spatial pattern fidelity of Physics-Informed Neural Networks (PINNs) in 1D Schnakenberg reaction-diffusion Turing systems.

## Key Result
Hard constraints do **not** improve wavelength preservation. The bottleneck is **spectral bias** — neural networks preferentially learn low-frequency modes, and neither HC enforcement nor Fourier features reliably capture the correct dominant wavenumber for Turing patterns.

## Structure
```
HC_PINN_Turing/
├── physics/
│   └── schnakenberg.py      # Model equations, linear stability, dispersion relation
├── numerical/
│   └── fdm_solver.py        # Finite-difference reference solver
├── models/
│   └── pinn.py              # PINN, HC-PINN, FF-PINN architectures
├── train.py                 # PINN vs HC-PINN training and comparison
├── train_ff.py              # Fourier Features PINN experiment
└── results/                 # Output figures and data (gitignored)
```

## Requirements
- Python 3.10+
- PyTorch
- numpy, scipy, matplotlib, pandas, tqdm

## Quick Start
```bash
# 1. Linear stability analysis
python physics/schnakenberg.py

# 2. FDM reference solution
python numerical/fdm_solver.py

# 3. PINN vs HC-PINN comparison
python train.py

# 4. Fourier Features PINN experiment
python train_ff.py
```

## Results Summary

| Model | low_k (γ=220) | mid_k (γ=900) |
|-------|:---:|:---:|
| FDM k* | 6.26 ✅ | 12.52 ✅ |
| PINN k | 6.28 ✅ | 6.28 ❌ |
| HC-PINN k | 6.28 ✅ | 6.28 ❌ |
| FF-PINN σ=5.0 | — | 6.28 ❌ |
| FF-PINN σ=12.5 | — | 6.28 ❌ |
| FF-PINN σ=25.0 | — | 31.42 ❌ |

At low k, all methods succeed. At high k, all fail — always converging to the fundamental mode (n=1) regardless of BC encoding.
