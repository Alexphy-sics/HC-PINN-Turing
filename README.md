# HC-PINN Turing Pattern Wavelength Preservation Study

## Overview
Investigation of whether boundary constraint encoding (soft vs hard) affects the spatial pattern fidelity of Physics-Informed Neural Networks (PINNs) in 1D Schnakenberg reaction-diffusion Turing systems. The core question is one of pattern selection versus pattern representation: a PINN may converge to a physically admissible mode that is not the one selected by the Turing instability, and this failure is independent of how boundary conditions are encoded. The spectral monitor introduced in this work is proposed as a transferable diagnostic principle for detecting such mode selection failures in PINN-based reaction-diffusion simulations.

## Key Result
Hard constraints do not improve wavelength preservation. The bottleneck is spectral bias: neural networks preferentially learn low-frequency modes, and neither HC enforcement nor Fourier features reliably capture the correct dominant wavenumber for Turing patterns. A spectral monitor (FFT-based, run every 100 epochs) achieved 100% detection rate (15/15) for mode collapse events with zero false positives (0/10) on Schnakenberg data, and zero false alarms (0/40) on Burgers cross-validation, supporting its use as a transferable diagnostic principle for reaction-diffusion systems.

## Structure
```
HC_PINN_Turing/
├── physics/
│   └── schnakenberg.py           # Model equations, linear stability, dispersion relation
├── numerical/
│   └── fdm_solver.py             # Finite-difference reference solver (Crank-Nicolson)
├── models/
│   └── pinn.py                   # PINN, HC-PINN, FF-PINN architectures
├── experiments/
│   ├── burgers_monitor.py        # Burgers verification: spectral monitor cross-validation
│   ├── spectral_monitor.py       # Spectral monitor class (importable, with self-test demo)
│   ├── fdm_convergence.py        # FDM convergence and grid independence
│   ├── fig0_dispersion.py        # Linear stability dispersion relation plot
│   ├── fig10_fragment.py         # Diagnostic fragment comparison
│   ├── gamma_sweep.py            # Gamma sweep (220, 350, 500, 650, 900)
│   ├── mlp_baseline.py           # Pure data-driven MLP baseline
│   ├── multi_seed.py             # 5-seed statistics for all configurations
│   ├── regenerate_figures.py     # Figure regeneration pipeline
│   ├── sensitivity_Nint.py       # Interior point count sensitivity
│   └── spectral_evolution.py     # Spectral bias evolution during training
├── analysis/
│   ├── ntk_analysis.py           # Neural Tangent Kernel analysis
│   └── spectral_residual.py      # PDE residual spectral decomposition
├── train.py                      # PINN vs HC-PINN training and comparison
├── train_ff.py                   # Fourier Features PINN experiment
├── build_paper.py                # One-click paper build pipeline
├── matplotlibrc.py               # Matplotlib style configuration
└── results/                      # Output figures and data (gitignored; contact author for data)
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

# 5. Spectral monitor demo (self-test with dummy model)
python experiments/spectral_monitor.py --demo

# 6. Burgers verification (20k epochs, ~30 min on GPU)
python experiments/burgers_monitor.py
```

For the full experiment pipeline, see [RUN_GUIDE.md](RUN_GUIDE.md). For a quick smoke test of the Burgers monitor, use `python experiments/burgers_monitor.py --quick`.

## Results Summary

Multi-seed statistics (N=5 seeds, 12,000 Adam epochs with cosine annealing; data from `experiments/multi_seed.py`).

| Model | $\gamma$ | $k^*_\text{theory}$ | $k_\text{dom}$ (mean $\pm$ std) | $\Delta k$ |
|:---|:---:|:---:|:---:|:---:|
| FDM (reference) | 220 | 6.24 | 6.26 | -- |
| FDM (reference) | 900 | 12.61 | 12.52 $\pm$ 0.02 | -- |
| PINN (soft BC) | 220 | 6.24 | 6.28 $\pm$ 0.00 | 0.04 |
| PINN (soft BC) | 900 | 12.61 | 6.28 $\pm$ 0.00 | 6.33 |
| HC-PINN (hard BC) | 220 | 6.24 | 6.28 $\pm$ 0.00 | 0.04 |
| HC-PINN (hard BC) | 900 | 12.61 | 6.28 $\pm$ 0.00 | 6.33 |
| FF-PINN $\sigma=5.0$ | 900 | 12.61 | 6.28 $\pm$ 0.00 | 6.33 |
| FF-PINN $\sigma=12.5$ | 900 | 12.61 | 10.05 $\pm$ 3.08 | 2.56 |
| FF-PINN $\sigma=25.0$ | 900 | 12.61 | 18.85 $\pm$ 8.89 | 6.24 |

At low $\gamma$, all methods recover the correct wavenumber. At high $\gamma$, all neural network architectures collapse to the fundamental mode ($k \approx 6.28$, $n=1$) regardless of boundary constraint encoding, a failure driven by spectral bias rather than boundary treatment. FF-PINN at $\sigma=12.5$ shows partial recovery (3/5 seeds correct) but with high variance, while $\sigma=25.0$ overshoots to spurious high-wavenumber modes.

## Paper

The manuscript *When Hard Constraints Are Not Enough: Spectral Mode Selection Failure in PINN-Based Turing Pattern Simulation* (v25) is currently under review. The `paper/` directory and all figure data are available from the corresponding author on request.
