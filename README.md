HC-PINN Turing Pattern Wavelength Preservation Study
Overview
Investigation of whether boundary constraint encoding (soft vs hard) affects the spatial pattern fidelity of Physics-Informed Neural Networks (PINNs) in 1D Schnakenberg reaction-diffusion Turing systems.
This repository accompanies the paper submitted to AIP Advances (ADV26-AR-03676).
Key Result
Hard constraints do not improve wavelength preservation. The bottleneck is spectral bias — neural networks preferentially learn low-frequency modes, and neither HC enforcement nor Fourier features reliably capture the correct dominant wavenumber for Turing patterns.
Structure
HC_PINN_Turing/
├── physics/
│   └── schnakenberg.py           # Model equations, linear stability, dispersion relation
├── numerical/
│   └── fdm_solver.py             # Finite-difference reference solver (Crank-Nicolson)
├── models/
│   └── pinn.py                   # PINN, HC-PINN, FF-PINN architectures
├── experiments/
│   ├── fdm_convergence.py        # FDM convergence & grid independence
│   ├── fig0_dispersion.py        # Linear stability dispersion relation plot
│   ├── fig10_fragment.py         # Diagnostic fragment comparison
│   ├── gamma_sweep.py            # γ sweep (220, 350, 500, 650, 900)
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
└── results/                      # Output figures and data (gitignored)
Requirements

Python 3.10+
PyTorch
numpy, scipy, matplotlib, pandas, tqdm

Quick Start
bash# 1. Linear stability analysis
python physics/schnakenberg.py

# 2. FDM reference solution
python numerical/fdm_solver.py

# 3. PINN vs HC-PINN comparison
python train.py

# 4. Fourier Features PINN experiment
python train_ff.py
For the full experiment pipeline (8 steps with dependency ordering), see RUN_GUIDE.md.
Results Summary
Multi-seed statistics (N=5 seeds, 12,000 Adam epochs with cosine annealing; data from experiments/multi_seed.py).
Modelγk*_theoryk_dom (mean ± std)ΔkFDM (reference)2206.246.26—FDM (reference)90012.6112.52 ± 0.02—PINN (soft BC)2206.246.28 ± 0.000.04PINN (soft BC)90012.616.28 ± 0.006.33HC-PINN (hard BC)2206.246.28 ± 0.000.04HC-PINN (hard BC)90012.616.28 ± 0.006.33FF-PINN σ=5.090012.616.28 ± 0.006.33FF-PINN σ=12.590012.6110.05 ± 3.082.56FF-PINN σ=25.090012.6118.85 ± 8.896.24
At low γ, all methods recover the correct wavenumber. At high γ, all neural network architectures collapse to the fundamental mode (k ≈ 6.28, n=1) regardless of boundary constraint encoding — a failure driven by spectral bias rather than boundary treatment. FF-PINN at σ=12.5 shows partial recovery (3/5 seeds correct) but with high variance, while σ=25.0 overshoots to spurious high-wavenumber modes.
