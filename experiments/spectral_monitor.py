"""Spectral Monitor: online FFT-based dominant-wavenumber tracker for PINN training.

Implements the diagnostic framework described in §2.2 and §4 of the paper:
"Pattern selection or pattern representation — the monitor detects whether the
PINN converges to the physically admissible target mode k* or collapses to a
low-frequency mode (typically k ~ 6.28, the fundamental admissible mode of the
domain)."

Usage:
    python experiments/spectral_monitor.py --demo

The monitor class SpectralMonitor is importable:
    from experiments.spectral_monitor import SpectralMonitor
    monitor = SpectralMonitor(k_target=12.57, delta=1.0, interval=100)
    monitor.report(epoch, pinn_model, x_grid, t_eval)
"""

import numpy as np
from typing import Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO, format="[SpectralMonitor] %(message)s")
logger = logging.getLogger("SpectralMonitor")


class SpectralMonitor:
    """Online diagnostic that tracks the PINN's dominant spatial wavenumber.

    Every `interval` epochs, the monitor:
        1. Evaluates the PINN at t = t_eval on the spatial grid x_grid
        2. Applies a Hanning window and FFT to extract the spatial spectrum
        3. Identifies k_dom = argmax |FFT(hat(u))|
        4. Checks whether k_dom lies within the expected band
           [k_target - delta, k_target + delta]
        5. Logs an alarm if k_dom is outside the band

    The accumulated history is saved as attributes so it can be plotted after
    training completes.
    """

    def __init__(
        self,
        k_target: float,
        delta: float = 1.0,
        interval: int = 100,
        L: float = 1.0,
        Nx: int = 256,
        t_eval: float = 5.0,
        tolerance: float = 1e-8,
    ):
        """
        Args:
            k_target: Expected dominant wavenumber (k* from Turing analysis).
            delta: Half-width of admissible band [k_target-delta, k_target+delta].
            interval: Log every N epochs.
            L: Domain length (physical, not non-dimensionalised).
            Nx: Number of spatial points for FFT evaluation.
            t_eval: Time at which to evaluate the PINN solution for FFT.
            tolerance: Minimum FFT amplitude threshold; below this the spectrum
                       is considered degenerate and no alarm is raised.
        """
        self.k_target = k_target
        self.delta = delta
        self.interval = interval
        self.L = L
        self.Nx = Nx
        self.t_eval = t_eval
        self.tolerance = tolerance

        # Pre-compute FFT grid
        self.x_grid = np.linspace(0, L, Nx)
        self.k_fft = 2 * np.pi * np.fft.fftfreq(Nx, L / Nx)
        self.pos_mask = self.k_fft > 0
        self.k_pos = self.k_fft[self.pos_mask]

        # History buffers
        self.epochs: list[int] = []
        self.k_dom_history: list[float] = []
        self.amplitude_history: list[np.ndarray] = []
        self.alarm_history: list[bool] = []
        self.full_spectra: list[np.ndarray] = []

        # Running stats
        self.n_alarms = 0
        self.n_checks = 0
        self._last_k_dom: Optional[float] = None

    def _compute_spectrum(self, u_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute windowed FFT and return (k_pos, amplitude_pos)."""
        u_dc = u_pred - u_pred.mean()
        window = np.hanning(self.Nx)
        u_windowed = u_dc * window
        fft_full = np.abs(np.fft.fft(u_windowed))
        return self.k_pos, fft_full[self.pos_mask]

    def _extract_k_dom(self, k_pos: np.ndarray, amplitude: np.ndarray) -> float:
        """Return k_dom = argmax |FFT| over positive wavenumbers."""
        if amplitude.max() < self.tolerance:
            return 0.0
        return k_pos[np.argmax(amplitude)]

    def check(
        self, u_pred: np.ndarray, epoch: int, force: bool = False
    ) -> Optional[dict]:
        """Run one monitor check.

        Args:
            u_pred: Predicted u(x, t_eval) -- array of shape (Nx,).
            epoch: Current training epoch.
            force: If True, run even if epoch % interval != 0.

        Returns:
            A dict with keys {'k_dom', 'in_band', 'alarm'} if a check was
            performed, or None if skipped.
        """
        if (epoch % self.interval != 0 or epoch == 0) and not force:
            return None

        k_pos, amplitude = self._compute_spectrum(u_pred)
        k_dom = self._extract_k_dom(k_pos, amplitude)

        in_band = (self.k_target - self.delta <= k_dom <= self.k_target + self.delta)
        alarm = not in_band
        self._last_k_dom = k_dom

        self.epochs.append(epoch)
        self.k_dom_history.append(k_dom)
        self.amplitude_history.append(amplitude)
        self.alarm_history.append(alarm)
        self.full_spectra.append(
            np.interp(self.k_pos, k_pos, amplitude, left=0, right=0)
        )
        self.n_checks += 1
        if alarm:
            self.n_alarms += 1

        status = "in band" if in_band else "OUT OF BAND"
        logger.info(
            f"Epoch {epoch:5d} | k_dom = {k_dom:6.2f} | "
            f"target = {self.k_target:6.2f} +/- {self.delta:.2f} | {status}"
        )

        return {"k_dom": k_dom, "in_band": in_band, "alarm": alarm}

    def report(
        self, epoch: int, pinn_model, x_grid: np.ndarray, t_eval: float
    ) -> Optional[dict]:
        """Convenience wrapper: evaluate a PyTorch model and run monitor.check().

        Args:
            epoch: Current epoch.
            pinn_model: Callable(x, t) returning (u, v) -- any of PINN / HCPINN / FFPINN.
            x_grid: 1D spatial grid (Nx,) for evaluation.
            t_eval: Scalar time for evaluation.

        Returns:
            Same as check().
        """
        import torch

        x_t = torch.tensor(x_grid, dtype=torch.float32).unsqueeze(1)
        t_t = torch.full((len(x_grid), 1), t_eval, dtype=torch.float32)

        with torch.no_grad():
            if hasattr(pinn_model, "device"):
                x_t = x_t.to(pinn_model.device)
                t_t = t_t.to(pinn_model.device)
            u_pred, _ = pinn_model(x_t, t_t)
            u_pred = u_pred.cpu().numpy().flatten()

        return self.check(u_pred, epoch)

    def summary(self) -> dict:
        """Return a snapshot dict of monitor state."""
        return {
            "n_checks": self.n_checks,
            "n_alarms": self.n_alarms,
            "alarm_rate": self.n_alarms / max(self.n_checks, 1),
            "last_k_dom": self._last_k_dom,
            "k_target": self.k_target,
            "delta": self.delta,
            "interval": self.interval,
        }

    def plot_history(self, save_path: Optional[str] = None):
        """Plot k_dom vs epoch with target band."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

        # Panel 1: k_dom trajectory
        ax1.plot(
            self.epochs,
            self.k_dom_history,
            "C0-o",
            markersize=4,
            lw=1.5,
            label=r"$k_{\mathrm{dom}}$",
        )
        ax1.axhline(
            self.k_target,
            color="C2",
            ls="--",
            lw=1.5,
            label=r"$k^*$ (target)",
        )
        ax1.fill_between(
            self.epochs,
            self.k_target - self.delta,
            self.k_target + self.delta,
            alpha=0.1,
            color="C2",
            label=r"$\pm\delta$ band",
        )
        ax1.set_ylabel(r"$k_{\mathrm{dom}}$")
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        # Mark alarms
        alarm_epochs = [e for e, a in zip(self.epochs, self.alarm_history) if a]
        alarm_k = [k for k, a in zip(self.k_dom_history, self.alarm_history) if a]
        ax1.scatter(alarm_epochs, alarm_k, color="C3", marker="x", s=60, zorder=5, label="alarm")

        # Panel 2: spectral heatmap
        spec = np.array(self.full_spectra).T
        vmax = np.percentile(spec, 98) if spec.size > 0 else 1.0
        im = ax2.pcolormesh(
            self.epochs,
            self.k_pos,
            spec,
            shading="auto",
            cmap="inferno",
            vmin=0,
            vmax=vmax,
        )
        ax2.axhline(self.k_target, color="C2", ls="--", lw=1.5)
        ax2.fill_between(
            self.epochs if len(self.epochs) > 1 else [0, 1],
            self.k_target - self.delta,
            self.k_target + self.delta,
            alpha=0.1,
            color="C2",
        )
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("k")
        plt.colorbar(im, ax=ax2, label="|FFT|")

        fig.suptitle(
            f"Spectral Monitor: k*={self.k_target}, d={self.delta}, "
            f"alarms={self.n_alarms}/{self.n_checks}",
            fontsize=13,
            fontweight="bold",
        )
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Plot saved to {save_path}")
        plt.close()


# ============================================================
# Demo / self-test
# ============================================================

def _demo():
    """Run a self-test with a dummy NN that exhibits mode collapse."""
    import torch
    import torch.nn as nn

    class DummyModel(nn.Module):
        """Trivial NN that outputs a sinusoid with controllable k."""
        def __init__(self, L=1.0):
            super().__init__()
            self.L = L
            self.k0 = nn.Parameter(torch.tensor(6.28))
            self.amp = nn.Parameter(torch.tensor(0.5))

        def forward(self, x, t):
            u = self.amp * torch.sin(self.k0 * x) * torch.exp(-0.1 * t)
            v = 0.5 * torch.ones_like(u)
            return u, v

    device = torch.device("cpu")
    L, Nx, t_eval = 1.0, 256, 5.0
    x_grid = np.linspace(0, L, Nx)

    # Test case 1: model converges to target
    print("=" * 60)
    print("Test 1: Model converges to target mode (k=12.57)")
    print("=" * 60)
    model = DummyModel(L=L)
    model.k0.data.fill_(12.57)
    mon = SpectralMonitor(
        k_target=12.57, delta=1.0, interval=50, L=L, Nx=Nx, t_eval=t_eval
    )
    for epoch in range(0, 501, 25):
        mon.report(epoch, model, x_grid, t_eval)
    summary = mon.summary()
    print(f"  Checks={summary['n_checks']}, Alarms={summary['n_alarms']}")
    assert summary["n_alarms"] == 0, "Expected zero alarms for in-band mode"
    print("  PASS\n")

    # Test case 2: model collapses to fundamental mode
    print("=" * 60)
    print("Test 2: Model collapses to fundamental mode (k=6.28)")
    print("=" * 60)
    model2 = DummyModel(L=L)
    model2.k0.data.fill_(6.28)
    mon2 = SpectralMonitor(
        k_target=12.57, delta=1.0, interval=50, L=L, Nx=Nx, t_eval=t_eval
    )
    for epoch in range(0, 501, 25):
        mon2.report(epoch, model2, x_grid, t_eval)
    summary2 = mon2.summary()
    print(f"  Checks={summary2['n_checks']}, Alarms={summary2['n_alarms']}")
    assert summary2["n_alarms"] > 0, "Expected alarms for collapsed mode"
    print("  PASS\n")

    print("All tests passed.")

    # Save demo plots
    mon.plot_history(save_path="results/spectral_monitor_demo_pass.png")
    mon2.plot_history(save_path="results/spectral_monitor_demo_fail.png")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Spectral Monitor for PINN training")
    parser.add_argument("--demo", action="store_true", help="Run self-test demo")
    args = parser.parse_args()

    if args.demo:
        import os
        os.makedirs("results", exist_ok=True)
        _demo()
    else:
        print("Usage: python experiments/spectral_monitor.py --demo")
        print()
        print("To integrate into training, import SpectralMonitor:")
        print("  from experiments.spectral_monitor import SpectralMonitor")
        print()
        print("Example:")
        print("  monitor = SpectralMonitor(k_target=12.57, delta=1.0, interval=100)")
        print("  for epoch in range(1, N_epochs + 1):")
        print("      # ... training step ...")
        print("      monitor.report(epoch, model, x_grid, t_eval)")
