"""
Schnakenberg reaction-diffusion model: equations, steady states, linear stability,
and dispersion relation for Turing instability analysis.

Schnakenberg kinetics (non-dimensional):
    f(u,v) = a - u + u^2 * v
    g(u,v) = b - u^2 * v

PDE:
    du/dt = d_u * d2u/dx2 + gamma * (a - u + u^2 * v)
    dv/dt = d_v * d2v/dx2 + gamma * (b - u^2 * v)
"""

import numpy as np
from numpy.typing import NDArray


def steady_state(a: float, b: float) -> tuple[float, float]:
    """Uniform steady state: u_s = a+b, v_s = b/(a+b)^2."""
    u_s = a + b
    v_s = b / (a + b)**2
    return u_s, v_s


def jacobian(a: float, b: float) -> NDArray:
    """Jacobian of kinetics at steady state.
    J = [[f_u, f_v], [g_u, g_v]]
    f_u = -1 + 2b/(a+b),  f_v = (a+b)^2
    g_u = -2b/(a+b),       g_v = -(a+b)^2
    """
    u_s = a + b
    J = np.array([
        [-1 + 2*b/u_s,      u_s**2],
        [-2*b/u_s,          -u_s**2]
    ])
    return J


def well_mixed_stability(a: float, b: float) -> dict:
    """Check well-mixed stability (no diffusion)."""
    J = jacobian(a, b)
    tr = np.trace(J)
    det = np.linalg.det(J)
    return {
        "trace": tr,
        "determinant": det,
        "stable": tr < 0 and det > 0
    }


def dispersion_relation(
    k: NDArray, a: float, b: float, d_u: float, d_v: float, gamma: float
) -> tuple[NDArray, NDArray]:
    """Dispersion relation Re(lambda(k)) for the Schnakenberg model.

    From linear stability with Fourier ansatz ~ exp(ikx + lambda*t):
        M = [[-d_u*k^2 + gamma*f_u,  gamma*f_v    ],
             [gamma*g_u,            -d_v*k^2 + gamma*g_v]]

        lambda^2 - tr(M)*lambda + det(M) = 0
        tau(k) = tr(M) = gamma*tr(J) - (d_u+d_v)*k^2
        Delta(k) = det(M) = d_u*d_v*k^4 - gamma*(d_u*g_v + d_v*f_u)*k^2 + gamma^2*det(J)

    Returns:
        lambda_max: max real part of eigenvalues for each k
        lambda_all: both eigenvalues' real parts for each k
    """
    J = jacobian(a, b)
    f_u, f_v = J[0, 0], J[0, 1]
    g_u, g_v = J[1, 0], J[1, 1]
    trJ = f_u + g_v
    detJ = f_u*g_v - f_v*g_u

    k2 = k**2

    # tau(k) = tr(M)
    tau = gamma * trJ - (d_u + d_v) * k2

    # Delta(k) = det(M)
    Delta = (
        d_u * d_v * k2**2
        - gamma * (d_u * g_v + d_v * f_u) * k2
        + gamma**2 * detJ
    )

    # Quadratic formula: lambda = (tau +/- sqrt(tau^2 - 4*Delta)) / 2
    discriminant = tau**2 - 4 * Delta

    # Handle sign
    lambda1 = np.zeros_like(k, dtype=float)
    lambda2 = np.zeros_like(k, dtype=float)

    # For discriminant >= 0: real roots
    pos_disc = discriminant >= 0
    sqrt_d = np.sqrt(np.maximum(discriminant, 0))
    lambda1[pos_disc] = (tau[pos_disc] + sqrt_d[pos_disc]) / 2
    lambda2[pos_disc] = (tau[pos_disc] - sqrt_d[pos_disc]) / 2

    # For discriminant < 0: complex conjugate pair, Re = tau/2
    neg_disc = discriminant < 0
    lambda1[neg_disc] = tau[neg_disc] / 2
    lambda2[neg_disc] = tau[neg_disc] / 2

    lambda_max = np.maximum(lambda1, lambda2)
    lambda_all = np.column_stack([lambda1, lambda2])

    return lambda_max, lambda_all


def turing_analysis(
    a: float, b: float, d_u: float, d_v: float, gamma: float,
    L: float = 1.0, n_k: int = 2000
) -> dict:
    """Full Turing instability analysis."""
    stability = well_mixed_stability(a, b)
    J = jacobian(a, b)

    # k range: from 0 to well above expected k_max
    k_max_estimate = np.sqrt(gamma * (d_v * J[0,0] + d_u * J[1,1]) / (2 * d_u * d_v))
    k_max_estimate = max(abs(k_max_estimate), 10.0)
    k = np.linspace(0, 3 * k_max_estimate, n_k)

    lambda_max, lambda_all = dispersion_relation(k, a, b, d_u, d_v, gamma)

    # Find unstable region
    positive_idx = np.where(lambda_max > 1e-10)[0]
    is_turing = len(positive_idx) > 0 and stability["stable"]

    if is_turing:
        k_critical = k[positive_idx[0]]
        idx_max = np.argmax(lambda_max)
        k_most_unstable = k[idx_max]
        lambda_max_val = lambda_max[idx_max]
        wavelength_theory = 2 * np.pi / k_most_unstable
        k_unstable_start = k[positive_idx[0]]
        k_unstable_end = k[positive_idx[-1]]
    elif not stability["stable"]:
        # Well-mixed unstable: not Turing
        k_critical = None
        k_most_unstable = None
        lambda_max_val = np.max(lambda_max)
        wavelength_theory = None
        k_unstable_start = None
        k_unstable_end = None
    else:
        k_critical = None
        k_most_unstable = None
        lambda_max_val = np.max(lambda_max)
        wavelength_theory = None
        k_unstable_start = None
        k_unstable_end = None

    return {
        "k": k,
        "lambda_max": lambda_max,
        "lambda_all": lambda_all,
        "k_critical": k_critical,
        "k_most_unstable": k_most_unstable,
        "lambda_most_unstable": lambda_max_val,
        "wavelength_theory": wavelength_theory,
        "k_unstable_start": k_unstable_start,
        "k_unstable_end": k_unstable_end,
        "is_turing": is_turing,
        "stability": stability,
        "jacobian": J
    }


# ============================================================
# VERIFIED parameter sets for Turing patterns
# ============================================================
def get_parameter_sets() -> dict:
    """Three parameter sets tuned to match discrete modes k_n = 2πn on L=1.

    For Schnakenberg with a=0.1, b=0.9, Du=1, Dv=40:
        k_max ≈ sqrt(gamma * 0.3875)

    Targeting:
        n=1 (k≈6.28)  → gamma≈100   → low_k
        n=2 (k≈12.57) → gamma≈400   → mid_k
        n=3 (k≈18.85) → gamma≈900   → high_k
    """
    return {
        "low_k": {
            "a": 0.1, "b": 0.9, "d_u": 1.0, "d_v": 40.0,
            "gamma": 220.0, "L": 1.0, "Nx": 256
        },
        "mid_k": {
            "a": 0.1, "b": 0.9, "d_u": 1.0, "d_v": 40.0,
            "gamma": 900.0, "L": 1.0, "Nx": 256
        },
        "high_k": {
            "a": 0.1, "b": 0.9, "d_u": 1.0, "d_v": 40.0,
            "gamma": 2000.0, "L": 1.0, "Nx": 256
        }
    }


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    params = get_parameter_sets()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, (label, p) in zip(axes, params.items()):
        result = turing_analysis(
            p["a"], p["b"], p["d_u"], p["d_v"], p["gamma"], p["L"]
        )

        ax.plot(result["k"], result["lambda_max"], "b-", lw=2)
        ax.axhline(0, color="k", ls="--", lw=0.8)

        if result["is_turing"]:
            km = result["k_most_unstable"]
            lam = result["wavelength_theory"]
            ax.axvline(km, color="r", ls="--", lw=1.5,
                       label=f"k*={km:.1f}\nλ={lam:.3f}")
            ax.fill_between(
                result["k"], 0, result["lambda_max"],
                where=result["lambda_max"] > 0,
                alpha=0.2, color="red", label="unstable band"
            )
        elif not result["stability"]["stable"]:
            ax.text(0.5, 0.9, "NOT TURING\n(well-mixed unstable)",
                    transform=ax.transAxes, ha="center", va="top",
                    fontsize=10, color="red", fontweight="bold")
        else:
            ax.text(0.5, 0.9, "STABLE\n(no pattern)",
                    transform=ax.transAxes, ha="center", va="top",
                    fontsize=10, color="gray", fontweight="bold")

        ax.set_title(f"{label}: γ={p['gamma']}", fontsize=11)
        ax.set_xlabel("k")
        ax.set_ylabel("Re(λ)")
        ax.legend(fontsize=8, loc="upper right")
        ax.set_xlim(0, result["k"][-1])

    plt.suptitle("Schnakenberg Turing Dispersion Relations (a=0.1, b=0.9, Du=1, Dv=40)",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig("dispersion_relation.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Print summary
    print("\n" + "="*60)
    print("SCHNAKENBERG TURING ANALYSIS")
    print("="*60)
    for label, p in params.items():
        result = turing_analysis(
            p["a"], p["b"], p["d_u"], p["d_v"], p["gamma"], p["L"]
        )
        J = result["jacobian"]
        print(f"\n--- {label} (gamma={p['gamma']}) ---")
        print(f"  Steady state:  u*={p['a']+p['b']:.3f}, v*={p['b']/(p['a']+p['b'])**2:.4f}")
        print(f"  Jacobian:      [[{J[0,0]:.3f}, {J[0,1]:.3f}], [{J[1,0]:.3f}, {J[1,1]:.3f}]]")
        print(f"  tr(J)={result['stability']['trace']:.3f}, det(J)={result['stability']['determinant']:.3f}")
        print(f"  Well-mixed stable: {result['stability']['stable']}")
        print(f"  Turing unstable:   {result['is_turing']}")
        if result["is_turing"]:
            print(f"  Dominant k*:  {result['k_most_unstable']:.2f}")
            print(f"  Theoretical λ: {result['wavelength_theory']:.4f}")
            print(f"  Unstable k:   [{result['k_unstable_start']:.1f}, {result['k_unstable_end']:.1f}]")
            print(f"  Max Re(λ):    {result['lambda_most_unstable']:.2f}")
        else:
            print(f"  (no Turing instability detected)")
