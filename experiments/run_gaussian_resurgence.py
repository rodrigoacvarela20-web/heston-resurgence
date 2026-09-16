from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ftfinance.resurgence import (
    GaussianTailModel,
    gaussian_tail_exact,
    gaussian_tail_asymptotic,
    gaussian_tail_borel_pade,
    gaussian_tail_fluctuation_coefficients,
    gaussian_borel_exact,
    borel_pade,
    large_order_action_estimates,
    optimal_truncation_order,
)

FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
FIGURES.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)


def main():
    model = GaussianTailModel(mean_return=0.0, sigma=0.2, T=1.0, threshold=-0.3)
    action = model.instanton_action
    eps_grid = np.linspace(0.08, 0.8, 50)

    exact = gaussian_tail_exact(eps_grid, model)
    asym = np.array([
        gaussian_tail_asymptotic(eps, model, optimal_truncation_order(eps, action, max_order=80))
        for eps in eps_grid
    ])
    bp = np.array([gaussian_tail_borel_pade(eps, model, n_coeffs=18).real for eps in eps_grid])

    rel_asym = np.abs(asym - exact) / exact
    rel_bp = np.abs(bp - exact) / exact

    np.savetxt(
        RESULTS / "gaussian_resurgence_benchmark.csv",
        np.column_stack([eps_grid, exact, asym, bp, rel_asym, rel_bp]),
        delimiter=",",
        header="epsilon,exact,optimal_truncation,borel_pade,relerr_optimal,relerr_borel_pade",
        comments="",
    )

    plt.figure(figsize=(7, 4.5))
    plt.semilogy(eps_grid, exact, label="exact Gaussian tail")
    plt.semilogy(eps_grid, asym, "--", label="optimally truncated instanton series")
    plt.semilogy(eps_grid, bp, ":", label="Borel-Pade")
    plt.xlabel(r"noise strength $\epsilon$")
    plt.ylabel(r"$P(X_T \leq x_\star)$")
    plt.title("Exactly solvable financial-tail resurgence benchmark")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "gaussian_tail_resummation.png", dpi=180)
    plt.close()

    coeffs = gaussian_tail_fluctuation_coefficients(30, action)
    estimates = large_order_action_estimates(coeffs)
    np.savetxt(
        RESULTS / "gaussian_large_order.csv",
        np.column_stack([np.arange(1, len(coeffs)), estimates]),
        delimiter=",",
        header="n,action_estimate",
        comments="",
    )

    plt.figure(figsize=(7, 4.5))
    plt.plot(np.arange(1, len(coeffs)), estimates, "o-", ms=3, label=r"$n c_{n-1}/c_n$")
    plt.axhline(-action, linestyle="--", label=r"exact nearest Borel singularity $-A$")
    plt.xlabel("order n")
    plt.ylabel("large-order action estimate")
    plt.title("Large-order coefficients recover the Borel singularity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "gaussian_large_order_action.png", dpi=180)
    plt.close()

    pade_num, pade_den = borel_pade(coeffs[:18])
    zeta = np.linspace(-1.8 * action, 2.5 * action, 1000)
    mask = np.abs(zeta + action) > 0.02 * action
    exact_borel = gaussian_borel_exact(zeta[mask] + 0j, action)
    pade_borel = np.polyval(pade_num, zeta[mask]) / np.polyval(pade_den, zeta[mask])

    plt.figure(figsize=(7, 4.5))
    plt.plot(zeta[mask], np.real(exact_borel), label="exact Borel transform")
    plt.plot(zeta[mask], np.real(pade_borel), "--", label="Pade continuation")
    plt.axvline(-action, linestyle=":", label=r"branch point $\zeta=-A$")
    plt.ylim(-1, 7)
    plt.xlabel(r"Borel coordinate $\zeta$")
    plt.ylabel(r"Re $\mathcal{B}(\zeta)$")
    plt.title("Borel-plane singularity in the solvable benchmark")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "gaussian_borel_plane.png", dpi=180)
    plt.close()

    print(f"Instanton action A = {action:.8f}")
    print(f"Predicted Borel singularity = {-action:.8f}")
    print(f"Last ratio estimate = {estimates[-1]:.8f}")
    print(f"Max relative error, optimal truncation = {rel_asym.max():.3e}")
    print(f"Max relative error, Borel-Pade = {rel_bp.max():.3e}")


if __name__ == "__main__":
    main()