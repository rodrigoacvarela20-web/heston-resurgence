from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ftfinance.resurgence import (
    quartic_factor_coefficients,
    quartic_factor_exact,
    quartic_factor_partial_sum,
    quartic_factor_borel_pade,
    quartic_saddle_action,
    large_order_action_estimates,
    borel_pade,
)

FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
FIGURES.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)


def optimal_partial_sum(g, max_order=80):
    coeffs = quartic_factor_coefficients(max_order)
    terms = np.abs(coeffs * g ** np.arange(max_order))
    n_opt = int(np.argmin(terms) + 1)
    return quartic_factor_partial_sum(g, n_opt), n_opt


def main():
    action = quartic_saddle_action()
    coeffs = quartic_factor_coefficients(36)
    estimates = large_order_action_estimates(coeffs)

    g_grid = np.linspace(0.005, 0.25, 70)
    exact = quartic_factor_exact(g_grid)
    optimal = np.empty_like(g_grid)
    n_opt = np.empty_like(g_grid, dtype=int)
    bp = np.empty_like(g_grid)

    for i, g in enumerate(g_grid):
        optimal[i], n_opt[i] = optimal_partial_sum(g)
        bp[i] = quartic_factor_borel_pade(g, n_coeffs=18).real

    rel_opt = np.abs(optimal - exact) / exact
    rel_bp = np.abs(bp - exact) / exact

    np.savetxt(
        RESULTS / "quartic_factor_resurgence.csv",
        np.column_stack([g_grid, exact, optimal, bp, n_opt, rel_opt, rel_bp]),
        delimiter=",",
        header="g,exact,optimal_truncation,borel_pade,n_opt,relerr_optimal,relerr_borel_pade",
        comments="",
    )

    plt.figure(figsize=(7, 4.5))
    plt.plot(g_grid, exact, label="exact nonlinear factor")
    plt.plot(g_grid, optimal, "--", label="optimal perturbative truncation")
    plt.plot(g_grid, bp, ":", label="Borel-Pade")
    plt.xlabel("nonlinearity g")
    plt.ylabel("Z(g)")
    plt.title("Nonlinear stochastic-factor resurgence benchmark")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "quartic_factor_resummation.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.semilogy(g_grid, rel_opt, label="optimal truncation")
    plt.semilogy(g_grid, rel_bp, label="Borel-Pade")
    plt.xlabel("nonlinearity g")
    plt.ylabel("relative error")
    plt.title("Resummation error in nonlinear factor model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "quartic_factor_errors.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.plot(np.arange(1, len(coeffs)), estimates, "o-", ms=3, label=r"$n a_{n-1}/a_n$")
    plt.axhline(action, linestyle="--", label="complex-saddle action")
    plt.xlabel("order n")
    plt.ylabel("action estimate")
    plt.title("Large-order growth detects the complex saddle")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "quartic_factor_large_order.png", dpi=180)
    plt.close()

    num, den = borel_pade(coeffs[:18])
    poles = np.roots(den)
    poles = poles[np.argsort(np.abs(poles - action))]
    np.savetxt(
        RESULTS / "quartic_factor_pade_poles.csv",
        np.column_stack([np.real(poles), np.imag(poles)]),
        delimiter=",",
        header="real,imag",
        comments="",
    )

    print(f"Predicted nearest saddle/Borel action = {action:.8f}")
    print(f"Last large-order estimate = {estimates[-1]:.8f}")
    print(f"Nearest Pade pole = {poles[0].real:.8f} {poles[0].imag:+.8f}i")
    print(f"Max relative error, optimal truncation = {rel_opt.max():.3e}")
    print(f"Max relative error, Borel-Pade = {rel_bp.max():.3e}")


if __name__ == "__main__":
    main()
