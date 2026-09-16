from pathlib import Path
import sys
import math
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import pade
from scipy.linalg import LinAlgWarning

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ftfinance.hamiltonian import solve_hamiltonian_shooting
from ftfinance.heston import HestonParams
from ftfinance.heston_resurgence import (
    find_heston_tail_saddle,
    heston_borel_poles,
    heston_large_order_extrapolation,
    heston_tail_borel_pade_fluctuation,
    heston_tail_exact,
    heston_tail_fluctuation_coefficients,
    heston_tail_leading_prefactor,
    heston_tail_optimal_truncation,
)

FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
FIGURES.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)


def main():
    warnings.filterwarnings("ignore", category=LinAlgWarning)
    params = HestonParams(mu=0.0, kappa=2.0, theta=0.04, xi=0.45, rho=-0.7, v0=0.04)
    threshold = -0.25
    T = 1.0

    saddle = find_heston_tail_saddle(params, threshold, T=T, dps=90)
    coeffs, _ = heston_tail_fluctuation_coefficients(params, threshold, T=T, n_terms=31, dps=110, saddle=saddle)
    extrapolated, estimates = heston_large_order_extrapolation(coeffs, start_order=12, degree=3)

    hamiltonian = solve_hamiltonian_shooting(params, threshold, T=T, n_steps=800)

    np.savetxt(
        RESULTS / "heston_fluctuation_coefficients.csv",
        np.column_stack([np.arange(len(coeffs)), coeffs]),
        delimiter=",",
        header="n,c_n",
        comments="",
    )

    n = np.arange(1, len(coeffs))
    np.savetxt(
        RESULTS / "heston_large_order.csv",
        np.column_stack([n, estimates]),
        delimiter=",",
        header="n,n_c_nm1_over_c_n",
        comments="",
    )

    eps_grid = np.linspace(0.04, 0.8, 45)
    exact_probability = np.empty_like(eps_grid)
    exact_fluctuation = np.empty_like(eps_grid)
    optimal_probability = np.empty_like(eps_grid)
    borel_probability = np.empty_like(eps_grid)
    optimal_orders = np.empty_like(eps_grid, dtype=int)

    pade_coeffs = coeffs[:24]
    for i, epsilon in enumerate(eps_grid):
        exact_probability[i], exact_fluctuation[i] = heston_tail_exact(epsilon, params, threshold, T=T, saddle=saddle)
        optimal_fluctuation, optimal_orders[i] = heston_tail_optimal_truncation(epsilon, coeffs)
        borel_fluctuation = heston_tail_borel_pade_fluctuation(epsilon, pade_coeffs).real
        leading = heston_tail_leading_prefactor(epsilon, saddle)
        optimal_probability[i] = leading * optimal_fluctuation
        borel_probability[i] = leading * borel_fluctuation

    rel_optimal = np.abs(optimal_probability - exact_probability) / exact_probability
    rel_borel = np.abs(borel_probability - exact_probability) / exact_probability

    np.savetxt(
        RESULTS / "heston_tail_resummation.csv",
        np.column_stack([
            eps_grid,
            exact_probability,
            optimal_probability,
            borel_probability,
            optimal_orders,
            rel_optimal,
            rel_borel,
        ]),
        delimiter=",",
        header="epsilon,exact_tail,optimal_tail,borel_pade_tail,n_opt,relerr_optimal,relerr_borel_pade",
        comments="",
    )

    stability_rows = []
    for n_coeffs in range(12, 27, 2):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", LinAlgWarning)
            poles, _, _ = heston_borel_poles(coeffs[:n_coeffs])
        nearest = poles[np.argmin(np.abs(poles + saddle.rate))]
        stability_rows.append([n_coeffs, nearest.real, nearest.imag, abs(nearest + saddle.rate)])

    np.savetxt(
        RESULTS / "heston_pade_stability.csv",
        np.asarray(stability_rows),
        delimiter=",",
        header="n_coeffs,nearest_pole_real,nearest_pole_imag,distance_to_minus_rate",
        comments="",
    )

    summary = np.array([[
        saddle.p_star,
        saddle.rate,
        saddle.lambda_second,
        hamiltonian["px"][0],
        hamiltonian["action"],
        extrapolated,
        stability_rows[-1][1],
        stability_rows[-1][2],
        rel_borel.max(),
    ]])
    np.savetxt(
        RESULTS / "heston_resurgence_summary.csv",
        summary,
        delimiter=",",
        header="p_star,affine_rate,lambda_second,hamiltonian_px,hamiltonian_action,large_order_extrapolation,nearest_pade_real,nearest_pade_imag,max_relerr_borel_pade",
        comments="",
    )

    plt.figure(figsize=(7, 4.5))
    plt.semilogy(eps_grid, exact_probability, label="exact affine Heston tail")
    plt.semilogy(eps_grid, np.abs(optimal_probability), "--", label="optimal asymptotic truncation")
    plt.semilogy(eps_grid, borel_probability, ":", label="Borel-Pade")
    plt.xlabel(r"noise strength $\epsilon$")
    plt.ylabel(r"$P(X_T \leq x_\star)$")
    plt.title("Heston left-tail resummation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "heston_tail_resummation.png", dpi=190)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.semilogy(eps_grid, rel_optimal, label="optimal asymptotic truncation")
    plt.semilogy(eps_grid, rel_borel, label="Borel-Pade")
    plt.xlabel(r"noise strength $\epsilon$")
    plt.ylabel("relative error")
    plt.title("Heston tail: perturbation versus Borel resummation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "heston_resummation_error.png", dpi=190)
    plt.close()

    fit_n = np.arange(12, len(coeffs), dtype=float)
    fit = np.polyfit(1.0 / fit_n, estimates[11:], 3)
    smooth_n = np.linspace(10.0, 80.0, 400)
    smooth_estimate = np.polyval(fit, 1.0 / smooth_n)

    plt.figure(figsize=(7, 4.5))
    plt.plot(n, estimates, "o", ms=3, label=r"$n c_{n-1}/c_n$")
    plt.plot(smooth_n, smooth_estimate, "-", label=r"cubic extrapolation in $1/n$")
    plt.axhline(-saddle.rate, linestyle="--", label=r"$-I(x_\star)$ from affine saddle")
    plt.xlabel("order n")
    plt.ylabel("Borel-action estimate")
    plt.title("Heston large-order growth recovers the rare-event action")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "heston_large_order_action.png", dpi=190)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    marker_cycle = ["o", "s", "^"]
    for marker, n_coeffs in zip(marker_cycle, (16, 20, 24)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", LinAlgWarning)
            poles, _, _ = heston_borel_poles(coeffs[:n_coeffs])
        visible = poles[(np.abs(poles.real) < 2.0) & (np.abs(poles.imag) < 1.0)]
        plt.scatter(visible.real, visible.imag, marker=marker, s=34, label=f"Pade from {n_coeffs} coeffs")
    plt.axvline(-saddle.rate, linestyle="--", label=r"predicted leading singularity $-I$")
    plt.axhline(0.0, linewidth=0.8)
    plt.xlabel(r"Re $\zeta$")
    plt.ylabel(r"Im $\zeta$")
    plt.title("First Heston Borel plane")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "heston_borel_plane.png", dpi=190)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.plot(hamiltonian["t"], hamiltonian["x"], label=r"$x_\star(t)$")
    plt.plot(hamiltonian["t"], hamiltonian["v"], label=r"$v_\star(t)$")
    plt.xlabel("t")
    plt.ylabel("instanton coordinates")
    plt.title("Continuous Heston instanton from Hamiltonian shooting")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "heston_hamiltonian_instanton.png", dpi=190)
    plt.close()

    nearest = stability_rows[-1]
    print(f"Real affine saddle p* = {saddle.p_star:.12f}")
    print(f"Affine large-deviation rate I = {saddle.rate:.12f}")
    print(f"Hamiltonian shooting p_x = {hamiltonian['px'][0]:.12f}")
    print(f"Hamiltonian action = {hamiltonian['action']:.12f}")
    print(f"Large-order extrapolated Borel location = {extrapolated:.12f}")
    print(f"Predicted leading Borel location = {-saddle.rate:.12f}")
    print(f"Nearest Pade pole ({int(nearest[0])} coeffs) = {nearest[1]:.12f} {nearest[2]:+.3e}i")
    print(f"Max relative Borel-Pade error on epsilon grid = {rel_borel.max():.3e}")
    print(f"Max relative optimal-truncation error on epsilon grid = {rel_optimal.max():.3e}")


if __name__ == "__main__":
    main()
