from pathlib import Path
import sys
import warnings

import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np
from scipy.linalg import LinAlgWarning

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ftfinance.heston import HestonParams
from ftfinance.heston_resurgence import (
    find_heston_adjacent_sheet_saddle,
    find_heston_tail_saddle,
    heston_conformal_borel_coefficients,
    heston_conformal_pade_poles,
    heston_negative_moment_pole,
    heston_tail_fluctuation_coefficients,
)

FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
FIGURES.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)


def select_secondary_pair(zeta_poles, target):
    candidates = [z for z in zeta_poles if z.imag > 0 and 1.5 < z.real < 5.0 and 1.0 < z.imag < 4.0]
    if not candidates:
        raise RuntimeError("no secondary conjugate-pole candidate found")
    return min(candidates, key=lambda z: abs(z - target))


def main():
    warnings.filterwarnings("ignore", category=LinAlgWarning)
    params = HestonParams(mu=0.0, kappa=2.0, theta=0.04, xi=0.45, rho=-0.7, v0=0.04)
    threshold = -0.25
    T = 1.0

    real_saddle = find_heston_tail_saddle(params, threshold, T=T, dps=90)
    moment_pole = heston_negative_moment_pole(params, T=T, bracket=(-6.7, -6.2))
    adjacent = find_heston_adjacent_sheet_saddle(params, threshold, T=T, dps=100, reference_saddle=real_saddle)
    target = adjacent["singulant"]

    coeffs_mp, _ = heston_tail_fluctuation_coefficients(
        params,
        threshold,
        T=T,
        n_terms=45,
        dps=145,
        saddle=real_saddle,
        return_mpmath=True,
    )
    conformal = heston_conformal_borel_coefficients(coeffs_mp, real_saddle.rate, dps=110)

    with open(RESULTS / "heston_conformal_borel_coefficients_45.csv", "w", encoding="utf-8") as handle:
        handle.write("n,bw_n\n")
        for n, value in enumerate(conformal):
            handle.write(f"{n},{mp.nstr(value, 60)}\n")

    rows = []
    for n_coeffs in range(24, 46):
        w_poles, zeta_poles = heston_conformal_pade_poles(conformal, real_saddle.rate, n_coeffs=n_coeffs)
        pole = select_secondary_pair(zeta_poles, target)
        w = w_poles[np.argmin(np.abs(zeta_poles - pole))]
        rows.append([n_coeffs, w.real, w.imag, pole.real, pole.imag, abs(pole - target)])

    rows = np.asarray(rows)
    np.savetxt(
        RESULTS / "heston_secondary_conformal_pade.csv",
        rows,
        delimiter=",",
        header="n_coeffs,w_real,w_imag,zeta_real,zeta_imag,distance_to_adjacent_sheet_action",
        comments="",
    )

    last = rows[-1]
    relative_gap = last[5] / abs(target)
    monodromy = 2.0 * np.pi * params.kappa * params.theta / (params.xi**2)
    summary = np.array([[moment_pole, adjacent["p"].real, target.real, target.imag, monodromy, last[3], last[4], last[5], relative_gap]])
    np.savetxt(
        RESULTS / "heston_secondary_summary.csv",
        summary,
        delimiter=",",
        header="first_negative_moment_pole,adjacent_sheet_p,adjacent_action_real,adjacent_action_imag,2pi_kappatheta_over_xi2,pade_real_45,pade_imag_45,absolute_gap,relative_gap",
        comments="",
    )

    order_subset = [28, 32, 36, 40, 45]
    plt.figure(figsize=(7.2, 5.0))
    for n_coeffs in order_subset:
        row = rows[rows[:, 0] == n_coeffs][0]
        plt.scatter(row[3], row[4], s=42, label=f"conformal Pade, N={n_coeffs}")
        plt.scatter(row[3], -row[4], s=42)
    plt.scatter(target.real, target.imag, marker="x", s=90, linewidths=2.0, label="adjacent-sheet action")
    plt.scatter(target.real, -target.imag, marker="x", s=90, linewidths=2.0)
    plt.scatter(-real_saddle.rate, 0.0, marker="*", s=110, label=r"leading obstruction $-I$")
    plt.axhline(0.0, linewidth=0.8)
    plt.xlabel(r"Re $\zeta$")
    plt.ylabel(r"Im $\zeta$")
    plt.title("Heston secondary Borel-plane search")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "heston_secondary_borel_search.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7.2, 4.7))
    plt.plot(rows[:, 0], rows[:, 3], "o-", ms=3.5, label=r"Re $\zeta_{\rm Pade}^{(2)}$")
    plt.plot(rows[:, 0], rows[:, 4], "s-", ms=3.5, label=r"Im $\zeta_{\rm Pade}^{(2)}$")
    plt.axhline(target.real, linestyle="--", label=r"Re $\Delta S_1$")
    plt.axhline(target.imag, linestyle=":", label=r"Im $\Delta S_1$")
    plt.xlabel("number of perturbative coefficients")
    plt.ylabel("secondary singularity coordinate")
    plt.title("Stability of the secondary conformal-Pade pair")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "heston_secondary_stability.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7.2, 3.5))
    plt.axhline(0.0, linewidth=0.8)
    plt.scatter([real_saddle.p_star], [0], marker="o", s=70, label=r"physical saddle $p_\star$")
    plt.scatter([moment_pole], [0], marker="|", s=250, linewidths=2.5, label="first moment pole")
    plt.scatter([adjacent["p"].real], [0], marker="x", s=80, linewidths=2.0, label="adjacent-sheet stationary point")
    plt.xlim(adjacent["p"].real - 1.2, -1.5)
    plt.yticks([])
    plt.xlabel("real Laplace variable p")
    plt.title("Analytic continuation along the negative Heston moment direction")
    plt.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    plt.savefig(FIGURES / "heston_adjacent_sheet_structure.png", dpi=200)
    plt.close()

    print(f"Physical saddle p* = {real_saddle.p_star:.12f}")
    print(f"Leading rate I = {real_saddle.rate:.12f}")
    print(f"First negative moment pole = {moment_pole:.12f}")
    print(f"Adjacent-sheet stationary point p1 = {adjacent['p'].real:.12f}")
    print(f"Adjacent-sheet action DeltaS1 = {target.real:.12f} {target.imag:+.12f}i")
    print(f"Log monodromy scale 2*pi*kappa*theta/xi^2 = {monodromy:.12f}")
    print(f"N=45 conformal-Pade pair = {last[3]:.12f} {last[4]:+.12f}i")
    print(f"Relative distance to adjacent-sheet action = {relative_gap:.3%}")


if __name__ == "__main__":
    main()
