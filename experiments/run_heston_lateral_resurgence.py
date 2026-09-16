from pathlib import Path
import argparse
import cmath
import sys

import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ftfinance.heston import HestonParams
from ftfinance.heston_resurgence import (
    find_heston_adjacent_sheet_saddle,
    find_heston_tail_saddle,
    fit_heston_effective_stokes_multiplier,
    fit_heston_secondary_singulant,
    heston_adjacent_sheet_fluctuation_coefficients,
    heston_conformal_borel_coefficients,
    heston_secondary_lateral_stokes,
    heston_secondary_log_model_coefficients,
    heston_tail_fluctuation_coefficients,
)

FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
FIGURES.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)


def write_mp_series(path, header, values, digits=120):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(header + "\n")
        for n, value in enumerate(values):
            handle.write(f"{n},{mp.nstr(value, digits)}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse-coefficients", action="store_true", help="reuse the checked-in 75-coefficient files")
    args = parser.parse_args()

    params = HestonParams(mu=0.0, kappa=2.0, theta=0.04, xi=0.45, rho=-0.7, v0=0.04)
    threshold = -0.25
    T = 1.0
    physical = find_heston_tail_saddle(params, threshold, T=T, dps=90)
    adjacent = find_heston_adjacent_sheet_saddle(params, threshold, T=T, dps=110, reference_saddle=physical)
    delta = adjacent["singulant"]

    coeff_path = RESULTS / "heston_fluctuation_coefficients_75.csv"
    conformal_path = RESULTS / "heston_conformal_borel_coefficients_75.csv"
    if args.reuse_coefficients and coeff_path.exists() and conformal_path.exists():
        coeffs_mp = [mp.mpf(v) for v in np.loadtxt(coeff_path, delimiter=",", skiprows=1, dtype=str)[:, 1]]
        conformal_mp = [mp.mpf(v) for v in np.loadtxt(conformal_path, delimiter=",", skiprows=1, dtype=str)[:, 1]]
    else:
        coeffs_mp, _ = heston_tail_fluctuation_coefficients(
            params, threshold, T=T, n_terms=75, dps=250, saddle=physical, return_mpmath=True
        )
        conformal_mp = heston_conformal_borel_coefficients(coeffs_mp, physical.rate, dps=230)
        write_mp_series(coeff_path, "n,c_n", coeffs_mp, digits=150)
        write_mp_series(conformal_path, "n,bw_n", conformal_mp, digits=120)

    conformal = np.asarray([float(v) for v in conformal_mp], dtype=float)
    sector = heston_adjacent_sheet_fluctuation_coefficients(
        params, threshold, T=T, n_terms=10, dps=135, reference_saddle=physical
    )
    d = sector["coefficients"]
    np.savetxt(
        RESULTS / "heston_adjacent_fluctuation_coefficients_10.csv",
        np.column_stack([np.arange(len(d)), d.real, d.imag]),
        delimiter=",",
        header="n,d_n_real,d_n_imag",
        comments="",
    )

    # Direct lateral discontinuity. No Stokes-strength fit is used here.
    orders = (55, 60, 65, 70, 75)
    radii = (0.18, 0.20, 0.225, 0.25, 0.275, 0.30, 0.325, 0.35)
    lateral_rows = []
    chain75 = None
    for n_coeffs in orders:
        for radius in radii:
            result = heston_secondary_lateral_stokes(
                radius,
                conformal_mp,
                physical.rate,
                delta,
                d,
                sector["prefactor_ratio"],
                n_coeffs=n_coeffs,
                dps=115,
            )
            S = result["stokes"]
            lateral_rows.append([n_coeffs, radius, S.real, S.imag, abs(S + 1.0), len(result["chain"])])
            if n_coeffs == 75 and abs(radius - 0.20) < 1e-12:
                chain75 = result["chain"]
    lateral_rows = np.asarray(lateral_rows, dtype=float)
    np.savetxt(
        RESULTS / "heston_lateral_stokes.csv",
        lateral_rows,
        delimiter=",",
        header="n_coeffs,epsilon_radius,stokes_real,stokes_imag,distance_to_minus_one,chain_poles",
        comments="",
    )

    if chain75 is not None:
        np.savetxt(
            RESULTS / "heston_secondary_pade_chain_75.csv",
            np.asarray([[w.real, w.imag, z.real, z.imag, r.real, r.imag] for w, z, r in chain75]),
            delimiter=",",
            header="w_real,w_imag,zeta_real,zeta_imag,residue_real,residue_imag",
            comments="",
        )

    # Independent late-order location update with 75 coefficients.
    location_rows = []
    for start in (45, 50, 55, 58, 60, 62):
        fit = fit_heston_secondary_singulant(conformal, physical.rate, delta, start_order=start, end_order=74)
        zeta = fit["zeta"]
        location_rows.append([start, zeta.real, zeta.imag, abs(zeta - delta), abs(zeta - delta) / abs(delta), fit["weighted_rms"]])
    location_rows = np.asarray(location_rows)
    np.savetxt(
        RESULTS / "heston_secondary_singulant_fit_75.csv",
        location_rows,
        delimiter=",",
        header="start_order,zeta_real,zeta_imag,absolute_gap,relative_gap,weighted_rms",
        comments="",
    )

    # Updated late-order strength fit, retained as a cross-check rather than the primary Stokes determination.
    model, w1 = heston_secondary_log_model_coefficients(d, physical.rate, delta, n_terms=75, fluctuation_order=4, dps=140)
    strength_rows = []
    for start in (50, 55, 60):
        fit = fit_heston_effective_stokes_multiplier(
            conformal, model, sector["prefactor_ratio"], w1, start_order=start, end_order=74
        )
        S = fit["stokes"]
        strength_rows.append([start, S.real, S.imag, fit["nrmse"]])
    strength_rows = np.asarray(strength_rows)
    np.savetxt(
        RESULTS / "heston_stokes_fit_75.csv",
        strength_rows,
        delimiter=",",
        header="start_order,stokes_real,stokes_imag,weighted_nrmse",
        comments="",
    )

    # Hyperasymptotic hierarchy: after extracting the exponential sector and using S=-1,
    # the directly computed jump should reproduce the adjacent fluctuation series itself.
    hyper_rows = []
    for radius in radii:
        result = heston_secondary_lateral_stokes(
            radius, conformal_mp, physical.rate, delta, d, sector["prefactor_ratio"], n_coeffs=75, dps=115
        )
        epsilon = result["epsilon"]
        reduced_jump = -result["discontinuity"] / (sector["prefactor_ratio"] * cmath.exp(-delta / epsilon))
        for order in (0, 1, 2, 4, 9):
            approximation = sum(d[n] * epsilon ** n for n in range(order + 1))
            error = abs(reduced_jump - approximation) / abs(reduced_jump)
            hyper_rows.append([radius, order, reduced_jump.real, reduced_jump.imag, approximation.real, approximation.imag, error])
    hyper_rows = np.asarray(hyper_rows)
    np.savetxt(
        RESULTS / "heston_hyperasymptotic_jump.csv",
        hyper_rows,
        delimiter=",",
        header="epsilon_radius,adjacent_order,reduced_jump_real,reduced_jump_imag,approx_real,approx_imag,relative_error",
        comments="",
    )

    # Figures.
    plt.figure(figsize=(7.2, 4.8))
    for n_coeffs in orders:
        rows = lateral_rows[lateral_rows[:, 0] == n_coeffs]
        plt.plot(rows[:, 1], rows[:, 2], "o-", ms=3, label=f"Re S, N={n_coeffs}")
    plt.axhline(-1.0, linestyle="--", label="S = -1")
    plt.xlabel(r"$|\varepsilon|$ on the secondary Stokes ray")
    plt.ylabel(r"$\mathrm{Re}\,S_{\mathrm{lateral}}$")
    plt.title("Direct lateral-discontinuity Stokes estimate")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(FIGURES / "heston_lateral_stokes.png", dpi=220)
    plt.close()

    plt.figure(figsize=(7.2, 4.8))
    for order in (0, 1, 2, 4, 9):
        rows = hyper_rows[hyper_rows[:, 1] == order]
        plt.semilogy(rows[:, 0], rows[:, 6], "o-", ms=3, label=f"adjacent series through d{order}")
    plt.xlabel(r"$|\varepsilon|$ on the secondary Stokes ray")
    plt.ylabel("relative error")
    plt.title("Hyperasymptotic reconstruction of the lateral jump")
    plt.legend(fontsize=7.5)
    plt.tight_layout()
    plt.savefig(FIGURES / "heston_hyperasymptotic_jump.png", dpi=220)
    plt.close()

    plt.figure(figsize=(7.0, 5.0))
    if chain75 is not None:
        zetas = np.asarray([z for _, z, _ in chain75])
        plt.scatter(zetas.real, zetas.imag, s=42, label="N=75 Pade cut poles")
    plt.scatter([delta.real], [delta.imag], marker="x", s=100, linewidths=2, label="adjacent-sheet action")
    plt.xlabel(r"Re $\zeta$")
    plt.ylabel(r"Im $\zeta$")
    plt.title("Pole condensation used in the lateral jump")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "heston_lateral_pade_chain.png", dpi=220)
    plt.close()

    headline = lateral_rows[(lateral_rows[:, 0] == 75) & (np.isclose(lateral_rows[:, 1], 0.20))][0]
    late = location_rows[location_rows[:, 0] == 62][0]
    fit60 = strength_rows[strength_rows[:, 0] == 60][0]
    summary = np.asarray([[headline[2], headline[3], headline[4], late[1], late[2], late[4], fit60[1], fit60[2], fit60[3]]])
    np.savetxt(
        RESULTS / "heston_lateral_summary.csv",
        summary,
        delimiter=",",
        header="lateral_stokes_real,lateral_stokes_imag,lateral_distance_to_minus_one,latefit62_real,latefit62_imag,latefit62_relative_gap,latefit_stokes_real,latefit_stokes_imag,latefit_stokes_nrmse",
        comments="",
    )

    print("Adjacent singulant:", delta)
    print("Direct lateral Stokes (N=75, |eps|=0.20):", complex(headline[2], headline[3]))
    print("Distance to -1:", headline[4])
    print("75-coefficient late-order singulant (orders 62-74):", complex(late[1], late[2]))
    print("Relative singulant gap:", late[4])
    print("Late-order Stokes cross-check (orders 60-74):", complex(fit60[1], fit60[2]), "NRMSE", fit60[3])


if __name__ == "__main__":
    main()
