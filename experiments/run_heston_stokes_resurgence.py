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
    fit_heston_effective_stokes_multiplier,
    fit_heston_secondary_singulant,
    heston_adjacent_sheet_fluctuation_coefficients,
    heston_conformal_borel_coefficients,
    heston_conformal_pade_poles,
    heston_tail_fluctuation_coefficients,
    heston_secondary_log_model_coefficients,
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

    physical = find_heston_tail_saddle(params, threshold, T=T, dps=90)
    adjacent = find_heston_adjacent_sheet_saddle(params, threshold, T=T, dps=110, reference_saddle=physical)
    target = adjacent["singulant"]

    coeffs_mp, _ = heston_tail_fluctuation_coefficients(
        params,
        threshold,
        T=T,
        n_terms=60,
        dps=205,
        saddle=physical,
        return_mpmath=True,
    )
    conformal_mp = heston_conformal_borel_coefficients(coeffs_mp, physical.rate, dps=190)
    conformal = np.asarray([float(v) for v in conformal_mp], dtype=float)

    with open(RESULTS / "heston_fluctuation_coefficients_60.csv", "w", encoding="utf-8") as handle:
        handle.write("n,c_n\n")
        for n, value in enumerate(coeffs_mp):
            handle.write(f"{n},{mp.nstr(value, 120)}\n")

    with open(RESULTS / "heston_conformal_borel_coefficients_60.csv", "w", encoding="utf-8") as handle:
        handle.write("n,bw_n\n")
        for n, value in enumerate(conformal_mp):
            handle.write(f"{n},{mp.nstr(value, 100)}\n")

    adjacent_sector = heston_adjacent_sheet_fluctuation_coefficients(
        params,
        threshold,
        T=T,
        n_terms=6,
        dps=125,
        reference_saddle=physical,
    )
    d = adjacent_sector["coefficients"]
    np.savetxt(
        RESULTS / "heston_adjacent_fluctuation_coefficients.csv",
        np.column_stack([np.arange(len(d)), d.real, d.imag]),
        delimiter=",",
        header="n,d_n_real,d_n_imag",
        comments="",
    )

    # Secondary conformal-Pade stability with the extended 60-coefficient set.
    pade_rows = []
    for n_coeffs in range(45, 61):
        w_poles, zeta_poles = heston_conformal_pade_poles(conformal_mp, physical.rate, n_coeffs=n_coeffs, scale=0.55)
        pole = select_secondary_pair(zeta_poles, target)
        w = w_poles[np.argmin(np.abs(zeta_poles - pole))]
        pade_rows.append([n_coeffs, w.real, w.imag, pole.real, pole.imag, abs(pole - target), abs(pole - target) / abs(target)])
    pade_rows = np.asarray(pade_rows)
    np.savetxt(
        RESULTS / "heston_secondary_conformal_pade_60.csv",
        pade_rows,
        delimiter=",",
        header="n_coeffs,w_real,w_imag,zeta_real,zeta_imag,absolute_gap,relative_gap",
        comments="",
    )

    # Direct late-order location inference from a conjugate logarithmic pair.
    location_rows = []
    for start in (30, 35, 40, 42, 45, 48, 50):
        fit = fit_heston_secondary_singulant(conformal, physical.rate, target, start_order=start, end_order=59)
        z = fit["zeta"]
        w = fit["w"]
        location_rows.append([start, w.real, w.imag, z.real, z.imag, abs(z - target), abs(z - target) / abs(target), fit["weighted_rms"]])
    location_rows = np.asarray(location_rows)
    np.savetxt(
        RESULTS / "heston_secondary_singulant_fit.csv",
        location_rows,
        delimiter=",",
        header="start_order,w_real,w_imag,zeta_real,zeta_imag,absolute_gap,relative_gap,weighted_rms",
        comments="",
    )

    # Stokes-strength fits with increasingly many adjacent-sector fluctuations.
    stokes_rows = []
    models = {}
    w1 = None
    for fluct_order in range(4):
        model, w1_model = heston_secondary_log_model_coefficients(
            d,
            physical.rate,
            target,
            n_terms=60,
            fluctuation_order=fluct_order,
            dps=120,
        )
        models[fluct_order] = model
        w1 = w1_model
        for start in (30, 35, 40, 45, 50):
            fit = fit_heston_effective_stokes_multiplier(
                conformal,
                model,
                adjacent_sector["prefactor_ratio"],
                w1,
                start_order=start,
                end_order=59,
                asymptotic_weighting=True,
            )
            S = fit["stokes"]
            K = fit["K"]
            stokes_rows.append([fluct_order, start, K.real, K.imag, S.real, S.imag, fit["nrmse"]])
    stokes_rows = np.asarray(stokes_rows)
    np.savetxt(
        RESULTS / "heston_stokes_fit.csv",
        stokes_rows,
        delimiter=",",
        header="fluctuation_order,start_order,K_real,K_imag,S_real,S_imag,weighted_nrmse",
        comments="",
    )

    # Out-of-sample strength/phase test: fit orders 40--49, predict 50--59.
    model = models[2]
    train = np.arange(40, 50)
    test = np.arange(50, 60)
    train_scale = train * (abs(w1) ** train)
    train_design = np.column_stack([2 * model[train].real * train_scale, -2 * model[train].imag * train_scale])
    train_target = conformal[train] * train_scale
    solution, *_ = np.linalg.lstsq(train_design, train_target, rcond=None)
    K_holdout = complex(solution[0], solution[1])
    S_holdout = 2j * np.pi * K_holdout / adjacent_sector["prefactor_ratio"]

    all_orders = np.arange(35, 60)
    all_scale = all_orders * (abs(w1) ** all_orders)
    actual_normalized = conformal[all_orders] * all_scale
    predicted = 2 * np.real(K_holdout * model[all_orders])
    predicted_normalized = predicted * all_scale
    is_holdout = (all_orders >= 50).astype(int)
    np.savetxt(
        RESULTS / "heston_secondary_strength_holdout.csv",
        np.column_stack([all_orders, actual_normalized, predicted_normalized, is_holdout]),
        delimiter=",",
        header="n,normalized_actual,normalized_adjacent_sector_model,is_holdout",
        comments="",
    )
    test_actual = actual_normalized[all_orders >= 50]
    test_pred = predicted_normalized[all_orders >= 50]
    holdout_nrmse = np.linalg.norm(test_pred - test_actual) / np.linalg.norm(test_actual)

    # Figures.
    plt.figure(figsize=(7.2, 5.0))
    for n_coeffs in (45, 48, 52, 56, 60):
        row = pade_rows[pade_rows[:, 0] == n_coeffs][0]
        plt.scatter(row[3], row[4], s=40, label=f"conformal Pade, N={n_coeffs}")
    plt.scatter(target.real, target.imag, marker="x", s=95, linewidths=2.0, label="adjacent-sheet action")
    plt.scatter(target.real, -target.imag, marker="x", s=95, linewidths=2.0)
    plt.axhline(0.0, linewidth=0.8)
    plt.xlabel(r"Re $\zeta$")
    plt.ylabel(r"Im $\zeta$")
    plt.title("Extended Heston secondary Borel-plane search")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "heston_secondary_borel_search.png", dpi=210)
    plt.close()

    plt.figure(figsize=(7.2, 4.8))
    plt.plot(pade_rows[:, 0], pade_rows[:, 3], "o-", ms=3.5, label=r"Re $\zeta_{\rm Pade}^{(2)}$")
    plt.plot(pade_rows[:, 0], pade_rows[:, 4], "s-", ms=3.5, label=r"Im $\zeta_{\rm Pade}^{(2)}$")
    plt.axhline(target.real, linestyle="--", label=r"Re $\Delta S_1$")
    plt.axhline(target.imag, linestyle=":", label=r"Im $\Delta S_1$")
    plt.xlabel("number of perturbative coefficients")
    plt.ylabel("secondary singularity coordinate")
    plt.title("Secondary conformal-Pade stability through 60 coefficients")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "heston_secondary_stability.png", dpi=210)
    plt.close()

    plt.figure(figsize=(7.2, 4.8))
    plt.plot(location_rows[:, 0], location_rows[:, 3], "o-", label=r"Re $\zeta_{\rm late}$")
    plt.plot(location_rows[:, 0], location_rows[:, 4], "s-", label=r"Im $\zeta_{\rm late}$")
    plt.axhline(target.real, linestyle="--", label=r"Re $\Delta S_1$")
    plt.axhline(target.imag, linestyle=":", label=r"Im $\Delta S_1$")
    plt.xlabel("first order used in late-order fit")
    plt.ylabel("inferred singularity coordinate")
    plt.title("Direct late-order inference of the secondary singulant")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "heston_secondary_singulant_fit.png", dpi=210)
    plt.close()

    plt.figure(figsize=(7.2, 4.8))
    for fluct_order, marker in ((0, "o"), (2, "s")):
        rows = stokes_rows[stokes_rows[:, 0] == fluct_order]
        plt.plot(rows[:, 1], rows[:, 4], marker + "-", label=fr"Re $S_{{\rm eff}}$, fluct. order {fluct_order}")
        plt.plot(rows[:, 1], rows[:, 5], marker + "--", label=fr"Im $S_{{\rm eff}}$, fluct. order {fluct_order}")
    plt.axhline(-1.0, linestyle=":", label=r"$-1$")
    plt.axhline(0.0, linewidth=0.8)
    plt.xlabel("first perturbative order in fit window")
    plt.ylabel(r"effective Stokes multiplier $S_{\rm eff}$")
    plt.title("Adjacent-sector strength from late conformal-Borel coefficients")
    plt.legend(fontsize=7.5)
    plt.tight_layout()
    plt.savefig(FIGURES / "heston_stokes_multiplier.png", dpi=210)
    plt.close()

    plt.figure(figsize=(7.2, 4.8))
    plt.plot(all_orders, actual_normalized, "o-", ms=3.5, label="actual conformal-Borel coefficients")
    plt.plot(all_orders, predicted_normalized, "--", label="adjacent-sector prediction")
    plt.axvline(49.5, linestyle=":", label="fit / holdout boundary")
    plt.xlabel("order n")
    plt.ylabel(r"$n |w_1|^n b_n$")
    plt.title("Out-of-sample phase and strength test")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "heston_secondary_coefficient_prediction.png", dpi=210)
    plt.close()

    headline_stokes = stokes_rows[(stokes_rows[:, 0] == 2) & (stokes_rows[:, 1] == 40)][0]
    late_fit = location_rows[location_rows[:, 0] == 45][0]
    last_pade = pade_rows[-1]

    summary = np.array([[
        target.real,
        target.imag,
        last_pade[3],
        last_pade[4],
        last_pade[6],
        late_fit[3],
        late_fit[4],
        late_fit[6],
        headline_stokes[4],
        headline_stokes[5],
        headline_stokes[6],
        S_holdout.real,
        S_holdout.imag,
        holdout_nrmse,
    ]])
    np.savetxt(
        RESULTS / "heston_stokes_summary.csv",
        summary,
        delimiter=",",
        header="adjacent_action_real,adjacent_action_imag,pade60_real,pade60_imag,pade60_relative_gap,latefit45_real,latefit45_imag,latefit45_relative_gap,stokes40_59_real,stokes40_59_imag,stokes40_59_weighted_nrmse,holdout_stokes_real,holdout_stokes_imag,holdout_nrmse",
        comments="",
    )

    print(f"Adjacent action = {target.real:.12f} {target.imag:+.12f}i")
    print(f"Pade N=60 = {last_pade[3]:.12f} {last_pade[4]:+.12f}i; relative gap = {last_pade[6]:.3%}")
    print(f"Late-order fit (45--59) = {late_fit[3]:.12f} {late_fit[4]:+.12f}i; relative gap = {late_fit[6]:.3%}")
    print(f"Adjacent fluctuations d0..d3 = {[float(v.real) for v in d[:4]]}")
    print(f"Effective Stokes S (orders 40--59, d0..d2) = {headline_stokes[4]:.9f} {headline_stokes[5]:+.9f}i")
    print(f"Weighted in-window NRMSE = {headline_stokes[6]:.3%}")
    print(f"Holdout fit S (40--49) = {S_holdout.real:.9f} {S_holdout.imag:+.9f}i")
    print(f"Holdout prediction NRMSE (50--59) = {holdout_nrmse:.3%}")


if __name__ == "__main__":
    main()
