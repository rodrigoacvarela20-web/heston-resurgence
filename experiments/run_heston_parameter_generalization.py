from pathlib import Path
import argparse
import math
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ftfinance.heston import HestonParams
from ftfinance.heston_resurgence import (
    find_heston_first_adjacent_sheet_saddle,
    find_heston_tail_saddle,
    fit_heston_secondary_singulant,
    heston_adjacent_sheet_fluctuation_coefficients,
    heston_conformal_borel_coefficients,
    heston_picard_lefschetz_scan,
    heston_secondary_lateral_stokes,
    heston_secondary_pade_chain,
    heston_tail_fluctuation_coefficients,
)

RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

BASE = dict(mu=0.0, kappa=2.0, theta=0.04, xi=0.45, rho=-0.7, v0=0.04)
BASE_THRESHOLD = -0.25
BASE_T = 1.0
AXES = {
    "xstar": [-0.15, -0.20, -0.25, -0.30, -0.35],
    "T": [0.50, 0.75, 1.00, 1.50, 2.00],
    "rho": [-0.90, -0.70, -0.50, -0.30, 0.00],
    "xi": [0.25, 0.35, 0.45, 0.60, 0.80],
    "kappa": [0.75, 1.25, 2.00, 3.00, 4.00],
}


def make_case(axis, value):
    params_dict = dict(BASE)
    threshold = BASE_THRESHOLD
    T = BASE_T
    if axis == "xstar":
        threshold = float(value)
    elif axis == "T":
        T = float(value)
    else:
        params_dict[axis] = float(value)
    return HestonParams(**params_dict), threshold, T


def structural_case(axis, value):
    params, threshold, T = make_case(axis, value)
    physical = find_heston_tail_saddle(params, threshold, T=T, dps=65)
    adjacent = find_heston_first_adjacent_sheet_saddle(
        params, threshold, T=T, dps=80, reference_saddle=physical, p_min=-100.0
    )
    local = heston_adjacent_sheet_fluctuation_coefficients(
        params, threshold, T=T, n_terms=2, dps=90, reference_saddle=physical, adjacent=adjacent
    )
    monodromy = 2.0 * math.pi * params.kappa * params.theta / (params.xi ** 2)
    # Slow local Hessians require longer flow time to leave the saddle neighborhood.
    tau_max = min(700.0, max(90.0, 24.0 / max(abs(local["lambda_second"]), 0.02)))
    scan = heston_picard_lefschetz_scan(
        params,
        threshold,
        T=T,
        contour_real=-1.0,
        angle_offsets=(-0.02, 0.0, 0.02),
        tau_max=tau_max,
        reference_saddle=physical,
        adjacent=adjacent,
        adjacent_local=local,
    )
    by_offset = {round(r["offset"], 5): r for r in scan["rows"]}
    below = by_offset[-0.02]
    wall = by_offset[0.0]
    above = by_offset[0.02]
    return {
        "axis": axis,
        "value": float(value),
        "threshold": float(threshold),
        "T": float(T),
        "kappa": float(params.kappa),
        "theta": float(params.theta),
        "xi": float(params.xi),
        "rho": float(params.rho),
        "p_star": physical.p_star,
        "rate": physical.rate,
        "pole1": adjacent["moment_poles"][0],
        "pole2": adjacent["moment_poles"][1],
        "p1": adjacent["p"].real,
        "delta_real": adjacent["singulant"].real,
        "delta_imag": adjacent["singulant"].imag,
        "monodromy_pred": monodromy,
        "monodromy_rel_error": abs(adjacent["singulant"].imag - monodromy) / abs(monodromy),
        "stokes_angle": math.atan2(adjacent["singulant"].imag, adjacent["singulant"].real),
        "lambda2_adj_abs": abs(local["lambda_second"]),
        "tau_max": tau_max,
        "selected_branch": scan["selected_branch"],
        "intersection_below": below["intersection_number"],
        "intersection_above": above["intersection_number"],
        "wall_min_distance": wall["min_distance_to_physical_saddle"],
    }


def high_order_case(payload):
    axis, value, n_terms = payload
    try:
        params, threshold, T = make_case(axis, value)
        physical = find_heston_tail_saddle(params, threshold, T=T, dps=60)
        adjacent = find_heston_first_adjacent_sheet_saddle(
            params, threshold, T=T, dps=75, reference_saddle=physical, p_min=-100.0
        )
        coeffs_mp, _ = heston_tail_fluctuation_coefficients(
            params, threshold, T=T, n_terms=n_terms, dps=125, saddle=physical, return_mpmath=True
        )
        conformal_mp = heston_conformal_borel_coefficients(coeffs_mp, physical.rate, dps=115)
        conformal = np.asarray([float(v) for v in conformal_mp], dtype=float)
        chain = heston_secondary_pade_chain(
            conformal_mp, physical.rate, adjacent["singulant"], n_coeffs=n_terms, dps=75, angle_window=0.40
        )
        if chain:
            nearest = min(chain, key=lambda item: abs(item[1] - adjacent["singulant"]))
            pade_zeta = nearest[1]
            pade_gap = abs(pade_zeta - adjacent["singulant"]) / abs(adjacent["singulant"])
        else:
            pade_zeta = complex(float("nan"), float("nan"))
            pade_gap = float("nan")

        start_order = max(14, n_terms - 12)
        try:
            fitted = fit_heston_secondary_singulant(
                conformal, physical.rate, adjacent["singulant"], start_order=start_order, end_order=n_terms - 1
            )
            fit_zeta = fitted["zeta"]
            fit_gap = abs(fit_zeta - adjacent["singulant"]) / abs(adjacent["singulant"])
        except Exception:
            fit_zeta = complex(float("nan"), float("nan"))
            fit_gap = float("nan")

        sector = heston_adjacent_sheet_fluctuation_coefficients(
            params, threshold, T=T, n_terms=10, dps=90, reference_saddle=physical, adjacent=adjacent
        )
        # A fixed |epsilon| is not meaningful once |Delta S_1| changes by an
        # order of magnitude across the parameter sweep.  Keep the exponential
        # suppression approximately fixed and optimally truncate the adjacent
        # fluctuation series before evaluating the lateral discontinuity.
        epsilon_radius = abs(adjacent["singulant"]) / 20.0
        epsilon_complex = epsilon_radius * np.exp(1j * np.angle(adjacent["singulant"]))
        adjacent_terms = [abs(complex(c) * epsilon_complex ** n) for n, c in enumerate(sector["coefficients"])]
        adjacent_optimal_terms = int(np.argmin(adjacent_terms) + 1)
        lateral = heston_secondary_lateral_stokes(
            epsilon_radius,
            conformal_mp,
            physical.rate,
            adjacent["singulant"],
            sector["coefficients"][:adjacent_optimal_terms],
            sector["prefactor_ratio"],
            n_coeffs=n_terms,
            angle_window=0.40,
            dps=75,
        )
        S = lateral["stokes"]
        return {
            "axis": axis,
            "value": float(value),
            "n_terms": int(n_terms),
            "pade_real": pade_zeta.real,
            "pade_imag": pade_zeta.imag,
            "pade_rel_gap": pade_gap,
            "fit_real": fit_zeta.real,
            "fit_imag": fit_zeta.imag,
            "fit_rel_gap": fit_gap,
            "stokes_real": S.real,
            "stokes_imag": S.imag,
            "stokes_distance_minus_one": abs(S + 1.0),
            "epsilon_radius": epsilon_radius,
            "adjacent_optimal_terms": adjacent_optimal_terms,
            "lambda2_adj_abs": abs(sector["lambda_second"]),
            "chain_poles": len(lateral["chain"]),
            "error": "",
        }
    except Exception as exc:
        return {
            "axis": axis,
            "value": float(value),
            "n_terms": int(n_terms),
            "pade_real": float("nan"),
            "pade_imag": float("nan"),
            "pade_rel_gap": float("nan"),
            "fit_real": float("nan"),
            "fit_imag": float("nan"),
            "fit_rel_gap": float("nan"),
            "stokes_real": float("nan"),
            "stokes_imag": float("nan"),
            "stokes_distance_minus_one": float("nan"),
            "epsilon_radius": float("nan"),
            "adjacent_optimal_terms": 0,
            "lambda2_adj_abs": float("nan"),
            "chain_poles": 0,
            "error": f"{type(exc).__name__}: {exc}",
        }


def write_csv(path, rows, columns):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(",".join(columns) + "\n")
        for row in rows:
            fields = []
            for col in columns:
                value = row[col]
                if isinstance(value, str):
                    fields.append(value.replace(",", ";"))
                elif isinstance(value, (int, np.integer)):
                    fields.append(str(value))
                else:
                    fields.append(f"{float(value):.15g}")
            handle.write(",".join(fields) + "\n")


def make_figures(structural, high_order):
    axis_labels = {
        "xstar": r"$x_\star$",
        "T": r"$T$",
        "rho": r"$\rho$",
        "xi": r"$\xi$",
        "kappa": r"$\kappa$",
    }
    for axis in AXES:
        rows = sorted([r for r in structural if r["axis"] == axis], key=lambda r: r["value"])
        x = np.asarray([r["value"] for r in rows])
        dr = np.asarray([r["delta_real"] for r in rows])
        di = np.asarray([r["delta_imag"] for r in rows])
        mono = np.asarray([r["monodromy_pred"] for r in rows])
        plt.figure(figsize=(6.6, 4.5))
        plt.plot(x, dr, "o-", label=r"$\mathrm{Re}\,\Delta S_1$")
        plt.plot(x, di, "s-", label=r"$\mathrm{Im}\,\Delta S_1$")
        plt.plot(x, mono, "--", label=r"$2\pi\kappa\theta/\xi^2$")
        plt.xlabel(axis_labels[axis])
        plt.ylabel("action")
        plt.title(f"Adjacent Heston singulant versus {axis}")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(FIGURES / f"heston_generalization_singulant_{axis}.png", dpi=220)
        plt.close()

    # Aggregate robustness panels.
    labels = []
    gaps = []
    stokes = []
    for axis in AXES:
        rows = sorted([r for r in high_order if r["axis"] == axis], key=lambda r: r["value"])
        for row in rows:
            labels.append(f"{axis}={row['value']:g}")
            gaps.append(row["pade_rel_gap"])
            stokes.append(row["stokes_distance_minus_one"])
    idx = np.arange(len(labels))
    plt.figure(figsize=(10.5, 4.8))
    plt.semilogy(idx, gaps, "o", label="secondary Borel-location relative gap")
    plt.semilogy(idx, stokes, "s", label=r"$|S_{\rm lateral}+1|$")
    for boundary in np.cumsum([len(AXES[a]) for a in AXES])[:-1] - 0.5:
        plt.axvline(boundary, linewidth=0.6)
    plt.xticks(idx, labels, rotation=70, ha="right", fontsize=6.5)
    plt.ylabel("relative diagnostic error")
    plt.title("Two-sector resurgence across one-parameter Heston sweeps")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "heston_generalization_robustness.png", dpi=220)
    plt.close()

    # Topological wall crossing summary.
    axes_names = list(AXES)
    topo = np.full((len(axes_names), 5), np.nan)
    wall = np.full_like(topo, np.nan)
    for i, axis in enumerate(axes_names):
        rows = sorted([r for r in structural if r["axis"] == axis], key=lambda r: r["value"])
        topo[i, :] = [r["intersection_above"] - r["intersection_below"] for r in rows]
        wall[i, :] = [r["wall_min_distance"] for r in rows]
    plt.figure(figsize=(7.2, 4.2))
    im = plt.imshow(topo, aspect="auto", vmin=-1.2, vmax=0.2)
    plt.colorbar(im, label=r"$\Delta\langle\Gamma_B,\mathcal{K}_1\rangle$")
    plt.yticks(range(len(axes_names)), axes_names)
    plt.xticks(range(5), ["low", "mid-low", "baseline", "mid-high", "high"])
    for i in range(topo.shape[0]):
        for j in range(topo.shape[1]):
            plt.text(j, i, f"{int(topo[i,j])}", ha="center", va="center", fontsize=9)
    plt.title("Picard--Lefschetz intersection jump across parameter sweeps")
    plt.tight_layout()
    plt.savefig(FIGURES / "heston_generalization_thimble_topology.png", dpi=220)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--structural-only", action="store_true")
    parser.add_argument("--n-terms", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    structural = []
    for axis, values in AXES.items():
        for value in values:
            print(f"[structure] {axis}={value}", flush=True)
            row = structural_case(axis, value)
            structural.append(row)
            print(
                f"  I={row['rate']:.5g}, Delta={row['delta_real']:.5g}+{row['delta_imag']:.5g}i, "
                f"topology {row['intersection_below']}->{row['intersection_above']}, wall d={row['wall_min_distance']:.2e}",
                flush=True,
            )

    structural_columns = [
        "axis", "value", "threshold", "T", "kappa", "theta", "xi", "rho", "p_star", "rate",
        "pole1", "pole2", "p1", "delta_real", "delta_imag", "monodromy_pred", "monodromy_rel_error",
        "stokes_angle", "lambda2_adj_abs", "tau_max", "selected_branch", "intersection_below",
        "intersection_above", "wall_min_distance",
    ]
    write_csv(RESULTS / "heston_parameter_sweep_structure.csv", structural, structural_columns)

    if args.structural_only:
        make_figures(structural, [])
        return

    payloads = [(axis, value, args.n_terms) for axis, values in AXES.items() for value in values]
    high_order = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(high_order_case, payload): payload for payload in payloads}
        for future in as_completed(futures):
            axis, value, _ = futures[future]
            row = future.result()
            high_order.append(row)
            if row["error"]:
                print(f"[high-order] {axis}={value}: FAIL {row['error']}", flush=True)
            else:
                print(
                    f"[high-order] {axis}={value}: Pade gap={row['pade_rel_gap']:.3%}, "
                    f"|S+1|={row['stokes_distance_minus_one']:.3%}",
                    flush=True,
                )

    high_order.sort(key=lambda r: (list(AXES).index(r["axis"]), r["value"]))
    high_columns = [
        "axis", "value", "n_terms", "pade_real", "pade_imag", "pade_rel_gap", "fit_real", "fit_imag",
        "fit_rel_gap", "stokes_real", "stokes_imag", "stokes_distance_minus_one", "epsilon_radius",
        "adjacent_optimal_terms", "lambda2_adj_abs", "chain_poles", "error",
    ]
    write_csv(RESULTS / "heston_parameter_sweep_resurgence.csv", high_order, high_columns)
    make_figures(structural, high_order)

    valid = [r for r in high_order if not r["error"] and np.isfinite(r["pade_rel_gap"])]
    print("\nSUMMARY", flush=True)
    print("points", len(valid), "/", len(high_order), flush=True)
    print("max monodromy rel error", max(r["monodromy_rel_error"] for r in structural), flush=True)
    print("topology jumps", sorted(set(r["intersection_above"] - r["intersection_below"] for r in structural)), flush=True)
    print("max wall distance", max(r["wall_min_distance"] for r in structural), flush=True)
    print("median Pade gap", float(np.median([r["pade_rel_gap"] for r in valid])), flush=True)
    print("max Pade gap", max(r["pade_rel_gap"] for r in valid), flush=True)
    print("median |S+1|", float(np.median([r["stokes_distance_minus_one"] for r in valid])), flush=True)
    print("max |S+1|", max(r["stokes_distance_minus_one"] for r in valid), flush=True)


if __name__ == "__main__":
    main()
