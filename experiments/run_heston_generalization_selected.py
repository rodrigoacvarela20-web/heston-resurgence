from pathlib import Path
import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from run_heston_parameter_generalization import make_case
from ftfinance.heston_resurgence import (
    find_heston_tail_saddle,
    find_heston_first_adjacent_sheet_saddle,
    fit_heston_secondary_singulant,
    heston_conformal_borel_coefficients,
    heston_secondary_pade_chain,
    heston_tail_fluctuation_coefficients,
)

CASES = [
    ("xstar", -0.15), ("xstar", -0.35),
    ("T", 0.5), ("T", 2.0),
    ("rho", -0.9), ("rho", 0.0),
    ("xi", 0.25), ("xi", 0.8),
    ("kappa", 0.75), ("kappa", 4.0),
]


def run_case(payload):
    axis, value, n_terms = payload
    try:
        params, threshold, T = make_case(axis, value)
        physical = find_heston_tail_saddle(params, threshold, T=T, dps=65)
        adjacent = find_heston_first_adjacent_sheet_saddle(
            params, threshold, T=T, dps=80, reference_saddle=physical, p_min=-120.0
        )
        coeffs, _ = heston_tail_fluctuation_coefficients(
            params, threshold, T=T, n_terms=n_terms, dps=150, saddle=physical, return_mpmath=True
        )
        conformal = heston_conformal_borel_coefficients(coeffs, physical.rate, dps=130)
        chain = heston_secondary_pade_chain(
            conformal, physical.rate, adjacent["singulant"], n_coeffs=n_terms,
            dps=85, angle_window=0.48, max_abs_w=0.94
        )
        nearest = min(chain, key=lambda item: abs(item[1] - adjacent["singulant"]))
        pade = nearest[1]
        pade_gap = abs(pade - adjacent["singulant"]) / abs(adjacent["singulant"])

        try:
            fit = fit_heston_secondary_singulant(
                [float(v) for v in conformal], physical.rate, adjacent["singulant"],
                start_order=max(14, n_terms - 12), end_order=n_terms - 1
            )
            fit_zeta = fit["zeta"]
            fit_gap = abs(fit_zeta - adjacent["singulant"]) / abs(adjacent["singulant"])
        except Exception:
            fit_zeta = complex(float("nan"), float("nan"))
            fit_gap = float("nan")

        return {
            "axis": axis, "value": value,
            "delta_real": adjacent["singulant"].real,
            "delta_imag": adjacent["singulant"].imag,
            "pade_real": pade.real, "pade_imag": pade.imag,
            "pade_rel_gap": pade_gap,
            "fit_real": fit_zeta.real, "fit_imag": fit_zeta.imag,
            "fit_rel_gap": fit_gap, "error": "",
        }
    except Exception as exc:
        return {
            "axis": axis, "value": value,
            "delta_real": np.nan, "delta_imag": np.nan,
            "pade_real": np.nan, "pade_imag": np.nan, "pade_rel_gap": np.nan,
            "fit_real": np.nan, "fit_imag": np.nan, "fit_rel_gap": np.nan,
            "error": f"{type(exc).__name__}: {exc}",
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-terms", type=int, default=40)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    payloads = [(axis, value, args.n_terms) for axis, value in CASES]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_case, payload): payload for payload in payloads}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            if row["error"]:
                print(f"{row['axis']}={row['value']}: {row['error']}", flush=True)
            else:
                print(f"{row['axis']}={row['value']}: Pade gap={row['pade_rel_gap']:.3%}", flush=True)

    order = {case: i for i, case in enumerate(CASES)}
    rows.sort(key=lambda row: order[(row["axis"], row["value"])])
    path = ROOT / "results" / "heston_generalization_selected40.csv"
    columns = [
        "axis", "value", "delta_real", "delta_imag", "pade_real", "pade_imag",
        "pade_rel_gap", "fit_real", "fit_imag", "fit_rel_gap", "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    valid = [row for row in rows if not row["error"]]
    print("median Pade gap", float(np.median([row["pade_rel_gap"] for row in valid])))
    print("max Pade gap", max(row["pade_rel_gap"] for row in valid))
    print("saved", path)


if __name__ == "__main__":
    main()
