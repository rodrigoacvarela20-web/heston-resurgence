from pathlib import Path
import sys
import cmath

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ftfinance.heston import HestonParams
from ftfinance.heston_resurgence import (
    find_heston_adjacent_sheet_saddle,
    find_heston_tail_saddle,
    heston_adjacent_sheet_fluctuation_coefficients,
    heston_negative_moment_pole,
    heston_picard_lefschetz_scan,
)

FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
FIGURES.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)


def main():
    params = HestonParams(mu=0.0, kappa=2.0, theta=0.04, xi=0.45, rho=-0.7, v0=0.04)
    threshold = -0.25
    T = 1.0
    contour_real = -1.0
    offsets = (-0.05, -0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05)

    physical = find_heston_tail_saddle(params, threshold, T=T, dps=90)
    adjacent = find_heston_adjacent_sheet_saddle(params, threshold, T=T, dps=100, reference_saddle=physical)
    local = heston_adjacent_sheet_fluctuation_coefficients(
        params, threshold, T=T, n_terms=3, dps=105, reference_saddle=physical
    )
    moment_pole = heston_negative_moment_pole(params, T=T, bracket=(-6.7, -6.2))
    scan = heston_picard_lefschetz_scan(
        params,
        threshold,
        T=T,
        contour_real=contour_real,
        angle_offsets=offsets,
        tau_max=90.0,
        reference_saddle=physical,
        adjacent=adjacent,
        adjacent_local=local,
    )

    with open(RESULTS / "heston_thimble_wall_crossing.csv", "w", encoding="utf-8") as handle:
        handle.write("offset,theta,intersection_number,n_intersections,min_distance_to_physical_saddle,end_real,end_imag,cross_real,cross_imag,cross_orientation\n")
        for row in scan["rows"]:
            crossing = row["intersections"][0] if row["intersections"] else None
            handle.write(
                f"{row['offset']:.12g},{row['theta']:.15g},{row['intersection_number']},{row['n_intersections']},"
                f"{row['min_distance_to_physical_saddle']:.15g},{row['end'].real:.15g},{row['end'].imag:.15g},"
                f"{'' if crossing is None else format(crossing['point'].real,'.15g')},"
                f"{'' if crossing is None else format(crossing['point'].imag,'.15g')},"
                f"{'' if crossing is None else crossing['orientation']}\n"
            )

    theta_s = scan["theta_stokes"]
    upper = adjacent["singulant"]
    summary = np.array([[
        theta_s,
        physical.p_star,
        adjacent["p"].real,
        moment_pole,
        upper.real,
        upper.imag,
        min(r["min_distance_to_physical_saddle"] for r in scan["rows"] if r["offset"] == 0.0),
        next(r["intersection_number"] for r in scan["rows"] if abs(r["offset"] - 0.02) < 1e-12),
        next(r["intersection_number"] for r in scan["rows"] if abs(r["offset"] + 0.02) < 1e-12),
    ]])
    np.savetxt(
        RESULTS / "heston_thimble_summary.csv",
        summary,
        delimiter=",",
        header="theta_stokes,p_star,p1,moment_pole,deltaS_real,deltaS_imag,heteroclinic_min_distance,intersection_above,intersection_below",
        comments="",
    )

    # Full wall-crossing geometry.
    plt.figure(figsize=(8.0, 5.6))
    selected = [(-0.02, "below Stokes wall"), (0.0, "Stokes wall"), (0.02, "above Stokes wall")]
    for offset, label in selected:
        _, path = scan["paths"][offset]
        # Trim paths only for plotting once they have clearly reached an endpoint/infinity.
        keep = (path.real > -9.2) & (path.real < 8.0) & (np.abs(path.imag) < 5.0)
        path_plot = path[keep]
        plt.plot(path_plot.real, path_plot.imag, linewidth=1.8, label=label)
    plt.axvline(contour_real, linestyle="--", linewidth=1.2, label=r"Bromwich contour $\Gamma_B$")
    plt.scatter([physical.p_star], [0.0], s=75, marker="o", zorder=5, label=r"physical saddle $p_\star$")
    plt.scatter([adjacent["p"].real], [0.0], s=85, marker="x", linewidths=2.0, zorder=5, label=r"adjacent saddle $p_1$")
    plt.scatter([moment_pole], [0.0], s=180, marker="|", linewidths=2.2, zorder=5, label="moment pole")
    plt.axhline(0.0, linewidth=0.7)
    plt.xlim(-8.8, 5.0)
    plt.ylim(-1.0, 4.5)
    plt.xlabel(r"Re $p$")
    plt.ylabel(r"Im $p$")
    plt.title("Heston Picard--Lefschetz wall crossing")
    plt.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    plt.savefig(FIGURES / "heston_thimble_wall_crossing.png", dpi=220)
    plt.close()

    # Integer intersection number as the Stokes phase is crossed.
    offsets_arr = np.asarray([r["offset"] for r in scan["rows"]])
    ints = np.asarray([r["intersection_number"] for r in scan["rows"]])
    plt.figure(figsize=(7.2, 4.4))
    plt.step(offsets_arr, ints, where="mid", linewidth=1.8)
    plt.scatter(offsets_arr, ints, s=34)
    plt.axvline(0.0, linestyle="--", linewidth=1.0, label="Stokes wall")
    plt.yticks([-1, 0])
    plt.xlabel(r"phase offset $\delta\vartheta$ (rad)")
    plt.ylabel(r"$\langle\Gamma_B,\mathcal{K}_1\rangle$")
    plt.title("Topological jump of the Heston dual-cycle intersection")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "heston_thimble_intersection_jump.png", dpi=220)
    plt.close()

    # Orientation of the unique crossing just above the wall.
    row = next(r for r in scan["rows"] if abs(r["offset"] - 0.02) < 1e-12)
    _, path = scan["paths"][0.02]
    crossing = row["intersections"][0]
    k = crossing["index"]
    lo = max(0, k - 45)
    hi = min(len(path), k + 46)
    local_path = path[lo:hi]
    point = crossing["point"]

    plt.figure(figsize=(6.8, 5.0))
    plt.plot(local_path.real, local_path.imag, linewidth=2.0, label=r"dual cycle $\mathcal{K}_1$")
    plt.axvline(contour_real, linestyle="--", linewidth=1.5, label=r"upward $\Gamma_B$")
    plt.scatter([point.real], [point.imag], s=65, zorder=5)
    # arrows show orientations
    j = min(25, len(local_path) - 2)
    z0, z1 = local_path[j], local_path[j + 1]
    dv = z1 - z0
    plt.annotate("", xy=(z1.real, z1.imag), xytext=(z0.real, z0.imag), arrowprops=dict(arrowstyle="->", lw=1.8))
    plt.annotate("", xy=(contour_real, point.imag + 0.35), xytext=(contour_real, point.imag - 0.05), arrowprops=dict(arrowstyle="->", lw=1.8))
    plt.text(point.real + 0.05, point.imag + 0.08, r"$\det(t_{\Gamma},t_{\mathcal{K}})<0$", fontsize=10)
    plt.xlim(contour_real - 0.65, contour_real + 0.65)
    plt.ylim(point.imag - 0.65, point.imag + 0.65)
    plt.xlabel(r"Re $p$")
    plt.ylabel(r"Im $p$")
    plt.title("Negative orientation of the unique Bromwich crossing")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "heston_thimble_orientation.png", dpi=220)
    plt.close()

    print(f"Stokes angle = {theta_s:.12f} rad")
    print(f"Physical saddle = {physical.p_star:.12f}")
    print(f"Adjacent saddle = {adjacent['p'].real:.12f}")
    print(f"Delta S1 = {upper.real:.12f} {upper.imag:+.12f}i")
    for row in scan["rows"]:
        print(
            f"offset={row['offset']:+.3f}: intersection={row['intersection_number']:+d}, "
            f"n={row['n_intersections']}, min|p-p*|={row['min_distance_to_physical_saddle']:.5g}"
        )


if __name__ == "__main__":
    main()
