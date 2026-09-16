"""One-command validation for the frozen v1.0 research release."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

TESTS = [[sys.executable, "-m", "pytest", "-q"]]
BENCHMARKS = [
    [sys.executable, "experiments/run_heston_resurgence.py"],
    [sys.executable, "experiments/run_gaussian_resurgence.py"],
    [sys.executable, "experiments/run_quartic_factor_resurgence.py"],
]
EXTENDED = [
    [sys.executable, "experiments/run_heston_stokes_resurgence.py"],
    [sys.executable, "experiments/run_heston_lateral_resurgence.py", "--reuse-coefficients"],
    [sys.executable, "experiments/run_heston_thimble_geometry.py"],
    [sys.executable, "experiments/run_heston_uniform_airy_fold.py"],
    [sys.executable, "experiments/run_heston_postcaustic_airy.py"],
]


def run(cmd: list[str]) -> None:
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the v1.0 research release.")
    parser.add_argument("--benchmarks", action="store_true", help="rerun the three core numerical benchmarks")
    parser.add_argument("--extended", action="store_true", help="also run slower secondary-sector and fold checks")
    args = parser.parse_args()
    for command in TESTS:
        run(command)
    if args.benchmarks or args.extended:
        for command in BENCHMARKS:
            run(command)
    if args.extended:
        for command in EXTENDED:
            run(command)
    print("\nRelease validation completed successfully.")


if __name__ == "__main__":
    main()
