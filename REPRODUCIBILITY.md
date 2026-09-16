# Reproducibility guide

## Environment

Recommended: Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## One-command release validation

```bash
python reproduce.py
```

This runs the 19 automated regression tests. To rerun the three core numerical benchmarks in one command, use:

```bash
python reproduce.py --benchmarks
```

The benchmark mode runs the leading Heston calculation plus the Gaussian and quartic validation problems.

## Paper build

A standard LaTeX + BibTeX installation is required.

```bash
make paper
```

The output is `paper/resurgent_stochastic_finance.pdf`.

## Secondary-sector calculations

```bash
python experiments/run_heston_secondary_resurgence.py
python experiments/run_heston_stokes_resurgence.py
python experiments/run_heston_lateral_resurgence.py --reuse-coefficients
python experiments/run_heston_thimble_geometry.py
```

Some of these calculations use arbitrary-precision derivatives/Cauchy extraction and can be substantially slower than the core validation.

## Parameter continuation and fold analysis

```bash
python experiments/run_heston_parameter_generalization.py
python experiments/run_heston_generalization_summary.py
python experiments/run_heston_resurgent_phase_diagram.py
python experiments/run_heston_uniform_airy_fold.py
python experiments/run_heston_global_phase_portrait.py
python experiments/run_heston_postcaustic_airy.py
```

The committed CSV files in `results/` and PNGs in `figures/` are the outputs used by the final manuscript. Heavy computations are intentionally not run by default in `reproduce.py`.

## Numerical standards

- Compare non-perturbative data with independently computed actions whenever possible.
- Check Padé features across approximation order before identifying them with saddles.
- Treat quoted percentage discrepancies as numerical diagnostics, not rigorous error bounds.
- Do not infer a renormalon from factorial growth alone.
- Preserve the distinction between local numerical/geometric evidence and a global analytic theorem.
