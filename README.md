# Resurgent Asymptotics of Heston Rare-Event Tails

**Research-project release (v1.0; not peer reviewed).** This repository studies whether non-perturbative saddle geometry in a small-noise Heston model is encoded in the large-order behaviour and Borel structure of a rare-event tail probability.

The project uses tools familiar from semiclassical analysis and quantum field theory - saddle expansions, instanton actions, Borel transforms, Padé continuation, Stokes phenomena, Picard-Lefschetz theory and uniform Airy asymptotics - but applies them to a classical stochastic-volatility problem. The claim is deliberately narrow: this is evidence for a resurgent description of one Heston tail observable, not a claim that financial markets are quantum systems.

## Main results

For the benchmark

```text
kappa=2, theta=v0=0.04, xi=0.45, rho=-0.7, T=1, x*= -0.25
```

the independent affine and Freidlin-Wentzell calculations give the same leading rare-event scale to numerical precision. High-order fluctuations recover the associated Borel scale, and Borel-Padé resummation extends the useful range of the small-noise expansion.

The first adjacent logarithmic sheet gives a stronger test. Its independently continued action is

```text
Delta S1 = 2.792240162467 +/- 2.482246047281 i
```

and a 75-coefficient late-order fit locates the corresponding secondary Borel singularity within `0.328%`. The direct lateral Borel jump gives

```text
S_lateral = -0.998565810 + 0.001484197 i
```

while an independent Picard-Lefschetz calculation gives the oriented intersection integer

```text
S01 = -1.
```

Parameter continuation shows that this structure survives on a connected nondegenerate domain. The continuation also reveals where the local transseries basis must reorganize:

- a correlation fold at `rho_c = 0.0299495156775`, where two real adjacent saddles coalesce and continue as a complex-conjugate pair;
- anti-Stokes crossings where the real part of the adjacent action vanishes;
- an Airy/Chester-Friedman-Ursell uniform description through the fold;
- coherent post-caustic continuation of lifted actions, local fluctuation sectors and projected Borel singularities.

The resulting picture is best described as a **resurgent atlas**: local transseries charts on regular parameter domains, Airy-uniform blocks at caustics, and changes of exponential dominance across anti-Stokes boundaries.

## Paper

The full derivation, numerical tests, limitations and references are in:

```text
paper/resurgent_stochastic_finance.pdf
paper/main.tex
```

The final manuscript title is **“Resurgent Asymptotics of Rare-Event Tails in the Heston Model: Borel Singularities, Stokes Data, and Caustic Continuation.”**

## Quick start

Create an environment and install the package:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run the release validation:

```bash
python reproduce.py
```

This runs the automated regression tests. To rerun the three core numerical benchmarks as well, use `python reproduce.py --benchmarks`. The heavier secondary-sector and fold checks are available with `python reproduce.py --extended`.

For the paper build:

```bash
make paper
```

For the heavier continuation experiments, see `REPRODUCIBILITY.md`.

## Repository layout

```text
src/ftfinance/       reusable numerical methods
experiments/         reproducible experiment entry points
tests/               automated regression tests
results/             numerical CSV outputs
figures/             generated research figures
paper/               LaTeX source and compiled manuscript
reproduce.py         one-command release validation
RESULTS.md           detailed numerical findings
ROADMAP.md           frozen scope + future research directions
```

## Scientific status

The strongest claims supported by the code and manuscript are:

1. the leading Heston large-deviation scale is recovered independently from affine, Hamiltonian and large-order calculations;
2. the first adjacent logarithmic-sheet action is quantitatively visible in high-order/Borel data;
3. the associated Stokes multiplier is consistent with `-1` both from a lateral Borel jump and from Picard-Lefschetz intersection geometry;
4. the local two-sector structure deforms robustly in parameter space until caustic or anti-Stokes boundaries require a new local basis;
5. the first correlation fold is uniformly resolved by an Airy normal form and continues into a complex-conjugate saddle pair.

The project **does not** claim a global theorem for the full Heston Riemann surface, resurgence of every Heston observable, empirical prediction of financial crashes, or the existence of financial renormalons.

## Reproducibility

The release test suite passes `19/19` tests in the frozen v1.0 environment used to build the paper. Exact run commands, expected outputs and the distinction between quick and expensive experiments are documented in `REPRODUCIBILITY.md`.

## Citation

If you use or discuss this research project, see `CITATION.cff` for the preferred citation metadata.

## License

See `LICENSE`.
