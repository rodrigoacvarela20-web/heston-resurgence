# Scientific and numerical audit — 16 September 2026

This is an internal code/numerical audit, **not external peer review or validation of a theorem**.

The original 19 tests plus five new tests passed locally (24 total). An independently integrated complex Riccati equation agreed with the affine cumulant-generating function at four real/complex arguments. The importance-sampling estimator previously allowed `0 * exp(large non-event log weight)` to yield NaN; it now exponentiates only event weights and validates inputs. Correlation endpoints and basic parameter checks were also improved.

## Newly observed compatibility failure and response

A fresh GitHub Actions run initially failed **three fold/post-caustic tests** with `mpmath 1.4.1` (`ComplexResult: square root of a negative number` in the fold solver). The original code had only been checked with `mpmath 1.3.0`. For a reproducible current release, `pyproject.toml` now limits the installed version to `mpmath>=1.3,<1.4` until the complex-branch continuation is reviewed and ported. A second GitHub Actions run with that constraint **passed the regression-test job**: [run 2](https://github.com/rodrigoacvarela20-web/heston-resurgence/actions/runs/35110403745). This is a *dependency compatibility workaround*, not a proof that the 1.4 branch discrepancy is scientifically harmless.

## Remaining scientific limitations

The PDF remains the original v1.0 manuscript, describing its original 19 tests. Code repairs, selected numerical agreement and a passing CI run do not independently verify all Stokes data, analytic continuation, caustics, high-order coefficient generation or novelty. In particular, some completed scripts reuse checked-in coefficients and three heavy experiments were not completed in the later replication attempt; see [replication status](HEAVY_REPLICATION_STATUS_2026-09-16.md). The small-noise deformation is a mathematical study, not a demonstrated financial forecasting method.

To verify the quick tests: `python -m pip install -e ".[dev]"` and `python -m pytest -q tests`. The extended experiments need separate, resource-bounded reproduction. Neither the passing CI job nor manuscript publication constitutes external scientific validation.
