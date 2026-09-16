# Scientific audit: 16 September 2026

This internal technical review is not external peer review.

The original 19 offline tests and five additional tests passed locally (24 total). An independently integrated complex Riccati equation agreed with the affine cumulant-generating function for four real or complex arguments. A confirmed numerical defect was fixed: non-event samples with extremely large likelihood-ratio logs previously produced an undefined zero-times-infinity expression. The importance-sampling estimator now exponentiates event weights only, rescales the contributions and validates inputs. Correlation endpoint handling and basic parameter checks were also improved.

The published PDF still describes the original v1.0 research results and 19 original tests. The code corrections and extra tests do not independently validate the manuscript's Stokes data, analytic continuation, caustic calculations or novelty. Full reproduction of expensive numerical experiments and external expert assessment remain outstanding. A separate set of later referee notes in the source archive did not constitute an updated paper-and-code release. The small-noise deformation is a mathematical study, not a demonstrated market forecasting method.

Run `python -m pip install -e ".[dev]"` followed by `python -m pytest -q tests`. The 24-test result was observed in a local copy; no GitHub-hosted continuous-integration result is claimed.