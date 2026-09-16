# Project status and future work

## v1.0 status: frozen research project

The scientific scope of the first project is complete. The frozen release includes:

- two solvable validation benchmarks;
- a small-noise Heston steepest-descent formulation;
- an independent Freidlin-Wentzell/Hamiltonian rare-event calculation;
- high-order coefficient generation and Borel-Padé analysis;
- an adjacent logarithmic-sheet saddle and local fluctuation sector;
- direct lateral Borel discontinuities and a Picard-Lefschetz Stokes sign;
- five-direction parameter continuation;
- a two-dimensional `(T,rho)` phase atlas;
- a correlation fold, Airy/CFU uniformization and post-caustic complex-saddle continuation.

No new theoretical layer should be added to the v1.0 paper. Corrections, reproducibility fixes and clearly documented numerical refinements are appropriate; new global sheet topology belongs to a follow-up project.

## Open problems for a separate project

1. **Global logarithmic-sheet graph.** Classify moment poles, sheets, saddle branches and thimble endpoints over a larger connected Heston parameter domain.
2. **Global multi-sector transseries.** Replace the current atlas of local charts by a sheet-aware global continuation where possible.
3. **Uniform Heston amplitudes.** Derive Heston-specific CFU amplitude coefficients beyond the canonical Airy block.
4. **Post-caustic Stokes data.** Repeat lateral-discontinuity and thimble tests directly for complex sectors after the fold.
5. **Additional two-dimensional slices.** Map `(xi,kappa)` and `(xstar,T)` in the same way as `(T,rho)`.
6. **Other observables/models.** Test option prices, expected shortfall, rough-Heston-like models or other affine diffusions.
7. **Renormalon question.** Only meaningful after constructing a genuine scale-dependent/RG problem whose factorial growth can be attributed to scale integration.

These are follow-up research directions, not missing steps required for the v1.0 result.
