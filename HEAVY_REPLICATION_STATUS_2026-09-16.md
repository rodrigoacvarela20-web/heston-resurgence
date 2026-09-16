# Numerical experiment reproduction — 16 September 2026

**Status: partial, not a reproduction of every expensive calculation or independent verification of the full paper.** Experiments ran in a fresh working *copy* of the published research package with NumPy/SciPy/mpmath and no network data; completed calculations can replace existing outputs in that copy. Existing GitHub figure/CSV binaries and the v1.0 PDF were not overwritten as a consequence of this audit.

| Script or calculation | Actual result |
|---|---|
| `run_heston_resurgence.py` | completed in ~15 s: affine rate `0.396018751741`, independent Hamiltonian action `0.396018650224`, nearest 26-term Padé pole `-0.397445141583`; maximum reported relative Borel-Padé error `1.845e-4` and maximum *optimally truncated* relative error `0.9603` on its epsilon grid. That high truncation error must not be concealed. |
| `run_gaussian_resurgence.py` | completed in ~5 s: action `1.125`, Borel-Padé maximum relative error `1.115e-9` on its chosen grid. |
| `run_quartic_factor_resurgence.py` | completed in ~5 s: expected nearest action `-0.25`, Borel-Padé maximum relative error `9.181e-9` on its chosen grid. |
| `run_heston_thimble_geometry.py` | completed in ~8 s; reported intersection integer -1 at sampled contour offsets. This is a result of the implementation, not an independent topological proof. |
| `run_heston_uniform_airy_fold.py` | completed in ~7 s; printed Airy slope `-1.0932684905` and action-splitting coefficient `1.5241544682`. |
| `run_heston_generalization_summary.py` | completed in ~9 s; selected Padé gaps ranged up to ~0.0430 (script's dimensionless units). |
| `run_heston_global_phase_portrait.py` | completed in ~10 s and reported a 763-grid-point phase portrait. |
| `run_heston_secondary_resurgence.py` | completed after a previous short timeout and reported adjacent singulant `2.792240162467 + 2.482246047281i`; 45-term Padé relative gap ~2.97%. |
| `run_heston_resurgent_phase_diagram.py` | **failed originally**: the high-precision real fold root carried an imaginary round-off component (~1e-86), and `float(mpc)` raised `TypeError`. Corrected in this repository by checking that the imaginary part is negligible and extracting `mp.re`. The corrected script completed; fold coefficient `3.21764446395`, `rho_c≈0.0299495156`, anti-Stokes maturities `T≈2.67815584`, `kappa≈5.47078862`, monodromy relative error ~`2.38e-15` over the supplied sweep. |
| `run_heston_stokes_resurgence.py`, `run_heston_postcaustic_airy.py`, full `run_heston_parameter_generalization.py` | **not completed** within individually bounded 10–32-second attempts. No equivalence to the paper's expensive 75-/higher-order calculations is claimed. |

Some scripts use *checked-in input coefficients/CSV files*: successful execution does not independently regenerate those expensive high-order coefficients. A complete replication requires resource-bounded fresh generation without reading precomputed values, independent numerical implementations, sensitivity to precision/order/contours, checked output hashes, and comparison of numerical tables and every figure to the PDF. This audit did **not** establish those. Tests and agreement at selected parameter points do not establish global resurgence, market predictability, or external peer review.

For the remaining work, run heavy scripts separately with explicit wall-time budgets and output directories on an appropriately resourced machine. Keep a source commit, dependency lockfile, numeric precision, seeds, exact commands, per-artifact checksums and failure logs. Avoid running in the main repository directory if the original paper's reference outputs need to be preserved.