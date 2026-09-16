# Manuscript overview

**Rodrigo Varela (September 2026)** — *Resurgent Asymptotics of Rare-Event Tails in the Heston Model: Borel Singularities, Stokes Data, and Caustic Continuation* (49-page research manuscript).

## Research question

The manuscript studies whether rare-event saddle geometry in a uniformly small-noise Heston stochastic-volatility model is reflected in the large-order and Borel structure of a specific left-tail probability. The scaled affine transform takes the exact form

$$\mathbb E\left[e^{pX_T/\varepsilon}\right]=e^{\Lambda(p)/\varepsilon},$$

which yields a one-dimensional inverse-Laplace integral suitable for saddle analysis.

## Results reported in the manuscript

At its benchmark parameter point, the physical affine saddle and a separately formulated Freidlin–Wentzell Hamiltonian shooting problem recover a matching large-deviation scale. The leading Borel obstruction is attributed to the **explicit CDF pole at `p=0`**, rather than misidentified as a second instanton. The first continued logarithmic sheet gives an adjacent singulant

$$\Delta S_1=2.792240162467\pm2.482246047281\,i.$$

The manuscript reports a 75-coefficient late-order estimate within approximately 0.33% of the independently continued action, a lateral-Borel numerical value `−0.998566 + 0.001484 i`, and an oriented Picard–Lefschetz intersection integer `−1`. It also studies parameter continuation, a fold caustic, anti-Stokes boundaries and an Airy-uniform approximation. **These are claims of the supplied manuscript, not independently rerun numerical results in this GitHub repository.**

## Interpretation

The claimed scope is *local resurgent evidence for this scaled observable on the tested parameter domain*. No global theorem for all Heston parameters or observables, quantum interpretation of markets, market-crash predictor, or financial renormalon is claimed. The underlying Heston/path-integral literature precedes this study; the manuscript does not claim that writing a path integral for Heston is itself new.

## Missing primary materials

The full original PDF and LaTeX source are prepared separately but not yet in the repository. The original figure files (37), computational source code, benchmark outputs and test suite are still missing from this GitHub publication. The LaTeX source cannot be rebuilt here without the original figures. Refer to [PUBLICATION_STATUS.md](../PUBLICATION_STATUS.md).