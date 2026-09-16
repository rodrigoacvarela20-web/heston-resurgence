# Current Results

## Heston benchmark

Parameters:

```text
mu = 0
kappa = 2
theta = 0.04
xi = 0.45
rho = -0.7
v0 = 0.04
T = 1
left-tail threshold x* = -0.25
```

For the uniformly small-noise Heston system,

```text
dX = (mu - V/2) dt + sqrt(epsilon V) dB
dV = kappa(theta-V) dt + sqrt(epsilon) xi sqrt(V) dW
```

the scaled affine transform obeys exactly

```text
E[exp(p X_T / epsilon)] = exp(Lambda(p) / epsilon).
```

The left-tail CDF is therefore a one-dimensional steepest-descent problem.

### Independent non-perturbative calculations

The affine saddle gives

```text
p*                   = -2.849561078040
affine rate I(x*)    =  0.396018751741
Lambda''(p*)         =  0.154480122015
```

The continuous Freidlin-Wentzell Hamiltonian shooting solver gives

```text
p_x                  = -2.849561078040
Hamiltonian action   =  0.396018650224
```

The relative action mismatch is at the numerical integration/discretization level.

### Large-order result

Using 31 arbitrary-precision saddle-expansion coefficients,

```text
c0 = 1
c1 = -1.599509864...
c2 = 6.018681066...
c3 = -37.99045699...
...
```

the ratios

```text
A_n = n c[n-1] / c[n]
```

are extrapolated as a cubic polynomial in `1/n`. The inferred Borel location is

```text
zeta_large_order = -0.396018496470
```

compared with

```text
-I(x*)            = -0.396018751741.
```

The absolute difference is approximately `2.55e-7`.

Padé approximants to the Borel transform produce a stable sequence of negative-real poles approaching the same scale.

### Resummation

The exact reference tail is obtained by numerical inversion of the affine transform on a contour through the real saddle. On

```text
epsilon in [0.04, 0.8]
```

24-coefficient Borel-Padé resummation has maximum relative error

```text
1.85e-4
```

while direct optimal truncation reaches an order-one relative error at the large-epsilon end of the interval.

### Interpretation

This is a real large-order/non-perturbative matching, but the leading singularity should not yet be called a second Heston instanton. The CDF inversion contains an explicit `1/p` pole at `p=0`. The exponent gap between the rare saddle and this pole is exactly the large-deviation rate `I(x*)`, naturally explaining the leading Borel scale at `-I`.

The next research target is therefore to remove or account for this leading obstruction, resolve subleading Borel singularities, locate secondary complex Heston saddles on the relevant analytic sheets, and test their action differences quantitatively.

## Reproduce

```bash
pip install -r requirements.txt
pytest
python experiments/run_heston_resurgence.py
```

Main outputs:

```text
figures/heston_large_order_action.png
figures/heston_borel_plane.png
figures/heston_tail_resummation.png
figures/heston_resummation_error.png
figures/heston_hamiltonian_instanton.png
results/heston_fluctuation_coefficients.csv
results/heston_large_order.csv
results/heston_pade_stability.csv
results/heston_tail_resummation.csv
results/heston_resurgence_summary.csv
paper/resurgent_stochastic_finance.pdf
```

## Secondary Heston analytic-sheet and Stokes test

The first negative real moment-explosion pole is

```text
p_pole = -6.453093949198
```

and the nearest stationary point on the adjacent logarithmic sheet is

```text
p1 = -7.998322139804
Delta S1 = 2.792240162467 + 2.482246047281 i.
```

The conjugate continuation gives the opposite imaginary sign. The imaginary action is fixed exactly by the affine logarithmic monodromy,

```text
2*pi*kappa*theta/xi^2 = 2.482246047281.
```

### 75-coefficient secondary location test

The physical-saddle coefficient generator uses high-precision Cauchy-integral Taylor extraction. Seventy-five coefficients are now stored and reproducible.

After the conformal map

```text
zeta = 4 I w / (1-w)^2,
```

the first pole of the order-75 secondary Pade condensation is

```text
zeta_BP^(2) = 2.831243366402 + 2.504669377762 i
relative gap to Delta S1 = 1.2042%.
```

A separate direct logarithmic late-order fit using orders 62--74 gives

```text
zeta_late^(2) = 2.780068703367 + 2.480740318773 i
relative gap to Delta S1 = 0.3283%.
```

The fit does not use the adjacent-saddle fluctuation coefficients.

### Adjacent-sheet fluctuation sector

The quadratic curvature at the adjacent saddle is

```text
Lambda''(p1) = -0.682102187389
steepest-descent scale a1 = 1.210807992003.
```

The first ten normalized adjacent coefficients are stored in
`results/heston_adjacent_fluctuation_coefficients_10.csv`. They begin

```text
d0 =  1
d1 =  0.279264491412
d2 = -0.006557981878
d3 =  0.114851104087
d4 =  0.224318685187
d5 =  0.493473124613
```

with local prefactor ratio

```text
C1/C0 = -0.169547214520 i.
```

A late-order branch-strength cross-check using the 75-coefficient sequence and orders 60--74 gives

```text
S_eff = -0.990549143 + 0.008669157 i
weighted NRMSE = 0.4167%.
```

### Direct lateral-discontinuity Stokes calculation

This is the main v0.4 result. The finite conformal-Pade approximant replaces the secondary logarithmic cut by a string of simple poles. If `zeta_j` and `Res_j` denote the selected pole positions and residues, the lateral Borel-Laplace discontinuity is approximated by

```text
Disc Phi0 = (2*pi*i/epsilon) sum_j exp(-zeta_j/epsilon) Res_j.
```

Placing epsilon on the Stokes ray `arg epsilon = arg DeltaS1` and dividing by the independently calculated adjacent sector gives, for 75 input coefficients and `|epsilon|=0.18`,

```text
S_lateral = -0.998565810 + 0.001484197 i
distance to -1 = 0.2064%.
```

At the same radius the estimate remains within roughly `0.2--0.3%` of `-1` as the input order is varied from 55 to 75. This computation does **not** fit the Stokes multiplier from late coefficients.

### Hyperasymptotic jump reconstruction

Define the reduced discontinuity

```text
R1 = - Disc Phi0 / [(C1/C0) exp(-DeltaS1/epsilon)].
```

If the simple connection multiplier is `S01=-1`, resurgence predicts `R1 ~ Phi1`, the adjacent fluctuation series. At `|epsilon|=0.18`:

```text
using only d0: relative error = 4.61%
using d0...d9: relative error = 0.207%
```

The improvement is asymptotic rather than monotone at larger `|epsilon|`, as expected once the adjacent series itself begins to diverge.

### Interpretation

The secondary evidence now has five mutually consistent components:

1. an adjacent analytic-sheet saddle is found independently;
2. its imaginary action is fixed by exact logarithmic monodromy;
3. a 75-coefficient late-order fit locates its singulant to `0.33%`;
4. its local fluctuation sector reproduces late coefficient phase and strength with a multiplier close to `-1`;
5. an independent lateral Pade-residue calculation gives the same Stokes multiplier to about `0.21%`, and the adjacent fluctuations reconstruct the jump to about `0.21%` at small `|epsilon|`.

The direct Borel evidence is complemented in v0.5 by an explicit Picard--Lefschetz flow calculation, described below.

### Reproduce

```bash
pytest
python experiments/run_heston_stokes_resurgence.py
python experiments/run_heston_lateral_resurgence.py --reuse-coefficients
```

Current automated tests before the v0.5 geometry test: `11/11` passing.

Main outputs:

```text
figures/heston_secondary_borel_search.png
figures/heston_secondary_stability.png
figures/heston_secondary_singulant_fit.png
figures/heston_stokes_multiplier.png
figures/heston_secondary_coefficient_prediction.png
results/heston_fluctuation_coefficients_75.csv
results/heston_conformal_borel_coefficients_75.csv
results/heston_adjacent_fluctuation_coefficients_10.csv
results/heston_secondary_pade_stability_75.csv
results/heston_secondary_singulant_fit_75.csv
results/heston_stokes_fit_75.csv
results/heston_lateral_stokes.csv
results/heston_hyperasymptotic_jump.csv
results/heston_lateral_summary.csv
figures/heston_lateral_stokes.png
figures/heston_hyperasymptotic_jump.png
figures/heston_lateral_pade_chain.png
paper/resurgent_stochastic_finance.pdf
```


## Picard--Lefschetz wall crossing — v0.5

The secondary Stokes angle is fixed by the independently computed singulant,

```text
theta_1 = arg(Delta S1) = 0.726693312478 rad.
```

For the rotated phase `h_theta(p)=exp(-i theta) [Lambda(p)-p x*]`, the upward flow is

```text
dp/dtau = conjugate(h_theta'(p)).
```

At `theta=theta_1`, one upward branch from the adjacent saddle `p1` approaches the physical saddle. With the finite integration cutoff used by the reproducible experiment,

```text
min |p-p*| = 7.9629e-5.
```

This is the expected heteroclinic connection at a Stokes wall.

To determine the connection integer, choose the upward Bromwich contour `Re p=-1`. The phase was moved on either side of the wall while following the same dual-cycle branch:

```text
delta theta = -0.05,...,-0.01 : no crossing, intersection number 0
delta theta = +0.01,...,+0.05 : exactly one crossing, intersection number -1
```

The unique crossing is from left to right while the Bromwich contour is oriented upward. Therefore

```text
det(t_Gamma_B,t_K1) < 0
<Gamma_B,K1> = -1.
```

With the paper's normalization this gives

```text
S01^PL = -1.
```

This is independent of the late-order and Borel-residue fits and explains

```text
S_lateral = -0.998565810 + 0.001484197 i
```

geometrically. The current benchmark therefore has mutually consistent action, large-order, local fluctuation, lateral-discontinuity and thimble-intersection data. The remaining limitation is that the global thimble connectivity was established numerically for this parameter point; a full analytic classification over a parameter domain remains future work.

### Reproduce the final geometry test

```bash
pytest
python experiments/run_heston_thimble_geometry.py
```

Current automated tests after the v0.6 continuation test: `14/14` passing.

New outputs:

```text
figures/heston_thimble_wall_crossing.png
figures/heston_thimble_intersection_jump.png
figures/heston_thimble_orientation.png
results/heston_thimble_wall_crossing.csv
results/heston_thimble_summary.csv
paper/resurgent_stochastic_finance.pdf
```


## Parameter continuation and robustness — v0.6

The benchmark calculation was continued along five one-parameter axes while fixing all remaining quantities at their benchmark values:

```text
x*    in [-0.35,-0.15]
T     in [0.5,2.0]
rho   in [-0.9,0]
xi    in [0.25,0.8]
kappa in [0.75,4.0]
```

Five points were evaluated per axis, for 25 structural continuation points in total.

### Exact adjacent-sheet monodromy

At every point,

```text
Im Delta S1 = 2*pi*kappa*theta/xi^2.
```

The maximum absolute discrepancy between the directly continued action and the monodromy prediction over the full sweep is

```text
9.8e-15.
```

Therefore `Im Delta S1` is invariant under changes of `x*`, `T` and `rho`, and changes with `kappa` and `xi` exactly as required by the logarithmic branch coefficient.

### Picard--Lefschetz topology

For every sweep point, the same adjacent dual cycle was followed through the secondary Stokes angle.  Using phase offsets `theta_1 +/- 0.02`, the oriented Bromwich intersection is

```text
below the wall :  0
above the wall : -1
```

at all 25 points.  Thus the sampled connected nondegenerate region has a stable connection integer

```text
S01^PL = -1.
```

This supports the continuation principle that the integer is homotopy-invariant while the adjacent saddle remains simple and no singular endpoint changes the thimble homology.

### High-order Borel test at parameter extremes

Forty physical-sector coefficients were generated at the two extremes of each of the five axes.  Comparing the nearest secondary conformal-Pade feature with the independently continued `Delta S1` gives

```text
median relative gap = 2.148%
maximum relative gap = 4.304%   at T=2.
```

All ten extreme cases retain a secondary Borel feature within `5%` of the predicted action.  The finite-order late-fit estimator is less uniform than the direct Pade-location diagnostic at some long-maturity and large-kappa points, so the latter is used as the conservative robustness metric.

### Correlation-direction caustic

Extending the `rho` continuation slightly past zero reveals a fold.  Solving

```text
Phi'(p_c;rho_c)  = 0
Phi''(p_c;rho_c) = 0
```

gives

```text
rho_c = 0.02994951567745894
p_c   = -13.470910377298026.
```

The adjacent Hessian approaches zero as `rho -> rho_c^-`.  At `rho=0`, for example,

```text
Phi''(p1) = -0.04674920675
Phi1(epsilon) = 1 + 19.533 epsilon + 2584.286 epsilon^2 + ...
```

so the ordinary Gaussian adjacent-saddle series is already poorly conditioned.  The loss of uniformity explains why a fixed-radius finite-Pade residue estimate of the Stokes amplitude becomes unreliable near this corner even though the Borel singularity location and Picard--Lefschetz intersection remain well defined.  The correct next description is a coalescing-saddle/Airy-type uniform transseries.

### Interpretation

The v0.6 result is therefore stronger and more nuanced than simple parameter insensitivity:

1. the logarithmic imaginary action is exactly universal within the continuation family;
2. the Stokes intersection integer is topologically stable on the sampled nondegenerate component;
3. the real part of the singulant deforms smoothly and remains visible in high-order Borel data;
4. the continuation identifies a codimension-one caustic where the nondegenerate saddle expansion must be reorganized rather than extrapolated blindly.

This is evidence for a **parameter-domain resurgent structure**, not a claim of unrestricted universality over all Heston parameters.

## Parameter-domain generalization — v0.6

A five-axis structural sweep was performed with five values per axis, holding the remaining benchmark parameters fixed:

```text
xstar in [-0.35,-0.15]
T     in [0.5,2.0]
rho   in [-0.9,0.0]
xi    in [0.25,0.8]
kappa in [0.75,4.0]
```

### Exact monodromy

At all 25 points,

```text
Im Delta S1 = 2*pi*kappa*theta/xi^2
```

with maximum relative discrepancy `2.38e-15` in the regenerated phase-diagram run. Thus the imaginary action is independent of `xstar`, `T` and `rho` in this adjacent-sheet regime and varies exactly linearly with `kappa` and as `xi^-2`.

### Topological continuation

At all 25 points the projected Picard--Lefschetz test gives

```text
below secondary Stokes wall : 0
above secondary Stokes wall : -1
```

so the connection integer is stable throughout the tested connected nondegenerate domain.

### Independent high-order tracking

At the ten extreme points of the five axes, a 40-coefficient conformal-Borel calculation gives

```text
median relative secondary-location gap = 2.15%
maximum relative secondary-location gap = 4.30%
```

relative to the independently continued adjacent action. The lower-order 32-term exploratory phase-diagram run gives a median gap of `3.39%` and a maximum of `5.01%`; the paper uses the reproducible 40-term extreme-point numbers for its quantitative summary.

### Correlation fold and complex continuation

The first double-stationary point along the benchmark correlation slice satisfies

```text
Phi'(p_c)=0
Phi''(p_c)=0
rho_c = 0.0299495156775
p_c   = -13.4709103772980
```

Below the fold there are two real adjacent stationary points. At the fold they coalesce, and above it they continue as a complex-conjugate pair. This is a caustic of the separate Gaussian saddle expansions and calls for a uniform Airy-type local approximation.

### Anti-Stokes boundaries

For positive real epsilon, the relative weight of the adjacent sector is controlled by `exp(-DeltaS1/epsilon)`. Equal exponential magnitude occurs at

```text
Re Delta S1 = 0.
```

The first crossings along two benchmark slices are

```text
T_A     = 2.67815584420614
kappa_A = 5.47078861767783
```

For parameter values beyond these crossings, the original physical-sector hierarchy is no longer the natural dominant ordering and the transseries basis must be reorganized.

### Generalized interpretation

The data support a local resurgent phase diagram:

```text
connected nondegenerate domain
    -> smooth Delta S1
    -> exact logarithmic monodromy
    -> stable S01 = -1

fold / caustic: Phi'=Phi''=0
    -> coalescing saddles
    -> Airy-type uniform transseries

anti-Stokes: Re Delta S1=0
    -> equal exponential magnitude
    -> change of dominant-sector basis
```

This is stronger than benchmark robustness but weaker than a global theorem: a complete classification still requires the full complex Heston moment-pole Riemann surface and higher-dimensional boundary surfaces.


## v0.7 — Uniform Airy fold and 2D `(T,rho)` atlas

The correlation caustic is now resolved with a local Chester--Friedman--Ursell cubic normal form. For the benchmark slice,

```text
rho_c = 0.02994951567745894
p_c   = -13.470910377298026
Phi_p,rho = -0.3552648106228952
Phi_ppp   = -0.06862874837952305
alpha = -2 Phi_p,rho / Phi_ppp = -10.3532358963695
sqrt(-alpha) = 3.21764446394712
```

Hence

```text
(p-p_c)^2 ~ alpha (rho-rho_c),
|p_+-p_-| ~ 2*3.21764446394712*(rho_c-rho)^(1/2),
|Phi_+-Phi_-| ~ 1.52415446818131*(rho_c-rho)^(3/2).
```

The Airy coordinate is

```text
zeta = -1.09326849050090*(rho-rho_c),
Z = zeta / epsilon^(2/3).
```

Direct high-precision Heston roots reproduce the square-root saddle separation and three-halves action law. The Airy-uniform canonical amplitude remains finite at the fold, while the isolated Gaussian-saddle asymptotic diverges as the Hessian vanishes.

A separate 2D scan defines the adjacent sector invariantly as the first stationary point between the first two negative moment poles at each real `(T,rho)`. On the grid:

```text
resolved parameter points = 763
max relative monodromy error = 1.79e-16
coarse anti-Stokes crossing at rho=-0.7: T ~= 2.67749
refined one-dimensional value:             T = 2.678155844206
```

The 2D atlas maps:

- the caustic curve `Phi_p = Phi_pp = 0`;
- the anti-Stokes curve `Re DeltaS1 = 0`;
- the Stokes-ray angle field `theta_S = arg DeltaS1`.

Inside the plotted real-adjacent domain `Im DeltaS1 = 2*pi*kappa*theta/xi^2 > 0`, so there is no positive-real-epsilon Stokes curve there; instead the Stokes ray rotates smoothly in the complex epsilon plane. At the caustic, the separate two-saddle transseries must be replaced by the Airy-uniform sector.

## v0.8 — Post-caustic complex-saddle continuation

The first adjacent pair has now been continued across the correlation fold on a common logarithmic sheet. For the benchmark slice,

```text
rho_c = 0.02994951567745894
p_c   = -13.470910377298026
M     = 2*pi*kappa*theta/xi^2 = 2.482246047280824...
```

For `rho > rho_c`, the stationary points form a complex-conjugate pair in `p`. The lower principal action must be lifted by `2 i M`, after which the two same-sheet actions remain centred exactly at `Im DeltaS = M`. In the tested continuation strip,

```text
max monodromy-centre error            = 0
max |Im zeta_CFU| after the fold      = 2.10e-17
```

The exact action difference defines a CFU coordinate that is positive before the fold and negative after it. Its local limit agrees with the derivative-based Airy coordinate.

### Airy re-expansion

The canonical uniform block obeys the standard two limits

```text
Ai(Z)  -> one decaying exponential,              Z > 0
Ai(-X) -> coherent complex-conjugate saddle pair, X > 0.
```

At `epsilon=0.004`, comparing the exact canonical Airy block with the leading saddle re-expansion gives

```text
max normalized error for |Z| >= 4 = 1.2463%
```

so the same uniform object quantitatively connects the pre-caustic and post-caustic saddle descriptions away from the central fold layer.

### Complex local fluctuations

At `rho=0.05`, the two projected complex saddles have conjugate local fluctuation sectors. Through the first six stored coefficients,

```text
max |d_n^- - conjugate(d_n^+)| = 0   (at stored precision).
```

For example,

```text
d_1^+ = -2.87635671842 + 32.1669377615 i
d_1^- = -2.87635671842 - 32.1669377615 i.
```

Their rapid growth near the caustic is expected from the loss of uniformity of isolated-saddle expansions.

### Post-caustic Borel tracking

With 32 physical-sector coefficients:

```text
rho     Pade location error     late-order pair-fit error
0.04       5.99%                    1.93%
0.05       5.88%                    1.33%
0.08       6.27%                    1.45%
```

The late-order fit uses only orders 20--31 of the conformal-Borel coefficients and treats the complex singulant as a fitted quantity. This provides a stronger post-caustic check than identifying the nearest finite-order Pade pole alone.

The combined continuation now reaches from two real adjacent saddles, through the Airy-uniform caustic, to a complex-conjugate pair whose action positions and local fluctuation data remain visible in the perturbative sector.
