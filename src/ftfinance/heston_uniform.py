from dataclasses import dataclass
import cmath
import math

import mpmath as mp
import numpy as np
from scipy.special import airy

from .heston import HestonParams
from .heston_resurgence import _heston_scaled_cgf_mp


@dataclass(frozen=True)
class HestonRhoFold:
    threshold: float
    T: float
    rho_c: float
    p_c: float
    phi_c: complex
    phi_rho: complex
    phi_prho: float
    phi_ppp: float
    alpha: float
    split_coefficient: float
    airy_slope: float
    action_split_coefficient: float


def _real_cuberoot(x):
    x = float(x)
    return math.copysign(abs(x) ** (1.0 / 3.0), x)


def find_heston_rho_fold(base_params, threshold, T=1.0, guess=(-13.47, 0.03), dps=70):
    """Locate the rho fold where two adjacent stationary points coalesce.

    The fold solves Phi_p=0 and Phi_pp=0 while rho is varied, with
    Phi(p,rho)=Lambda(p,rho)-p*threshold.  The returned derivatives define the
    local cubic normal form

        Phi = Phi_c + Phi_rho dr + Phi_prho dr q + Phi_ppp q^3/6 + ... .

    Consequently q^2 = alpha*dr with alpha=-2 Phi_prho/Phi_ppp and the
    uniform Airy coordinate is zeta=airy_slope*dr.
    """
    old = mp.mp.dps
    mp.mp.dps = int(dps)
    try:
        x = mp.mpf(str(threshold))
        Tm = mp.mpf(str(T))
        mu = float(base_params.mu)
        kappa = float(base_params.kappa)
        theta = float(base_params.theta)
        xi = float(base_params.xi)
        v0 = float(base_params.v0)

        def params_at(rho):
            return HestonParams(mu=mu, kappa=kappa, theta=theta, xi=xi, rho=float(mp.re(rho)), v0=v0)

        # mp differentiation needs rho to remain an mp variable, so reproduce
        # the affine expression here with rho left symbolic.
        km = mp.mpf(str(kappa)); th = mp.mpf(str(theta)); xim = mp.mpf(str(xi))
        v0m = mp.mpf(str(v0)); mum = mp.mpf(str(mu))
        def lam(p, rho):
            B = km - rho * xim * p
            gamma = mp.sqrt(B * B - xim * xim * (p * p - p))
            g = (B - gamma) / (B + gamma)
            e = mp.exp(-gamma * Tm)
            D = (B - gamma) / (xim * xim) * (1 - e) / (1 - g * e)
            C = mum * p * Tm + (km * th / (xim * xim)) * ((B - gamma) * Tm - 2 * mp.log((1 - g * e) / (1 - g)))
            return C + v0m * D

        def F(p, rho):
            return mp.diff(lambda q: lam(q, rho), p) - x

        def Fp(p, rho):
            return mp.diff(lambda q: lam(q, rho), p, 2)

        p0 = mp.mpf(str(guess[0])); r0 = mp.mpf(str(guess[1]))
        tol = mp.power(10, -min(40, max(20, int(dps)-15)))
        pc, rc = mp.findroot((F, Fp), (p0, r0), tol=tol, maxsteps=100)
        phi = lambda p, rho: lam(p, rho) - p * x
        phi_c = phi(pc, rc)
        phi_rho = mp.diff(lambda r: phi(pc, r), rc)
        phi_prho = mp.diff(lambda r: F(pc, r), rc)
        phi_ppp = mp.diff(lambda q: lam(q, rc), pc, 3)
        alpha = -2 * phi_prho / phi_ppp

        # In the benchmark real-fold orientation alpha<0: real saddles occur
        # for dr<0.  split_coefficient multiplies sqrt(-dr).
        split_coefficient = mp.sqrt(-alpha)
        A = phi_ppp / 2
        A13 = mp.sign(mp.re(A)) * abs(A) ** (mp.mpf(1) / 3)
        airy_slope = -phi_prho / A13
        action_split_coefficient = (mp.mpf(4) / 3) * abs(phi_prho) * split_coefficient
    finally:
        mp.mp.dps = old

    return HestonRhoFold(
        threshold=float(threshold), T=float(T), rho_c=float(mp.re(rc)), p_c=float(mp.re(pc)),
        phi_c=complex(phi_c), phi_rho=complex(phi_rho), phi_prho=float(mp.re(phi_prho)),
        phi_ppp=float(mp.re(phi_ppp)), alpha=float(mp.re(alpha)),
        split_coefficient=float(mp.re(split_coefficient)), airy_slope=float(mp.re(airy_slope)),
        action_split_coefficient=float(mp.re(action_split_coefficient)),
    )


def heston_fold_airy_coordinate(rho, epsilon, fold):
    """Return z=zeta/epsilon^(2/3) for the local Airy normal form."""
    epsilon = float(epsilon)
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    zeta = fold.airy_slope * (float(rho) - fold.rho_c)
    return zeta / (epsilon ** (2.0 / 3.0))


def heston_fold_uniform_shape(rho, epsilon, fold):
    """Leading canonical Airy shape, with the common exponential removed.

    This is epsilon^(1/3) Ai(zeta/epsilon^(2/3)).  It is the universal finite
    object replacing the singular Gaussian saddle prefactor at the fold.
    """
    z = heston_fold_airy_coordinate(rho, epsilon, fold)
    Ai, Aip, _, _ = airy(z)
    return epsilon ** (1.0 / 3.0) * Ai, epsilon ** (2.0 / 3.0) * Aip, z


def airy_one_saddle_asymptotic(z):
    """Leading Ai(z) asymptotic for positive real z."""
    z = np.asarray(z, dtype=float)
    out = np.full_like(z, np.nan, dtype=float)
    mask = z > 0
    out[mask] = np.exp(-(2.0 / 3.0) * z[mask] ** 1.5) / (2.0 * np.sqrt(np.pi) * z[mask] ** 0.25)
    return out


def airy_two_saddle_asymptotic(z):
    """Leading two-saddle Airy asymptotic on the oscillatory side.

    For negative real ``z=-x`` this returns

        Ai(-x) ~ pi^(-1/2) x^(-1/4) sin(2 x^(3/2)/3 + pi/4),

    i.e. the coherent sum of the two complex-conjugate saddle
    exponentials. Values with z>=0 are returned as NaN.
    """
    z = np.asarray(z, dtype=float)
    out = np.full_like(z, np.nan, dtype=float)
    mask = z < 0
    x = -z[mask]
    phase = (2.0 / 3.0) * x ** 1.5 + np.pi / 4.0
    out[mask] = np.sin(phase) / (np.sqrt(np.pi) * x ** 0.25)
    return out


def heston_fold_uniform_asymptotic_shape(rho, epsilon, fold):
    """Leading large-|Z| saddle re-expansion of the Airy fold block.

    On the real-saddle side (Z>0) this is the one-decaying-saddle Airy
    asymptotic. On the post-caustic side (Z<0) it is the interference of
    two complex-conjugate saddle exponentials.  The common Heston
    exponential and analytic CFU amplitude are stripped off, exactly as in
    :func:`heston_fold_uniform_shape`.
    """
    epsilon = float(epsilon)
    z = heston_fold_airy_coordinate(rho, epsilon, fold)
    if z > 0:
        value = float(airy_one_saddle_asymptotic(np.asarray([z]))[0])
        regime = "one-real-saddle"
    elif z < 0:
        value = float(airy_two_saddle_asymptotic(np.asarray([z]))[0])
        regime = "two-complex-saddles"
    else:
        value = float('nan')
        regime = "fold"
    return epsilon ** (1.0 / 3.0) * value, z, regime

@dataclass(frozen=True)
class HestonAdjacentPair:
    """First two adjacent stationary points continued through the rho fold.

    ``singulant_1`` and ``singulant_2`` live on one continuously chosen
    logarithmic sheet.  Before the fold both saddles are real and share the
    same logarithmic monodromy.  Beyond the fold they form a complex-conjugate
    pair in p while their actions remain on the same lifted logarithmic sheet.
    """
    rho: float
    p1: complex
    p2: complex
    singulant_1: complex
    singulant_2: complex
    cfu_zeta: complex
    average_singulant: complex
    monodromy: float
    regime: str


def _heston_complex_stationary_root(params, threshold, seed, T=1.0, dps=70):
    old = mp.mp.dps
    mp.mp.dps = int(dps)
    try:
        x = mp.mpf(str(threshold))
        seed = mp.mpc(seed)
        deriv = lambda p: mp.diff(lambda q: _heston_scaled_cgf_mp(q, params, T=T), p)
        root = mp.findroot(lambda p: deriv(p) - x, seed, solver="muller", tol=mp.power(10, -min(35, int(dps)//2)), maxsteps=100)
        return complex(root)
    finally:
        mp.mp.dps = old


def _heston_singulant_at(params, threshold, physical_p, saddle_p, T=1.0, dps=70):
    old = mp.mp.dps
    mp.mp.dps = int(dps)
    try:
        x = mp.mpf(str(threshold))
        p0 = mp.mpc(physical_p)
        p = mp.mpc(saddle_p)
        phi0 = _heston_scaled_cgf_mp(p0, params, T=T) - p0 * x
        phi = _heston_scaled_cgf_mp(p, params, T=T) - p * x
        return complex(phi0 - phi)
    finally:
        mp.mp.dps = old


def _cfu_zeta_from_singulant_pair(delta1, delta2, target=None):
    """CFU control coordinate from the two saddle actions.

    With the labeling used here the canonical phase difference obeys

        Phi(p1)-Phi(p2) = Delta2-Delta1 = 4 zeta^(3/2) / 3.

    The two-thirds power has three branches.  ``target`` selects the branch
    continuously connected to the local real fold coordinate.
    """
    value = 0.75 * (complex(delta2) - complex(delta1))
    if abs(value) == 0:
        return 0j
    r = abs(value) ** (2.0 / 3.0)
    arg = cmath.phase(value)
    candidates = [r * cmath.exp(1j * (2.0 / 3.0) * (arg + 2.0 * math.pi * k)) for k in range(3)]
    if target is None:
        return min(candidates, key=lambda z: abs(z.imag))
    return min(candidates, key=lambda z: abs(z - complex(target)))


def heston_adjacent_pair_through_fold(params, threshold, T=1.0, fold=None, dps=75):
    """Continue the first two adjacent Heston saddles through the rho caustic.

    The local fold provides seeds for both saddles.  For rho<rho_c the roots
    are real. For rho>rho_c they continue as a complex-conjugate pair.

    The closed-form affine CGF is logarithmic. On the post-caustic side the
    lower principal-branch singulant is shifted upward by one full logarithmic
    monodromy, ``4*pi*i*kappa*theta/xi**2``, so that both action branches are
    continuous continuations of the pre-caustic adjacent-sheet saddles.
    """
    from .heston_resurgence import find_heston_tail_saddle

    if fold is None:
        fold = find_heston_rho_fold(params, threshold, T=T, guess=(-13.5, 0.03), dps=max(55, dps-10))
    rho = float(params.rho)
    dr = rho - fold.rho_c
    # q^2 = alpha dr.  Use the square-root branch that sends the first
    # (less-negative) real saddle into the upper half p-plane after the fold.
    q2 = complex(fold.alpha * dr)
    q = cmath.sqrt(q2)
    if dr < 0 and q.real < 0:
        q = -q
    if dr > 0 and q.imag < 0:
        q = -q
    scale = 1.0 if abs(q) > 1e-5 else 1e-3
    seed1 = fold.p_c + (q if abs(q) > 1e-8 else scale)
    seed2 = fold.p_c - (q if abs(q) > 1e-8 else scale)
    if dr <= 0:
        # On the real side of the fold a bracketing scan is much more robust
        # than two Newton/Muller seeds, which can both fall into the same root
        # close to coalescence.
        from scipy.optimize import brentq
        from .heston_resurgence import heston_negative_moment_poles, _real_cgf_derivative
        pole1, pole2 = heston_negative_moment_poles(params, T=T, n_poles=2, p_min=-120.0)
        margin = max(2e-4, 2e-4 * abs(pole1-pole2))
        grid = np.linspace(pole1-margin, pole2+margin, 4000)
        roots = []
        previous = None
        for xx in grid:
            try:
                vv = _real_cgf_derivative(float(xx), params, T=T) - float(threshold)
            except Exception:
                previous = None
                continue
            if not np.isfinite(vv) or abs(vv) > 2e5:
                previous = None
                continue
            if previous is not None and previous[1] * vv < 0:
                aa, bb = previous[0], float(xx)
                rr = brentq(lambda z: _real_cgf_derivative(z, params, T=T)-float(threshold), aa, bb, xtol=1e-12, rtol=1e-12)
                if not roots or abs(rr-roots[-1]) > 1e-5:
                    roots.append(float(rr))
            previous = (float(xx), float(vv))
        if len(roots) < 2:
            # Extremely close to the caustic, use the local fold roots.
            p1 = _heston_complex_stationary_root(params, threshold, seed1, T=T, dps=dps)
            p2 = _heston_complex_stationary_root(params, threshold, seed2, T=T, dps=dps)
            roots = sorted([p1.real, p2.real], reverse=True)
        p1, p2 = complex(roots[0]), complex(roots[1])
    else:
        p1 = _heston_complex_stationary_root(params, threshold, seed1, T=T, dps=dps)
        p2 = _heston_complex_stationary_root(params, threshold, seed2, T=T, dps=dps)
        if p1.imag < p2.imag:
            p1, p2 = p2, p1

    physical = find_heston_tail_saddle(params, threshold, T=T, dps=max(55, dps-15))
    d1 = _heston_singulant_at(params, threshold, physical.p_star, p1, T=T, dps=dps)
    d2 = _heston_singulant_at(params, threshold, physical.p_star, p2, T=T, dps=dps)
    monodromy = 2.0 * math.pi * float(params.kappa) * float(params.theta) / float(params.xi) ** 2

    if dr <= 0:
        # Real adjacent saddles should live on the upper adjacent logarithmic
        # sheet.  Numerical approach from the opposite side can conjugate the
        # principal log, so choose the upper representative deterministically.
        if d1.imag < 0:
            d1 = d1.conjugate()
        if d2.imag < 0:
            d2 = d2.conjugate()
        regime = "two-real-adjacent-saddles"
    else:
        # p1 (upper p-plane) naturally lands near +M; p2 lands near -M on the
        # principal branch. Lift p2 by one full log monodromy 2M so both are
        # continuous around the common level M.
        if d1.imag < 0:
            d1 = d1.conjugate()
        if d2.imag > 0:
            d2 = d2.conjugate()
        d2 = d2 + 2j * monodromy
        regime = "complex-conjugate-adjacent-saddles"

    target = fold.airy_slope * dr
    zeta = _cfu_zeta_from_singulant_pair(d1, d2, target=target)
    average = 0.5 * (d1 + d2)
    return HestonAdjacentPair(
        rho=rho, p1=complex(p1), p2=complex(p2), singulant_1=complex(d1), singulant_2=complex(d2),
        cfu_zeta=complex(zeta), average_singulant=complex(average), monodromy=float(monodromy), regime=regime,
    )


def airy_stokes_matrices():
    """Canonical Airy Stokes matrices in WKB and oriented-thimble bases.

    For the WKB basis (growing, decaying), a standard simple-turning-point
    normalization has alternating triangular matrices with multiplier ``i``.
    Removing the Gaussian orientation phases converts the multiplier to the
    integer Picard-Lefschetz intersection number.  The exact placement of the
    off-diagonal entry depends on which exponential is dominant on a ray; the
    two elementary matrices below are the pair needed around one Airy fold.
    """
    wkb_upper = np.asarray([[1.0 + 0j, 1j], [0j, 1.0 + 0j]])
    wkb_lower = np.asarray([[1.0 + 0j, 0j], [1j, 1.0 + 0j]])
    thimble_upper = np.asarray([[1, 1], [0, 1]], dtype=int)
    thimble_lower = np.asarray([[1, 0], [1, 1]], dtype=int)
    return {
        "wkb_upper": wkb_upper,
        "wkb_lower": wkb_lower,
        "thimble_upper": thimble_upper,
        "thimble_lower": thimble_lower,
    }
