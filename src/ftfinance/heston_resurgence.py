from dataclasses import dataclass
import cmath
import math

import mpmath as mp
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

from .resurgence import borel_pade, borel_pade_sum, large_order_action_estimates


@dataclass(frozen=True)
class HestonTailSaddle:
    threshold: float
    T: float
    p_star: float
    rate: float
    lambda_second: float


def heston_scaled_cgf(p, params, T=1.0):
    """Scaled cumulant generating function for the uniform small-noise Heston model.

    For

        dX = (mu - V/2) dt + sqrt(epsilon V) dB,
        dV = kappa(theta-V) dt + sqrt(epsilon) xi sqrt(V) dW,

    one has exactly

        E[exp(p X_T / epsilon)] = exp(Lambda(p) / epsilon).

    The returned Lambda is independent of epsilon. The implementation uses the
    closed-form affine Riccati solution and accepts real or complex p.
    """
    p = complex(p)
    kappa = float(params.kappa)
    theta = float(params.theta)
    xi = float(params.xi)
    rho = float(params.rho)
    v0 = float(params.v0)
    mu = float(params.mu)

    B = kappa - rho * xi * p
    gamma = cmath.sqrt(B * B - xi * xi * (p * p - p))
    if gamma.real < 0.0 or (abs(gamma.real) < 1e-15 and gamma.imag < 0.0):
        gamma = -gamma

    g = (B - gamma) / (B + gamma)
    exp_term = cmath.exp(-gamma * T)
    D = (B - gamma) / (xi * xi) * (1.0 - exp_term) / (1.0 - g * exp_term)
    C = mu * p * T + (kappa * theta / (xi * xi)) * (
        (B - gamma) * T - 2.0 * cmath.log((1.0 - g * exp_term) / (1.0 - g))
    )
    return C + v0 * D


def _heston_scaled_cgf_mp(p, params, T=1.0):
    kappa = mp.mpf(str(params.kappa))
    theta = mp.mpf(str(params.theta))
    xi = mp.mpf(str(params.xi))
    rho = mp.mpf(str(params.rho))
    v0 = mp.mpf(str(params.v0))
    mu = mp.mpf(str(params.mu))
    T = mp.mpf(str(T))

    B = kappa - rho * xi * p
    gamma = mp.sqrt(B * B - xi * xi * (p * p - p))
    g = (B - gamma) / (B + gamma)
    exp_term = mp.exp(-gamma * T)
    D = (B - gamma) / (xi * xi) * (1 - exp_term) / (1 - g * exp_term)
    C = mu * p * T + (kappa * theta / (xi * xi)) * (
        (B - gamma) * T - 2 * mp.log((1 - g * exp_term) / (1 - g))
    )
    return C + v0 * D


def _real_cgf_derivative(p, params, T=1.0):
    scale = max(1.0, abs(float(p)))
    h = 2e-5 * scale
    f2p = heston_scaled_cgf(p + 2.0 * h, params, T=T).real
    fp = heston_scaled_cgf(p + h, params, T=T).real
    fm = heston_scaled_cgf(p - h, params, T=T).real
    f2m = heston_scaled_cgf(p - 2.0 * h, params, T=T).real
    return (-f2p + 8.0 * fp - 8.0 * fm + f2m) / (12.0 * h)


def find_heston_tail_saddle(params, threshold, T=1.0, dps=80):
    """Find the real saddle p* for a left-tail event X_T <= threshold.

    The saddle solves Lambda'(p*) = threshold with p* < 0. Its rate is

        I(x) = p* x - Lambda(p*).
    """
    typical = _real_cgf_derivative(0.0, params, T=T)
    if threshold >= typical:
        raise ValueError(f"threshold must lie below the deterministic endpoint {typical:.8g}")

    right = -1e-8
    f_right = _real_cgf_derivative(right, params, T=T) - threshold
    left = -0.25
    f_left = _real_cgf_derivative(left, params, T=T) - threshold
    while f_left > 0.0:
        left *= 1.5
        if left < -50.0:
            raise RuntimeError("could not bracket the real Heston tail saddle before the affine moment boundary")
        f_left = _real_cgf_derivative(left, params, T=T) - threshold

    p_guess = brentq(lambda p: _real_cgf_derivative(p, params, T=T) - threshold, left, right, xtol=1e-13, rtol=1e-13)

    old_dps = mp.mp.dps
    mp.mp.dps = dps
    try:
        p_mp = mp.mpf(str(p_guess))
        threshold_mp = mp.mpf(str(threshold))
        derivative = lambda p: mp.diff(lambda q: _heston_scaled_cgf_mp(q, params, T=T), p)
        p_star = mp.findroot(lambda p: derivative(p) - threshold_mp, p_mp, solver="newton", tol=mp.mpf("1e-45"))
        lam = _heston_scaled_cgf_mp(p_star, params, T=T)
        rate = p_star * threshold_mp - lam
        lambda_second = mp.diff(lambda q: _heston_scaled_cgf_mp(q, params, T=T), p_star, 2)
    finally:
        mp.mp.dps = old_dps

    if abs(mp.im(rate)) > mp.mpf("1e-30") or abs(mp.im(lambda_second)) > mp.mpf("1e-30"):
        raise RuntimeError("real saddle unexpectedly produced a complex rate or Hessian")

    return HestonTailSaddle(
        threshold=float(threshold),
        T=float(T),
        p_star=float(mp.re(p_star)),
        rate=float(mp.re(rate)),
        lambda_second=float(mp.re(lambda_second)),
    )


def _cauchy_taylor_coefficients_mp(func, center, order, radius=0.5, samples=None):
    """High-precision Taylor coefficients from a discrete Cauchy integral.

    This is substantially faster than repeated high-order numerical
    differentiation for the Heston affine transform.  The contour is kept
    local to the saddle so the principal closed-form branch remains analytic
    on the sampling circle.
    """
    if order < 0:
        raise ValueError("order must be nonnegative")
    radius = mp.mpf(str(radius))
    if radius <= 0:
        raise ValueError("radius must be positive")
    if samples is None:
        samples = max(256, 5 * (order + 1))
    samples = int(samples)

    values = []
    roots = []
    for j in range(samples):
        angle = 2 * mp.pi * j / samples
        root = mp.e ** (mp.j * angle)
        roots.append(root)
        values.append(func(center + radius * root))

    coefficients = []
    for k in range(order + 1):
        total = mp.mpc(0)
        for value, root in zip(values, roots):
            total += value * root ** (-k)
        coefficients.append(total / samples / (radius ** k))
    return coefficients


def _normal_moment(order):
    if order % 2:
        return mp.mpf("0")
    m = order // 2
    return mp.factorial(2 * m) / (mp.power(2, m) * mp.factorial(m))


def heston_tail_fluctuation_coefficients(params, threshold, T=1.0, n_terms=24, dps=100, saddle=None, return_mpmath=False):
    """Generate the saddle-point fluctuation series for the Heston left tail.

    The tail has the form

        P(X_T <= x) ~ exp(-I/eps) * (-1/p*) * sqrt(eps/(2 pi Lambda''(p*)))
                       * sum_{n>=0} c_n eps^n.

    Coefficients are generated by a formal expansion of the exact Bromwich
    integral around the real saddle. The algebra is carried out at arbitrary
    precision because c_n grows factorially.
    """
    if n_terms < 1:
        raise ValueError("n_terms must be positive")
    if saddle is None:
        saddle = find_heston_tail_saddle(params, threshold, T=T, dps=max(70, dps // 2))

    old_dps = mp.mp.dps
    mp.mp.dps = dps
    try:
        p0 = mp.mpf(str(saddle.p_star))
        max_s_power = 2 * (n_terms - 1)
        max_lambda_order = max_s_power + 2
        lam_taylor = _cauchy_taylor_coefficients_mp(
            lambda p: _heston_scaled_cgf_mp(p, params, T=T),
            p0,
            max_lambda_order,
            radius=0.25,
        )
        lambda_second = 2 * lam_taylor[2]
        sqrt_lambda_second = mp.sqrt(lambda_second)

        # exp(Q(s,z)) = sum_m E_m(z) s^m, where s=sqrt(epsilon).
        # Q_m is a single monomial z^(m+2), which lets us build the formal
        # exponential efficiently without symbolic-algebra dependencies.
        exp_series = [[mp.mpc(1)]]
        q_coeff = [None] * (max_s_power + 1)
        for m in range(1, max_s_power + 1):
            k = m + 2
            q_coeff[m] = lam_taylor[k] * (mp.j ** k) / (lambda_second ** (mp.mpf(k) / 2))

        for n in range(1, max_s_power + 1):
            poly = [mp.mpc(0)] * (3 * n + 1)
            for m in range(1, n + 1):
                previous = exp_series[n - m]
                shift = m + 2
                factor = (mp.mpf(m) / n) * q_coeff[m]
                needed = len(previous) + shift
                if needed > len(poly):
                    poly.extend([mp.mpc(0)] * (needed - len(poly)))
                for degree, value in enumerate(previous):
                    poly[degree + shift] += factor * value
            while len(poly) > 1 and poly[-1] == 0:
                poly.pop()
            exp_series.append(poly)

        amplitude_base = -mp.j / (p0 * sqrt_lambda_second)
        half_power_coeffs = []
        for n in range(max_s_power + 1):
            expectation = mp.mpc(0)
            for j in range(n + 1):
                amplitude = amplitude_base ** j
                poly = exp_series[n - j]
                for degree, value in enumerate(poly):
                    total_degree = degree + j
                    if total_degree % 2 == 0:
                        expectation += amplitude * value * _normal_moment(total_degree)
            half_power_coeffs.append(expectation)

        coeffs = []
        for n in range(n_terms):
            value = half_power_coeffs[2 * n]
            reality_tol = mp.mpf("1e-12")
            if abs(mp.im(value)) > reality_tol * max(1, abs(mp.re(value))):
                raise RuntimeError("saddle expansion did not reduce to a real integer-power series")
            coeffs.append(mp.mpf(mp.re(value)) if return_mpmath else float(mp.re(value)))
    finally:
        mp.mp.dps = old_dps

    if return_mpmath:
        return coeffs, saddle
    return np.asarray(coeffs, dtype=float), saddle


def heston_tail_leading_prefactor(epsilon, saddle):
    epsilon = float(epsilon)
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    return (
        math.exp(-saddle.rate / epsilon)
        * (-1.0 / saddle.p_star)
        * math.sqrt(epsilon / (2.0 * math.pi * saddle.lambda_second))
    )


def heston_tail_exact(epsilon, params, threshold, T=1.0, saddle=None, z_max=30.0):
    """Numerical affine-transform inversion of the small-noise Heston left tail.

    The contour is shifted through the real saddle and rescaled by sqrt(eps),
    which keeps the quadrature stable even when the probability is exponentially
    small. This is exact up to numerical quadrature and affine-branch precision.
    """
    epsilon = float(epsilon)
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if saddle is None:
        saddle = find_heston_tail_saddle(params, threshold, T=T)

    p0 = saddle.p_star
    scale = math.sqrt(epsilon / saddle.lambda_second)
    phi0 = -saddle.rate

    def integrand(z):
        p = p0 + 1j * scale * z
        delta_phi = heston_scaled_cgf(p, params, T=T) - p * threshold - phi0
        value = cmath.exp(delta_phi / epsilon) / p
        return value.real

    integral, _ = quad(integrand, 0.0, z_max, epsabs=2e-11, epsrel=2e-10, limit=600)
    normalized_fluctuation = p0 * math.sqrt(2.0 / math.pi) * integral
    probability = heston_tail_leading_prefactor(epsilon, saddle) * normalized_fluctuation
    return probability, normalized_fluctuation


def heston_tail_optimal_truncation(epsilon, coeffs):
    epsilon = float(epsilon)
    powers = epsilon ** np.arange(len(coeffs))
    terms = np.abs(coeffs * powers)
    n_terms = int(np.argmin(terms) + 1)
    return float(np.dot(coeffs[:n_terms], powers[:n_terms])), n_terms


def heston_tail_borel_pade(epsilon, coeffs, saddle):
    fluctuation = borel_pade_sum(float(epsilon), coeffs)
    return heston_tail_leading_prefactor(epsilon, saddle) * fluctuation


def heston_tail_borel_pade_fluctuation(epsilon, coeffs):
    return borel_pade_sum(float(epsilon), coeffs)


def heston_large_order_extrapolation(coeffs, start_order=10, degree=3):
    """Extrapolate n c_(n-1)/c_n to n=infinity as a polynomial in 1/n."""
    estimates = large_order_action_estimates(coeffs)
    first = max(1, int(start_order))
    n = np.arange(first, len(coeffs), dtype=float)
    values = estimates[first - 1 :]
    if len(values) <= degree:
        raise ValueError("not enough coefficients for the requested extrapolation")
    fit = np.polyfit(1.0 / n, values, degree)
    return float(np.polyval(fit, 0.0)), estimates


def heston_borel_poles(coeffs, denominator_degree=None):
    numerator, denominator = borel_pade(coeffs, denominator_degree=denominator_degree)
    return np.roots(denominator), numerator, denominator


def heston_negative_moment_pole(params, T=1.0, bracket=(-7.0, -6.0)):
    """Locate the first negative real Riccati/moment-explosion pole.

    For the closed-form Riccati solution this is a zero of
    ``1 - g exp(-gamma T)``.  Along the real axis beyond the discriminant
    turning point the denominator has unit-modulus phase structure, so a
    scalar zero of its imaginary part identifies the pole (the real part
    vanishes simultaneously to numerical precision).
    """
    from scipy.optimize import brentq

    def denominator(p):
        p = complex(p)
        kappa = float(params.kappa)
        xi = float(params.xi)
        rho = float(params.rho)
        B = kappa - rho * xi * p
        gamma = cmath.sqrt(B * B - xi * xi * (p * p - p))
        if gamma.real < 0.0 or (abs(gamma.real) < 1e-15 and gamma.imag < 0.0):
            gamma = -gamma
        g = (B - gamma) / (B + gamma)
        return 1.0 - g * cmath.exp(-gamma * T)

    pole = brentq(lambda q: denominator(q).imag, float(bracket[0]), float(bracket[1]), xtol=1e-14, rtol=1e-14)
    residual = denominator(pole)
    if abs(residual) > 5e-8:
        raise RuntimeError("moment-pole denominator did not vanish to the requested accuracy")
    return float(pole)



def heston_negative_moment_poles(params, T=1.0, n_poles=2, p_min=-60.0, samples=24000):
    """Locate the first ``n_poles`` negative real Heston moment poles.

    The poles are zeros of the affine Riccati denominator

        1 - g(p) exp(-gamma(p) T).

    On the oscillatory part of the negative real axis its real and imaginary
    parts vanish simultaneously at the moment-explosion points.  We scan for
    sign changes of the imaginary part and retain only roots whose full
    complex residual is small.  Results are ordered from the origin outward.
    """
    from scipy.optimize import brentq

    n_poles = int(n_poles)
    if n_poles < 1:
        raise ValueError("n_poles must be positive")
    p_min = float(p_min)
    if p_min >= -0.2:
        raise ValueError("p_min must extend along the negative real axis")

    def denominator(p):
        p = complex(p)
        kappa = float(params.kappa)
        xi = float(params.xi)
        rho = float(params.rho)
        B = kappa - rho * xi * p
        gamma = cmath.sqrt(B * B - xi * xi * (p * p - p))
        if gamma.real < 0.0 or (abs(gamma.real) < 1e-15 and gamma.imag < 0.0):
            gamma = -gamma
        g = (B - gamma) / (B + gamma)
        return 1.0 - g * cmath.exp(-gamma * T)

    grid = np.linspace(-0.25, p_min, int(samples))
    imag = np.asarray([denominator(q).imag for q in grid])
    roots = []
    for i in range(len(grid) - 1):
        fa, fb = imag[i], imag[i + 1]
        if abs(fa) < 1e-12 or abs(fb) < 1e-12:
            continue
        if fa * fb >= 0.0:
            continue
        a, b = grid[i], grid[i + 1]
        try:
            root = brentq(lambda q: denominator(q).imag, b, a, xtol=1e-13, rtol=1e-13)
        except ValueError:
            continue
        residual = abs(denominator(root))
        if residual > 2e-6:
            continue
        if roots and abs(root - roots[-1]) < 1e-5:
            continue
        roots.append(float(root))
        if len(roots) >= n_poles:
            return roots

    raise RuntimeError(f"found only {len(roots)} negative moment poles before p={p_min}")


def find_heston_first_adjacent_sheet_saddle(params, threshold, T=1.0, dps=90, reference_saddle=None,
                                             p_min=-60.0, scan_points=1600):
    """Track the first adjacent-sheet stationary point robustly.

    The first adjacent sector is defined as the first real solution of

        Lambda'(p) = threshold

    encountered after crossing the first negative moment pole, but before the
    second pole.  This avoids root solvers accidentally jumping to a more
    distant logarithmic sheet as parameters such as maturity are varied.
    """
    if reference_saddle is None:
        reference_saddle = find_heston_tail_saddle(params, threshold, T=T, dps=max(70, dps - 10))

    pole1, pole2 = heston_negative_moment_poles(params, T=T, n_poles=2, p_min=p_min)
    width = pole1 - pole2
    # Stay away from the poles, where the numerical derivative diverges.
    right = pole1 - max(1e-4, 2e-4 * width)
    left = pole2 + max(1e-4, 2e-4 * width)
    grid = np.linspace(right, left, int(scan_points))

    previous = None
    bracket = None
    for q in grid:
        try:
            value = _real_cgf_derivative(float(q), params, T=T) - float(threshold)
        except Exception:
            previous = None
            continue
        if not np.isfinite(value) or abs(value) > 1e4:
            previous = None
            continue
        if previous is not None and previous[1] * value < 0.0:
            bracket = (previous[0], float(q))
            break
        previous = (float(q), float(value))

    if bracket is None:
        raise RuntimeError("no adjacent stationary point found between the first two negative moment poles")

    root = brentq(lambda q: _real_cgf_derivative(q, params, T=T) - threshold,
                  bracket[1], bracket[0], xtol=1e-12, rtol=1e-12)
    local = max(1e-3, 2e-3 * width)
    adjacent = find_heston_adjacent_sheet_saddle(
        params,
        threshold,
        T=T,
        dps=dps,
        guesses=(root + local, root - local),
        reference_saddle=reference_saddle,
    )
    adjacent["moment_poles"] = (float(pole1), float(pole2))
    adjacent["root_bracket"] = tuple(float(v) for v in bracket)
    return adjacent


def find_heston_adjacent_sheet_saddle(params, threshold, T=1.0, dps=90, guesses=(-7.5, -8.5), reference_saddle=None):
    """Find the first real stationary point on the adjacent logarithmic sheet.

    The closed affine expression analytically continued past the first
    negative moment pole develops logarithmic monodromy.  The derivative is
    insensitive to adding an integer logarithmic branch constant, so the
    stationary point can be found from ``Lambda'(p)=threshold``.  The returned
    singulant is

        Delta = Phi(p_star) - Phi(p_1),  Phi(p)=Lambda(p)-p threshold,

    evaluated on the principal closed-form branch.  Its conjugate corresponds
    to continuation around the moment pole on the opposite side.
    """
    if reference_saddle is None:
        reference_saddle = find_heston_tail_saddle(params, threshold, T=T, dps=max(70, dps - 10))

    old_dps = mp.mp.dps
    mp.mp.dps = dps
    try:
        x = mp.mpf(str(threshold))
        deriv = lambda p: mp.diff(lambda q: _heston_scaled_cgf_mp(q, params, T=T), p)
        p1 = mp.findroot(lambda p: deriv(p) - x, tuple(mp.mpf(str(g)) for g in guesses), tol=mp.mpf("1e-50"))
        p0 = mp.mpf(str(reference_saddle.p_star))
        phi0 = _heston_scaled_cgf_mp(p0, params, T=T) - p0 * x
        phi1 = _heston_scaled_cgf_mp(p1, params, T=T) - p1 * x
        singulant = phi0 - phi1
        # Fix a deterministic convention for the adjacent logarithmic sheet:
        # the named singulant is the member of the conjugate pair in the
        # upper Borel half-plane. Numerical root finding can approach the real
        # saddle from either side and otherwise flip this sign.
        if mp.im(singulant) < 0:
            singulant = mp.conj(singulant)
    finally:
        mp.mp.dps = old_dps

    return {
        "p": complex(mp.re(p1)),
        "singulant": complex(singulant),
        "conjugate_singulant": complex(mp.conj(singulant)),
    }


def heston_conformal_borel_coefficients(coeffs, leading_rate, dps=100):
    """Compose the Borel series with the one-cut conformal map at high precision.

    The leading Heston CDF singularity lies at ``zeta=-I``.  The map

        zeta = 4 I w / (1-w)^2

    sends the cut plane with the ray ``(-infinity,-I]`` removed to the unit disk and moves the
    leading branch point to ``w=-1``.  Performing the composition in arbitrary
    precision is important: float64 composition suffers catastrophic
    cancellation at the orders relevant for secondary-singularity searches.
    """
    old_dps = mp.mp.dps
    mp.mp.dps = dps
    try:
        coeffs_mp = [mp.mpf(str(v)) if not isinstance(v, mp.mpf) else v for v in coeffs]
        n_terms = len(coeffs_mp)
        rate = mp.mpf(str(leading_rate))
        borel = [coeffs_mp[n] / mp.factorial(n) for n in range(n_terms)]
        z_of_w = [mp.mpf("0")] * n_terms
        for m in range(1, n_terms):
            z_of_w[m] = 4 * rate * m

        def truncated_convolution(a, b):
            out = [mp.mpf("0")] * n_terms
            for i, ai in enumerate(a):
                if ai == 0:
                    continue
                for j in range(min(len(b), n_terms - i)):
                    out[i + j] += ai * b[j]
            return out

        result = [mp.mpf("0")] * n_terms
        power = [mp.mpf("0")] * n_terms
        power[0] = mp.mpf("1")
        for n in range(n_terms):
            if n:
                power = truncated_convolution(power, z_of_w)
            for k in range(n_terms):
                result[k] += borel[n] * power[k]
        return result
    finally:
        mp.mp.dps = old_dps


def heston_conformal_pade_poles(conformal_coeffs, leading_rate, n_coeffs=None, scale=0.55):
    """Padé poles after the leading-cut conformal map.

    A geometric rescaling of the series variable is used only for numerical
    conditioning; returned poles are converted back to both the ``w`` and
    original Borel ``zeta`` planes.
    """
    from scipy.interpolate import pade

    if n_coeffs is None:
        n_coeffs = len(conformal_coeffs)
    n_coeffs = min(int(n_coeffs), len(conformal_coeffs))
    scaled = np.asarray([float(conformal_coeffs[n] * (mp.mpf(str(scale)) ** n)) for n in range(n_coeffs)], dtype=float)
    numerator, denominator = pade(scaled, max(1, (n_coeffs - 1) // 2))
    roots = np.roots(denominator)
    w_poles = scale * roots
    zeta_poles = np.asarray([4.0 * float(leading_rate) * w / (1.0 - w) ** 2 for w in w_poles], dtype=complex)
    return w_poles, zeta_poles


def heston_adjacent_sheet_fluctuation_coefficients(params, threshold, T=1.0, n_terms=6, dps=110, reference_saddle=None, adjacent=None):
    """Local fluctuation series around the first adjacent-sheet Heston saddle.

    The adjacent stationary point lies beyond the first negative moment pole.
    On the benchmark used in the paper its quadratic curvature is negative, so
    the local steepest-descent direction is horizontal.  More generally we set

        a^2 Lambda''(p_1) = -1,

    and expand p = p_1 + a sqrt(epsilon) z.  The returned coefficients d_n are
    normalized so d_0=1 in

        I_1(epsilon) ~ C_1 sqrt(epsilon) exp(Phi_1/epsilon)
                       sum_n d_n epsilon^n.

    ``prefactor_ratio`` is C_1/C_0 relative to the physical-saddle sector,
    using the same orientation convention as the Bromwich integral.
    """
    if n_terms < 1:
        raise ValueError("n_terms must be positive")
    if reference_saddle is None:
        reference_saddle = find_heston_tail_saddle(params, threshold, T=T, dps=max(70, dps - 20))

    old_dps = mp.mp.dps
    mp.mp.dps = dps
    try:
        x = mp.mpf(str(threshold))
        derivative = lambda p: mp.diff(lambda q: _heston_scaled_cgf_mp(q, params, T=T), p)
        if adjacent is None:
            adjacent = find_heston_first_adjacent_sheet_saddle(
                params, threshold, T=T, dps=max(80, dps - 15), reference_saddle=reference_saddle
            )
        p1 = mp.mpf(str(complex(adjacent["p"]).real))
        p0 = mp.mpf(str(reference_saddle.p_star))

        max_s_power = 2 * (n_terms - 1)
        max_lambda_order = max_s_power + 2
        lam_taylor = mp.taylor(lambda p: _heston_scaled_cgf_mp(p, params, T=T), p1, max_lambda_order)
        lambda_second = 2 * lam_taylor[2]
        descent_scale = mp.sqrt(-1 / lambda_second)

        exp_series = [[mp.mpc(1)]]
        q_coeff = [None] * (max_s_power + 1)
        for m in range(1, max_s_power + 1):
            k = m + 2
            q_coeff[m] = lam_taylor[k] * descent_scale ** k

        for n in range(1, max_s_power + 1):
            poly = [mp.mpc(0)] * (3 * n + 1)
            for m in range(1, n + 1):
                previous = exp_series[n - m]
                shift = m + 2
                factor = (mp.mpf(m) / n) * q_coeff[m]
                for degree, value in enumerate(previous):
                    poly[degree + shift] += factor * value
            exp_series.append(poly)

        amplitude_base = -descent_scale / p1
        half_power_coeffs = []
        for n in range(max_s_power + 1):
            expectation = mp.mpc(0)
            for j in range(n + 1):
                amplitude = amplitude_base ** j
                poly = exp_series[n - j]
                for degree, value in enumerate(poly):
                    total_degree = degree + j
                    if total_degree % 2 == 0:
                        expectation += amplitude * value * _normal_moment(total_degree)
            half_power_coeffs.append(expectation)

        coeffs = [half_power_coeffs[2 * n] for n in range(n_terms)]
        phi0 = _heston_scaled_cgf_mp(p0, params, T=T) - p0 * x
        phi1 = _heston_scaled_cgf_mp(p1, params, T=T) - p1 * x
        singulant = phi0 - phi1

        c1 = mp.j * descent_scale / (p1 * mp.sqrt(2 * mp.pi))
        lambda_second_0 = mp.mpf(str(reference_saddle.lambda_second))
        c0 = -1 / (p0 * mp.sqrt(2 * mp.pi * lambda_second_0))
        prefactor_ratio = c1 / c0
    finally:
        mp.mp.dps = old_dps

    return {
        "p": complex(p1),
        "singulant": complex(singulant),
        "lambda_second": complex(lambda_second),
        "descent_scale": complex(descent_scale),
        "prefactor_ratio": complex(prefactor_ratio),
        "coefficients": np.asarray([complex(v) for v in coeffs], dtype=complex),
    }



def heston_complex_saddle_fluctuation_coefficients(params, threshold, saddle_p, T=1.0, n_terms=6, dps=110, radius=0.05):
    """Local normalized fluctuation series around a complex Heston saddle.

    The local steepest coordinate is chosen so that ``a**2 Lambda''=-1``.
    The returned integer-power series is normalized to ``d_0=1``.  For real
    model parameters, conjugate saddles give conjugate coefficient sequences
    (up to the harmless overall thimble-orientation sign in the one-loop
    prefactor).  Additive logarithmic sheet shifts do not affect these local
    coefficients because they depend only on derivatives of the affine phase.
    """
    if n_terms < 1:
        raise ValueError("n_terms must be positive")
    old_dps = mp.mp.dps
    mp.mp.dps = int(dps)
    try:
        p1 = mp.mpc(saddle_p)
        max_s_power = 2 * (n_terms - 1)
        lam_taylor = _cauchy_taylor_coefficients_mp(
            lambda p: _heston_scaled_cgf_mp(p, params, T=T),
            p1, max_s_power + 2, radius=radius,
        )
        lambda_second = 2 * lam_taylor[2]
        descent_scale = mp.sqrt(-1 / lambda_second)
        # Fix a deterministic orientation.  The normalized coefficients are
        # invariant under the simultaneous reversal of the local thimble; this
        # convention additionally makes conjugate roots choose conjugate scales.
        if mp.re(descent_scale) < 0 or (abs(mp.re(descent_scale)) < mp.mpf('1e-40') and mp.im(descent_scale) < 0):
            descent_scale = -descent_scale

        exp_series = [[mp.mpc(1)]]
        q_coeff = [None] * (max_s_power + 1)
        for m in range(1, max_s_power + 1):
            k = m + 2
            q_coeff[m] = lam_taylor[k] * descent_scale ** k
        for n in range(1, max_s_power + 1):
            poly = [mp.mpc(0)] * (3 * n + 1)
            for m in range(1, n + 1):
                previous = exp_series[n - m]
                shift = m + 2
                factor = (mp.mpf(m) / n) * q_coeff[m]
                for degree, value in enumerate(previous):
                    poly[degree + shift] += factor * value
            exp_series.append(poly)

        amplitude_base = -descent_scale / p1
        half_power_coeffs = []
        for n in range(max_s_power + 1):
            expectation = mp.mpc(0)
            for j in range(n + 1):
                amplitude = amplitude_base ** j
                poly = exp_series[n - j]
                for degree, value in enumerate(poly):
                    total_degree = degree + j
                    if total_degree % 2 == 0:
                        expectation += amplitude * value * _normal_moment(total_degree)
            half_power_coeffs.append(expectation)
        coeffs = [half_power_coeffs[2 * n] for n in range(n_terms)]
        one_loop = mp.j * descent_scale / (p1 * mp.sqrt(2 * mp.pi))
    finally:
        mp.mp.dps = old_dps
    return {
        "p": complex(p1),
        "lambda_second": complex(lambda_second),
        "descent_scale": complex(descent_scale),
        "one_loop_prefactor": complex(one_loop),
        "coefficients": np.asarray([complex(v) for v in coeffs], dtype=complex),
    }


def _polyval_ascending_mp(coeffs, z):
    value = mp.mpc(0)
    for coefficient in reversed(coeffs):
        value = value * z + coefficient
    return value


def _polyder_ascending_mp(coeffs, z):
    return sum(mp.mpc(k) * coeffs[k] * z ** (k - 1) for k in range(1, len(coeffs)))


def heston_conformal_pade_pole_residues(conformal_coeffs, leading_rate, n_coeffs=None, scale=0.55, dps=90):
    """Return conformal-Pade poles and their residues in the Borel zeta plane.

    The conformal series represents ``B(zeta(w))`` with

        zeta(w) = 4 I w / (1-w)^2.

    A high-precision Pade solve is used for the coefficients, after which the
    denominator roots are refined with mpmath.  The returned residue is the
    residue of the reconstructed Borel transform with respect to ``zeta``.
    This is the quantity entering a lateral Laplace jump through

        Disc Phi = (2 pi i / epsilon) sum_j exp(-zeta_j/epsilon) Res_j.
    """
    if n_coeffs is None:
        n_coeffs = len(conformal_coeffs)
    n_coeffs = min(int(n_coeffs), len(conformal_coeffs))
    if n_coeffs < 4:
        raise ValueError("at least four conformal coefficients are required")

    old_dps = mp.mp.dps
    mp.mp.dps = dps
    try:
        rate = mp.mpf(str(leading_rate))
        scale_mp = mp.mpf(str(scale))
        series = [mp.mpf(str(conformal_coeffs[n])) * scale_mp ** n for n in range(n_coeffs)]
        denominator_degree = max(1, (n_coeffs - 1) // 2)
        numerator_degree = n_coeffs - 1 - denominator_degree
        numerator, denominator = mp.pade(series, numerator_degree, denominator_degree)

        # Root finding is seeded in double precision but each root is refined
        # against the arbitrary-precision denominator.  This is substantially
        # more stable than forming the Pade system itself in float64.
        roots0 = np.roots(np.asarray([float(v) for v in reversed(denominator)], dtype=float))
        roots = []
        for root0 in roots0:
            try:
                root = mp.findroot(
                    lambda u: _polyval_ascending_mp(denominator, u),
                    mp.mpc(root0),
                    tol=mp.mpf("1e-40"),
                    maxsteps=60,
                )
            except (ValueError, ZeroDivisionError):
                root = mp.mpc(root0)
            if not any(abs(root - previous) < mp.mpf("1e-24") for previous in roots):
                roots.append(root)

        rows = []
        for u in roots:
            w = scale_mp * u
            zeta = 4 * rate * w / (1 - w) ** 2
            denominator_prime = _polyder_ascending_mp(denominator, u)
            if denominator_prime == 0:
                continue
            residue_w = scale_mp * _polyval_ascending_mp(numerator, u) / denominator_prime
            dzeta_dw = 4 * rate * (1 + w) / (1 - w) ** 3
            residue_zeta = residue_w * dzeta_dw
            rows.append((complex(w), complex(zeta), complex(residue_zeta)))
    finally:
        mp.mp.dps = old_dps

    rows.sort(key=lambda item: abs(item[1]))
    return rows


def heston_secondary_pade_chain(conformal_coeffs, leading_rate, singulant, n_coeffs=None, scale=0.55, angle_window=0.30, max_abs_w=0.90, dps=90):
    """Select the Pade pole condensation associated with one secondary cut.

    The finite-order rational approximant represents a logarithmic branch cut
    by a string of simple poles.  For the first adjacent Heston sector the
    relevant string starts near ``singulant`` and bends toward conformal
    infinity.  Selection is geometric and does not use pole residues or a
    fitted Stokes multiplier.
    """
    target = complex(singulant)
    theta = cmath.phase(target)
    sign = 1.0 if target.imag >= 0 else -1.0
    min_modulus = 0.60 * abs(target)
    rows = heston_conformal_pade_pole_residues(
        conformal_coeffs, leading_rate, n_coeffs=n_coeffs, scale=scale, dps=dps
    )
    selected = []
    for w, zeta, residue in rows:
        if sign * zeta.imag <= 0 or abs(w) >= max_abs_w or abs(zeta) < min_modulus:
            continue
        angle_distance = abs(cmath.phase(cmath.exp(1j * (cmath.phase(zeta) - theta))))
        if angle_distance < angle_window and zeta.real > 0:
            selected.append((w, zeta, residue))
    return selected


def heston_secondary_lateral_stokes(epsilon_radius, conformal_coeffs, leading_rate, singulant, adjacent_coeffs, prefactor_ratio, n_coeffs=None, scale=0.55, angle_window=0.30, max_abs_w=0.90, dps=90):
    """Estimate the secondary Stokes multiplier from a lateral Pade jump.

    ``epsilon`` is placed on the Stokes ray ``arg epsilon = arg singulant``.
    The Pade pole string approximating the secondary logarithmic cut is crossed
    when the Laplace contour is continued laterally.  Cauchy's theorem then
    gives the finite-order discontinuity directly from the pole residues.

    With the convention used throughout the project,

        Disc Phi_0 = S (C_1/C_0) exp(-Delta/epsilon) Phi_1,

    where ``Phi_1`` is evaluated from the supplied adjacent fluctuation
    coefficients.  No late-order fit for ``S`` is performed here.
    """
    radius = float(epsilon_radius)
    if radius <= 0:
        raise ValueError("epsilon_radius must be positive")
    delta = complex(singulant)
    theta = cmath.phase(delta)
    epsilon = radius * cmath.exp(1j * theta)
    chain = heston_secondary_pade_chain(
        conformal_coeffs,
        leading_rate,
        delta,
        n_coeffs=n_coeffs,
        scale=scale,
        angle_window=angle_window,
        max_abs_w=max_abs_w,
        dps=dps,
    )
    if not chain:
        raise RuntimeError("no secondary Pade pole chain was identified")

    residue_sum = sum(cmath.exp(-zeta / epsilon) * residue for _, zeta, residue in chain)
    discontinuity = (2j * math.pi / epsilon) * residue_sum
    adjacent_fluctuation = sum(complex(value) * epsilon ** n for n, value in enumerate(adjacent_coeffs))
    normalization = complex(prefactor_ratio) * cmath.exp(-delta / epsilon) * adjacent_fluctuation
    stokes = discontinuity / normalization
    return {
        "epsilon": epsilon,
        "discontinuity": discontinuity,
        "adjacent_fluctuation": adjacent_fluctuation,
        "normalization": normalization,
        "stokes": stokes,
        "chain": chain,
    }

def heston_secondary_log_model_coefficients(adjacent_coeffs, leading_rate, singulant, n_terms, fluctuation_order=2, dps=100):
    """Conformal-Borel coefficient model generated by one adjacent saddle.

    If the discontinuity of the physical Borel transform at ``Delta`` is
    proportional to the adjacent-sector Borel transform, the local singular
    part can be written

        K B_1(zeta-Delta) log(1-w/w_1),

    after the leading-cut conformal map.  This routine returns the Taylor
    coefficients multiplying a unit K for one member of the conjugate pair.
    The real physical sequence is modeled as ``2 Re(K f_n)``.
    """
    if n_terms < 2:
        raise ValueError("n_terms must be at least two")
    fluctuation_order = min(int(fluctuation_order), len(adjacent_coeffs) - 1)
    if fluctuation_order < 0:
        raise ValueError("adjacent_coeffs must not be empty")

    old_dps = mp.mp.dps
    mp.mp.dps = dps
    try:
        rate = mp.mpf(str(leading_rate))
        delta = mp.mpc(singulant)
        coeffs = [mp.mpc(v) for v in adjacent_coeffs]
        root = mp.sqrt(1 + delta / rate)
        w1 = (root - 1) / (root + 1)

        def zeta_of_w(w):
            return 4 * rate * w / (1 - w) ** 2

        def adjacent_borel(t):
            return sum(coeffs[k] * t ** k / mp.factorial(k) for k in range(fluctuation_order + 1))

        local_taylor = [
            mp.diff(lambda w: adjacent_borel(zeta_of_w(w) - delta), w1, j) / mp.factorial(j)
            for j in range(fluctuation_order + 1)
        ]

        # Convert sum_j h_j (w-w1)^j into ordinary powers of w.
        poly = [mp.mpc(0)] * (fluctuation_order + 1)
        for j, value in enumerate(local_taylor):
            for r in range(j + 1):
                poly[r] += value * mp.binomial(j, r) * (-w1) ** (j - r)

        model = [mp.mpc(0)] * n_terms
        for n in range(1, n_terms):
            total = mp.mpc(0)
            for r, value in enumerate(poly):
                if r < n:
                    total -= value / ((n - r) * w1 ** (n - r))
            model[n] = total
    finally:
        mp.mp.dps = old_dps

    return np.asarray([complex(v) for v in model], dtype=complex), complex(w1)


def fit_heston_effective_stokes_multiplier(conformal_coeffs, model_coeffs, prefactor_ratio, w1, start_order=40, end_order=None, asymptotic_weighting=True):
    """Fit the logarithmic-branch strength and an effective Stokes multiplier.

    With our convention

        B_0(w) ~ K B_1 log(1-w/w_1) + conjugate,
        K = (S_eff / (2 pi i)) (C_1/C_0).

    The fit uses only the one complex constant K.  Optional asymptotic
    weighting removes the expected |w_1|^{-n}/n envelope so that each order in
    the selected window has comparable influence.
    """
    values = np.asarray(conformal_coeffs, dtype=float)
    model = np.asarray(model_coeffs, dtype=complex)
    if end_order is None:
        end_order = min(len(values), len(model)) - 1
    start_order = int(start_order)
    end_order = int(end_order)
    if start_order < 1 or end_order < start_order:
        raise ValueError("invalid fit window")

    orders = np.arange(start_order, end_order + 1)
    design = np.column_stack([2 * model[orders].real, -2 * model[orders].imag])
    target = values[orders].copy()
    if asymptotic_weighting:
        weights = orders * (abs(complex(w1)) ** orders)
        design = design * weights[:, None]
        target = target * weights

    solution, *_ = np.linalg.lstsq(design, target, rcond=None)
    K = complex(solution[0], solution[1])
    prediction = design @ solution
    nrmse = float(np.linalg.norm(prediction - target) / np.linalg.norm(target))
    ratio = complex(prefactor_ratio)
    stokes = 2j * np.pi * K / ratio
    return {"K": K, "stokes": stokes, "nrmse": nrmse, "start_order": start_order, "end_order": end_order}


def fit_heston_secondary_singulant(conformal_coeffs, leading_rate, initial_singulant, start_order=40, end_order=None):
    """Infer a secondary logarithmic singularity directly from late coefficients.

    This is a four-parameter Prony-like fit of a conjugate logarithmic pair,

        b_n ~ -2 Re[K/(n w_1^n)],

    in the conformally mapped Borel plane.  It is used as an independent
    late-order location diagnostic; local adjacent-saddle fluctuations are not
    supplied to the fit.
    """
    from scipy.optimize import least_squares

    values = np.asarray(conformal_coeffs, dtype=float)
    if end_order is None:
        end_order = len(values) - 1
    orders = np.arange(int(start_order), int(end_order) + 1)
    rate = float(leading_rate)
    delta0 = complex(initial_singulant)
    root0 = cmath.sqrt(1 + delta0 / rate)
    w0 = (root0 - 1) / (root0 + 1)
    scale_radius = abs(w0)
    target = values[orders]
    weights = orders * (scale_radius ** orders)

    def residual(q):
        w = complex(q[0], q[1])
        K = complex(q[2], q[3])
        pred = np.asarray([-2 * np.real(K / (n * w ** n)) for n in orders])
        return (pred - target) * weights

    initial = np.array([w0.real, w0.imag, 0.027, 0.003], dtype=float)
    result = least_squares(residual, initial, max_nfev=30000, xtol=1e-13, ftol=1e-13, gtol=1e-13)
    w = complex(result.x[0], result.x[1])
    K = complex(result.x[2], result.x[3])
    zeta = 4 * rate * w / (1 - w) ** 2
    rms = float(np.linalg.norm(residual(result.x)) / np.sqrt(len(orders)))
    return {"w": w, "zeta": zeta, "K": K, "weighted_rms": rms, "success": bool(result.success)}


def heston_phase_derivative(p, params, threshold, T=1.0, relative_step=1e-5):
    """Numerical holomorphic derivative of Phi(p)=Lambda(p)-p*threshold.

    A symmetric complex finite difference is used.  The derivative is
    insensitive to additive logarithmic monodromy constants, which makes it
    suitable for tracing Picard--Lefschetz flows between neighboring Heston
    logarithmic sheets away from the isolated moment pole.
    """
    p = complex(p)
    h = float(relative_step) * max(1.0, abs(p))
    fp = heston_scaled_cgf(p + h, params, T=T) - (p + h) * threshold
    fm = heston_scaled_cgf(p - h, params, T=T) - (p - h) * threshold
    return (fp - fm) / (2.0 * h)


def heston_thimble_direction(lambda_second, theta, upward=True, branch=0):
    """Local tangent angle for a one-dimensional Lefschetz flow.

    For h(p)=Phi(p) exp(-i theta), upward flow obeys

        dp/dtau = conjugate(h'(p)).

    Near a nondegenerate saddle the two upward directions have
    ``alpha=-arg(h'')/2 mod pi``; downward directions are rotated by pi/2.
    """
    hessian = complex(lambda_second) * cmath.exp(-1j * float(theta))
    alpha = -0.5 * cmath.phase(hessian)
    if not upward:
        alpha += 0.5 * math.pi
    alpha += int(branch) * math.pi
    return float(alpha)


def heston_integrate_thimble(params, threshold, saddle_p, lambda_second, theta, T=1.0, upward=True, branch=0,
                              tau_max=60.0, initial_radius=2e-4, max_step=0.05, rtol=1e-7, atol=1e-9):
    """Integrate one Heston Lefschetz-flow branch in the complex p plane.

    The flow uses the holomorphic phase ``Phi/epsilon`` with only the phase of
    epsilon retained.  Multiplying epsilon by a positive real number merely
    reparametrizes flow time and leaves the thimble geometry unchanged.
    """
    from scipy.integrate import solve_ivp

    theta = float(theta)
    direction = heston_thimble_direction(lambda_second, theta, upward=upward, branch=branch)
    z0 = complex(saddle_p) + float(initial_radius) * cmath.exp(1j * direction)
    epsilon_phase = cmath.exp(1j * theta)

    def rhs(_, y):
        z = complex(y[0], y[1])
        derivative = heston_phase_derivative(z, params, threshold, T=T) / epsilon_phase
        velocity = np.conj(derivative)
        if not upward:
            velocity = -velocity
        return [float(np.real(velocity)), float(np.imag(velocity))]

    solution = solve_ivp(
        rhs,
        (0.0, float(tau_max)),
        [z0.real, z0.imag],
        rtol=float(rtol),
        atol=float(atol),
        max_step=float(max_step),
    )
    path = solution.y[0] + 1j * solution.y[1]
    return solution.t, path


def heston_vertical_contour_intersections(path, contour_real):
    """Intersections of a polygonal complex path with an upward vertical line.

    Returns dictionaries containing the interpolated crossing point and the
    oriented intersection sign ``sign det(t_Gamma,t_path)``.  With the
    Bromwich contour oriented upward, a left-to-right path crossing has sign
    -1 and a right-to-left crossing has sign +1.
    """
    z = np.asarray(path, dtype=complex)
    contour_real = float(contour_real)
    rows = []
    for i in range(len(z) - 1):
        x0 = z[i].real - contour_real
        x1 = z[i + 1].real - contour_real
        if x0 == 0.0 and x1 == 0.0:
            continue
        if x0 * x1 > 0.0:
            continue
        dx = z[i + 1].real - z[i].real
        if abs(dx) < 1e-14:
            continue
        u = (contour_real - z[i].real) / dx
        if not (0.0 <= u <= 1.0):
            continue
        point = z[i] + u * (z[i + 1] - z[i])
        tangent = z[i + 1] - z[i]
        # det((0,1),(Re t, Im t)) = -Re t.
        orientation = -1 if tangent.real > 0 else 1
        rows.append({"index": i, "point": complex(point), "orientation": int(orientation)})
    return rows


def heston_picard_lefschetz_scan(params, threshold, T=1.0, contour_real=-1.0, angle_offsets=(-0.04, -0.02, 0.0, 0.02, 0.04),
                                  tau_max=60.0, reference_saddle=None, adjacent=None, adjacent_local=None, branch=None):
    """Trace the first adjacent dual cycle across its secondary Stokes wall.

    The branch is chosen so that at the exact Stokes angle it approaches the
    physical saddle.  For the project benchmark this produces the topology

        below the wall: no Bromwich intersection,
        on the wall: p1 -> p_star heteroclinic connection,
        above the wall: one negatively oriented Bromwich intersection.

    This integer wall-crossing is the Picard--Lefschetz origin of the observed
    Stokes multiplier S_01=-1 in the conventions of the paper.
    """
    if reference_saddle is None:
        reference_saddle = find_heston_tail_saddle(params, threshold, T=T, dps=80)
    if adjacent is None:
        adjacent = find_heston_adjacent_sheet_saddle(params, threshold, T=T, dps=90, reference_saddle=reference_saddle)
    if adjacent_local is None:
        adjacent_local = heston_adjacent_sheet_fluctuation_coefficients(
            params, threshold, T=T, n_terms=2, dps=90, reference_saddle=reference_saddle, adjacent=adjacent
        )

    theta_stokes = cmath.phase(adjacent["singulant"])
    p1 = adjacent["p"]
    lambda_second = adjacent_local["lambda_second"]

    # The heteroclinic branch can switch between the two local upward rays as
    # parameters are varied.  Select it geometrically at the Stokes wall
    # rather than hard-coding a branch label from one benchmark point.
    if branch is None:
        candidates = []
        for candidate_branch in (0, 1):
            _, candidate_path = heston_integrate_thimble(
                params, threshold, p1, lambda_second, theta_stokes, T=T, upward=True,
                branch=candidate_branch, tau_max=tau_max
            )
            distance = float(np.min(np.abs(candidate_path - reference_saddle.p_star)))
            candidates.append((distance, candidate_branch))
        branch = min(candidates)[1]
    branch = int(branch)

    rows = []
    paths = {}
    for offset in angle_offsets:
        theta = theta_stokes + float(offset)
        t, path = heston_integrate_thimble(
            params,
            threshold,
            p1,
            lambda_second,
            theta,
            T=T,
            upward=True,
            branch=branch,
            tau_max=tau_max,
        )
        intersections = heston_vertical_contour_intersections(path, contour_real)
        min_distance = float(np.min(np.abs(path - reference_saddle.p_star)))
        signed_intersection = int(sum(item["orientation"] for item in intersections))
        rows.append({
            "offset": float(offset),
            "theta": float(theta),
            "intersection_number": signed_intersection,
            "n_intersections": len(intersections),
            "min_distance_to_physical_saddle": min_distance,
            "end": complex(path[-1]),
            "intersections": intersections,
        })
        paths[float(offset)] = (t, path)

    return {
        "theta_stokes": float(theta_stokes),
        "physical_saddle": float(reference_saddle.p_star),
        "adjacent_saddle": complex(p1),
        "contour_real": float(contour_real),
        "selected_branch": int(branch),
        "rows": rows,
        "paths": paths,
    }
