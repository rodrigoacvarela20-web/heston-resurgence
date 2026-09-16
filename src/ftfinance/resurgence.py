from dataclasses import dataclass
import math
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import pade
from scipy.special import erfc, gammaln


@dataclass(frozen=True)
class GaussianTailModel:
    """Small-noise Gaussian log-return benchmark.

    X_T = mean_return + sigma * sqrt(epsilon * T) * Z,
    Z ~ N(0, 1), and we study P(X_T <= threshold).
    """

    mean_return: float = 0.0
    sigma: float = 0.2
    T: float = 1.0
    threshold: float = -0.4

    @property
    def gap(self):
        return self.mean_return - self.threshold

    @property
    def instanton_action(self):
        if self.gap <= 0:
            raise ValueError("The threshold must lie below the deterministic mean for a left-tail instanton.")
        return self.gap**2 / (2.0 * self.sigma**2 * self.T)


def gaussian_tail_exact(epsilon, model):
    epsilon = np.asarray(epsilon, dtype=float)
    if np.any(epsilon <= 0):
        raise ValueError("epsilon must be positive")
    z = model.gap / (model.sigma * np.sqrt(2.0 * epsilon * model.T))
    return 0.5 * erfc(z)


def gaussian_tail_fluctuation_coefficients(order, action):
    """Coefficients c_n in

    P_tail ~ exp(-A/eps) * sqrt(eps)/(2 sqrt(pi A)) * sum_n c_n eps^n.

    For the Gaussian benchmark,
        c_n = (-1)^n Gamma(n+1/2) / (sqrt(pi) A^n).
    """
    if order < 1:
        raise ValueError("order must be at least 1")
    if action <= 0:
        raise ValueError("action must be positive")

    n = np.arange(order, dtype=float)
    log_abs = np.array([gammaln(k + 0.5) - 0.5 * math.log(math.pi) - k * math.log(action) for k in n])
    return ((-1.0) ** n) * np.exp(log_abs)


def gaussian_tail_asymptotic(epsilon, model, n_terms):
    epsilon = float(epsilon)
    action = model.instanton_action
    coeffs = gaussian_tail_fluctuation_coefficients(n_terms, action)
    powers = epsilon ** np.arange(n_terms)
    fluct = float(np.dot(coeffs, powers))
    prefactor = math.exp(-action / epsilon) * math.sqrt(epsilon) / (2.0 * math.sqrt(math.pi * action))
    return prefactor * fluct


def optimal_truncation_order(epsilon, action, max_order=200):
    """Return the index after the smallest-magnitude asymptotic term."""
    coeffs = gaussian_tail_fluctuation_coefficients(max_order, action)
    terms = np.abs(coeffs * epsilon ** np.arange(max_order))
    return int(np.argmin(terms) + 1)


def borel_coefficients(coeffs):
    coeffs = np.asarray(coeffs, dtype=float)
    factorials = np.array([math.factorial(i) for i in range(len(coeffs))], dtype=float)
    return coeffs / factorials


def borel_pade(coeffs, denominator_degree=None):
    """Construct a near-diagonal Padé approximant to the Borel transform.

    Returns numerator and denominator polynomials in ascending conceptual use
    through scipy's poly1d objects.
    """
    b = borel_coefficients(coeffs)
    if denominator_degree is None:
        denominator_degree = max(1, (len(b) - 1) // 2)
    denominator_degree = min(denominator_degree, len(b) - 1)
    numerator, denominator = pade(b, denominator_degree)
    return numerator, denominator


def borel_pade_sum(epsilon, coeffs, lateral_imag=0.0, integration_limit=80.0, laguerre_order=96):
    """Borel-Padé resum sum_n c_n epsilon^n.

    The standard Borel-Laplace representation is
        integral_0^inf exp(-t) B(epsilon t) dt.

    On an unobstructed positive Borel ray we use Gauss-Laguerre quadrature,
    which absorbs the exp(-t) weight and is substantially more stable than an
    adaptive integral for high-order rational approximants. A nonzero
    `lateral_imag` switches to explicit contour quadrature for future Stokes
    experiments.
    """
    from scipy.special import roots_laguerre

    epsilon = float(epsilon)
    numerator, denominator = borel_pade(coeffs)

    if lateral_imag == 0.0:
        nodes, weights = roots_laguerre(laguerre_order)
        z = epsilon * nodes
        values = np.polyval(numerator, z) / np.polyval(denominator, z)
        return np.sum(weights * values)

    def integrand_real(t):
        z = epsilon * t + 1j * lateral_imag
        value = np.polyval(numerator, z) / np.polyval(denominator, z)
        return math.exp(-t) * float(np.real(value))

    def integrand_imag(t):
        z = epsilon * t + 1j * lateral_imag
        value = np.polyval(numerator, z) / np.polyval(denominator, z)
        return math.exp(-t) * float(np.imag(value))

    real = quad(integrand_real, 0.0, integration_limit, limit=400, epsabs=1e-12, epsrel=1e-11)[0]
    imag = quad(integrand_imag, 0.0, integration_limit, limit=400, epsabs=1e-12, epsrel=1e-11)[0]
    return real + 1j * imag


def gaussian_tail_borel_pade(epsilon, model, n_coeffs=20):
    action = model.instanton_action
    coeffs = gaussian_tail_fluctuation_coefficients(n_coeffs, action)
    fluct = borel_pade_sum(epsilon, coeffs)
    prefactor = math.exp(-action / epsilon) * math.sqrt(epsilon) / (2.0 * math.sqrt(math.pi * action))
    return prefactor * fluct


def gaussian_borel_exact(zeta, action):
    """Exact Borel transform of the Gaussian instanton fluctuation series."""
    return 1.0 / np.sqrt(1.0 + np.asarray(zeta) / action)


def large_order_action_estimates(coeffs):
    """Estimate the nearest Borel singularity from coefficient ratios.

    If c_n ~ C Gamma(n+beta) / A^n, then
        A_n ~ n * c_{n-1} / c_n.

    The sign is retained, so alternating coefficients reveal a negative-axis
    singularity.
    """
    coeffs = np.asarray(coeffs, dtype=float)
    n = np.arange(1, len(coeffs), dtype=float)
    return n * coeffs[:-1] / coeffs[1:]


def quartic_factor_coefficients(order):
    """Perturbative coefficients for the nonlinear factor integral

        Z(g) = 1/sqrt(pi) * integral exp(-x^2 - g x^4) dx
             ~ sum_n a_n g^n.

    The coefficients are
        a_n = (-1)^n Gamma(2n+1/2) / (sqrt(pi) n!).

    They have factorial large-order growth and the nearest Borel singularity is
    at zeta=-1/4, matching the action difference to the complex saddles
    x = +/- i/sqrt(2g).
    """
    if order < 1:
        raise ValueError("order must be at least 1")
    n = np.arange(order, dtype=float)
    log_abs = np.array([
        gammaln(2.0 * k + 0.5) - 0.5 * math.log(math.pi) - gammaln(k + 1.0)
        for k in n
    ])
    return ((-1.0) ** n) * np.exp(log_abs)


def quartic_factor_exact(g):
    """Exact nonlinear-factor partition function.

    Uses the exponentially scaled Bessel K function for numerical stability:

        Z(g) = exp(1/(8g)) K_{1/4}(1/(8g)) / (2 sqrt(pi g)).
    """
    from scipy.special import kve

    g = np.asarray(g, dtype=float)
    if np.any(g <= 0):
        raise ValueError("g must be positive")
    z = 1.0 / (8.0 * g)
    return kve(0.25, z) / (2.0 * np.sqrt(math.pi * g))


def quartic_factor_partial_sum(g, n_terms):
    coeffs = quartic_factor_coefficients(n_terms)
    return float(np.dot(coeffs, float(g) ** np.arange(n_terms)))


def quartic_factor_borel_pade(g, n_coeffs=18):
    coeffs = quartic_factor_coefficients(n_coeffs)
    return borel_pade_sum(float(g), coeffs)


def quartic_saddle_action():
    """Nearest complex saddle action difference in the g-expansion."""
    return -0.25
