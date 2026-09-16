"""Small-noise Heston simulation at epsilon=1 and left-tail estimators."""
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class HestonParams:
    mu: float = 0.0
    kappa: float = 2.0
    theta: float = 0.04
    xi: float = 0.5
    rho: float = -0.7
    v0: float = 0.04

    def __post_init__(self):
        values = (self.mu, self.kappa, self.theta, self.xi, self.rho, self.v0)
        if not all(math.isfinite(value) for value in values):
            raise ValueError('Heston parameters must be finite.')
        if self.kappa <= 0 or self.theta <= 0 or self.xi <= 0 or self.v0 < 0:
            raise ValueError('Require kappa, theta and xi > 0, and v0 >= 0.')
        if not -1 <= self.rho <= 1:
            raise ValueError('Correlation rho must lie in [-1, 1].')


def simulate_heston(params, T=1.0, n_steps=252, n_paths=10000, seed=0, controls=None):
    """Simulate log-price and variance using full-truncation Euler.

    controls are optional (n_steps, 2) Gaussian mean shifts under Q. The
    returned likelihood ratio is log(dP/dQ). Discretisation bias remains.
    """
    if not math.isfinite(T) or T <= 0 or not isinstance(n_steps, (int, np.integer)) or n_steps < 1:
        raise ValueError('T must be positive and n_steps a positive integer.')
    if not isinstance(n_paths, (int, np.integer)) or n_paths < 1:
        raise ValueError('n_paths must be a positive integer.')
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    X = np.zeros((n_paths, n_steps + 1))
    V = np.empty((n_paths, n_steps + 1))
    V[:, 0] = params.v0
    log_lr = np.zeros(n_paths)
    if controls is None:
        controls = np.zeros((n_steps, 2))
    controls = np.asarray(controls, dtype=float)
    if controls.shape != (n_steps, 2) or not np.all(np.isfinite(controls)):
        raise ValueError(f'controls must be finite and have shape {(n_steps, 2)}')
    rho_bar = np.sqrt(max(1.0 - params.rho**2, 0.0))
    for i in range(n_steps):
        u = controls[i]
        z = rng.normal(size=(n_paths, 2)) + u
        z1 = z[:, 0]
        z_v = params.rho * z1 + rho_bar * z[:, 1]
        v = np.maximum(V[:, i], 0.0)
        sqrt_v = np.sqrt(v)
        X[:, i + 1] = X[:, i] + (params.mu - 0.5 * v) * dt + sqrt_v * sqrt_dt * z1
        V[:, i + 1] = V[:, i] + params.kappa * (params.theta - v) * dt + params.xi * sqrt_v * sqrt_dt * z_v
        V[:, i + 1] = np.maximum(V[:, i + 1], 1e-12)
        log_lr += -(z @ u) + 0.5 * (u @ u)
    return X, V, log_lr


def estimate_left_tail_probability(X_terminal, threshold, log_weights=None):
    """Return probability estimate and i.i.d. standard error.

    Exponentiate only event weights; otherwise 0 * exp(large non-event weight)
    produces NaN. Rescale event contributions before computing their moments.
    """
    x = np.asarray(X_terminal, dtype=float)
    if x.ndim != 1 or x.size < 2 or not np.all(np.isfinite(x)) or not math.isfinite(threshold):
        raise ValueError('Require at least two finite observations and a finite threshold.')
    event = x <= threshold
    if log_weights is None:
        values = event.astype(float)
        return float(values.mean()), float(values.std(ddof=1) / np.sqrt(x.size))
    weights = np.asarray(log_weights, dtype=float)
    if weights.shape != x.shape or not np.all(np.isfinite(weights)):
        raise ValueError('log_weights must be finite and match observations in shape.')
    if not event.any():
        return 0.0, 0.0
    offset = max(0.0, float(np.max(weights[event])))
    scaled = np.zeros(x.size)
    scaled[event] = np.exp(weights[event] - offset)
    factor = math.exp(offset)
    return float(factor * scaled.mean()), float(factor * scaled.std(ddof=1) / np.sqrt(x.size))
