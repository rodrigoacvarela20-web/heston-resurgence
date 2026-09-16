from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class HestonParams:
    mu: float = 0.0
    kappa: float = 2.0
    theta: float = 0.04
    xi: float = 0.5
    rho: float = -0.7
    v0: float = 0.04


def simulate_heston(params, T=1.0, n_steps=252, n_paths=10000, seed=0, controls=None):
    """Simulate log-price X=log(S/S0) and variance V with full truncation Euler.

    controls: optional array (n_steps, 2) giving means of the two independent
    standard-normal shocks under an importance-sampling measure Q.

    Returns X, V, log_likelihood_ratio where the last term is log(dP/dQ).
    """
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
    if controls.shape != (n_steps, 2):
        raise ValueError(f"controls must have shape {(n_steps, 2)}")

    rho_bar = np.sqrt(max(1.0 - params.rho**2, 1e-14))

    for i in range(n_steps):
        u = controls[i]
        z = rng.normal(size=(n_paths, 2)) + u
        z1 = z[:, 0]
        z2 = z[:, 1]
        z_v = params.rho * z1 + rho_bar * z2

        v = np.maximum(V[:, i], 0.0)
        sqrt_v = np.sqrt(v)

        X[:, i + 1] = X[:, i] + (params.mu - 0.5 * v) * dt + sqrt_v * sqrt_dt * z1
        V[:, i + 1] = V[:, i] + params.kappa * (params.theta - v) * dt + params.xi * sqrt_v * sqrt_dt * z_v
        V[:, i + 1] = np.maximum(V[:, i + 1], 1e-12)

        # If z ~ N(u, I) under Q and N(0, I) under P,
        # log(dP/dQ) = -u.z + 0.5|u|^2.
        log_lr += -(z @ u) + 0.5 * (u @ u)

    return X, V, log_lr


def estimate_left_tail_probability(X_terminal, threshold, log_weights=None):
    event = X_terminal <= threshold
    if log_weights is None:
        values = event.astype(float)
    else:
        # Stable enough for the moderate controls used in the experiments.
        values = event.astype(float) * np.exp(log_weights)
    estimate = values.mean()
    standard_error = values.std(ddof=1) / np.sqrt(len(values))
    return estimate, standard_error
