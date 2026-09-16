import numpy as np


def step_residuals(x0, x1, v0, v1, params, dt):
    """Independent-normal residuals corresponding to one Euler Heston step."""
    v = max(float(v0), 1e-12)
    sqrt_vdt = np.sqrt(v * dt)

    z1 = (x1 - x0 - (params.mu - 0.5 * v) * dt) / sqrt_vdt
    zv = (v1 - v0 - params.kappa * (params.theta - v) * dt) / (params.xi * sqrt_vdt)
    rho_bar = np.sqrt(max(1.0 - params.rho**2, 1e-14))
    z2 = (zv - params.rho * z1) / rho_bar
    return z1, z2


def discrete_freidlin_wentzell_action(x, v, params, T=1.0):
    """Quadratic path action for the independent Brownian controls.

    This is the discrete small-noise/Freidlin-Wentzell cost, not the complete
    finite-dt Onsager-Machlup path density. It is the natural saddle-point
    objective for identifying a most-likely rare path.
    """
    x = np.asarray(x, dtype=float)
    v = np.asarray(v, dtype=float)
    if x.shape != v.shape:
        raise ValueError("x and v must have the same shape")
    dt = T / (len(x) - 1)

    total = 0.0
    for i in range(len(x) - 1):
        z1, z2 = step_residuals(x[i], x[i + 1], v[i], v[i + 1], params, dt)
        total += 0.5 * (z1 * z1 + z2 * z2)
    return total


def path_controls(x, v, params, T=1.0):
    """Return the two independent standardized shocks associated with a path."""
    x = np.asarray(x, dtype=float)
    v = np.asarray(v, dtype=float)
    dt = T / (len(x) - 1)
    controls = np.empty((len(x) - 1, 2))
    for i in range(len(x) - 1):
        controls[i] = step_residuals(x[i], x[i + 1], v[i], v[i + 1], params, dt)
    return controls
