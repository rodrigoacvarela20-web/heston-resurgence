import numpy as np
from scipy.optimize import minimize
from .action import discrete_freidlin_wentzell_action, path_controls


def find_terminal_instanton(params, threshold, T=1.0, n_steps=80, maxiter=2500):
    """Find a minimum-action Heston path ending at x(T)=threshold.

    The terminal variance is free. Interior variances are constrained positive.
    This is a discretized saddle-point calculation intended as a computational
    bridge between MSRJD/large-deviation language and importance sampling.
    """
    x_guess = np.linspace(0.0, threshold, n_steps + 1)
    v_guess = np.full(n_steps + 1, params.theta)
    v_guess[0] = params.v0

    # Optimize x_1,...,x_{N-1}, v_1,...,v_N.
    y0 = np.concatenate([x_guess[1:-1], v_guess[1:]])

    def unpack(y):
        x = np.empty(n_steps + 1)
        v = np.empty(n_steps + 1)
        x[0] = 0.0
        x[-1] = threshold
        x[1:-1] = y[: n_steps - 1]
        v[0] = params.v0
        v[1:] = y[n_steps - 1 :]
        return x, v

    def objective(y):
        x, v = unpack(y)
        if np.any(v <= 0):
            return 1e30
        return discrete_freidlin_wentzell_action(x, v, params, T=T)

    bounds = [(None, None)] * (n_steps - 1) + [(1e-8, None)] * n_steps
    result = minimize(objective, y0, method="L-BFGS-B", bounds=bounds, options={"maxiter": maxiter, "maxfun": 200000, "ftol": 1e-10, "gtol": 1e-7})
    x_star, v_star = unpack(result.x)
    controls = path_controls(x_star, v_star, params, T=T)

    return {
        "x": x_star,
        "v": v_star,
        "controls": controls,
        "action": float(result.fun),
        "success": bool(result.success),
        "message": result.message,
        "nit": int(result.nit),
    }
