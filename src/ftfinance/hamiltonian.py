import numpy as np
from scipy.integrate import solve_bvp


def instanton_ode(t, y, params):
    """Hamilton equations for the Freidlin-Wentzell Heston instanton.

    State y = (x, v, p_x, p_v), with

        H = p_x b_x + p_v b_v
            + v/2 (p_x^2 + 2 rho xi p_x p_v + xi^2 p_v^2).
    """
    x, v, px, pv = y
    v_safe = np.maximum(v, 1e-12)

    dx = params.mu - 0.5 * v_safe + v_safe * (px + params.rho * params.xi * pv)
    dv = params.kappa * (params.theta - v_safe) + v_safe * (params.rho * params.xi * px + params.xi**2 * pv)
    dpx = np.zeros_like(x)
    dpv = 0.5 * px + params.kappa * pv - 0.5 * (px**2 + 2.0 * params.rho * params.xi * px * pv + params.xi**2 * pv**2)
    return np.vstack([dx, dv, dpx, dpv])


def solve_hamiltonian_instanton(params, threshold, T=1.0, n_steps=100, tol=1e-5, max_nodes=10000):
    """Solve the instanton two-point boundary-value problem.

    Boundary conditions are x(0)=0, v(0)=v0, x(T)=threshold and p_v(T)=0.
    The final variance is free, hence the transversality condition p_v(T)=0.

    A discrete minimum-action solution is used only as the numerical initial
    guess; the returned path solves the continuous Hamilton equations.
    """
    from .instanton import find_terminal_instanton

    seed = find_terminal_instanton(params, threshold, T=T, n_steps=n_steps)
    t = np.linspace(0.0, T, n_steps + 1)
    x0 = seed["x"]
    v0 = seed["v"]
    controls = seed["controls"]
    dt = T / n_steps
    rho_bar = np.sqrt(max(1.0 - params.rho**2, 1e-14))

    # Convert discrete Gaussian means z* into continuous control rates hdot=z*/sqrt(dt),
    # then solve hdot = sigma^T p for the conjugate momenta.
    h1 = controls[:, 0] / np.sqrt(dt)
    h2 = controls[:, 1] / np.sqrt(dt)
    sqrt_v = np.sqrt(np.maximum(v0[:-1], 1e-12))
    pv_steps = h2 / (sqrt_v * params.xi * rho_bar)
    px_steps = h1 / sqrt_v - params.rho * params.xi * pv_steps

    px0 = np.r_[px_steps, px_steps[-1]]
    pv0 = np.r_[pv_steps, 0.0]
    y_guess = np.vstack([x0, v0, px0, pv0])

    def fun(t_eval, y):
        return instanton_ode(t_eval, y, params)

    def bc(ya, yb):
        return np.array([ya[0], ya[1] - params.v0, yb[0] - threshold, yb[3]])

    sol = solve_bvp(fun, bc, t, y_guess, tol=tol, max_nodes=max_nodes, verbose=0)
    t_out = np.linspace(0.0, T, n_steps + 1)
    x, v, px, pv = sol.sol(t_out)

    sqrt_v = np.sqrt(np.maximum(v[:-1], 1e-12))
    h1 = sqrt_v * (px[:-1] + params.rho * params.xi * pv[:-1])
    h2 = sqrt_v * params.xi * rho_bar * pv[:-1]
    proposal_controls = np.column_stack([h1, h2]) * np.sqrt(dt)
    action = 0.5 * np.sum((h1**2 + h2**2) * dt)

    return {
        "t": t_out,
        "x": x,
        "v": v,
        "px": px,
        "pv": pv,
        "controls": proposal_controls,
        "action": float(action),
        "success": bool(sol.success and np.all(v > 0)),
        "message": str(sol.message),
        "bc_residual": bc(sol.y[:, 0], sol.y[:, -1]),
        "discrete_seed_action": seed["action"],
    }


def solve_hamiltonian_shooting(params, threshold, T=1.0, n_steps=400, bracket=(-6.0, -1e-6)):
    """Fast shooting solution of the continuous Heston instanton equations.

    Since p_x is constant and p_v obeys a closed Riccati ODE, p_v is integrated
    backwards from the transversality condition p_v(T)=0. For a trial p_x the
    state equations are then integrated forward. A scalar root solve chooses
    p_x so that x(T)=threshold.

    This solver is independent of the affine-transform/Borel implementation and
    is useful for checking the large-deviation action numerically.
    """
    from scipy.integrate import solve_ivp
    from scipy.optimize import brentq

    rho_bar = np.sqrt(max(1.0 - params.rho**2, 1e-14))

    def integrate_for_px(px, dense_output=False):
        def pv_rhs(t, y):
            pv = y[0]
            return [
                0.5 * px
                + params.kappa * pv
                - 0.5 * (px**2 + 2.0 * params.rho * params.xi * px * pv + params.xi**2 * pv**2)
            ]

        p_sol = solve_ivp(
            pv_rhs,
            (T, 0.0),
            [0.0],
            rtol=2e-11,
            atol=2e-13,
            max_step=T / max(n_steps, 40),
            dense_output=True,
        )
        if not p_sol.success:
            raise RuntimeError(p_sol.message)

        def state_rhs(t, y):
            x, v = y
            pv = p_sol.sol(t)[0]
            dx = params.mu - 0.5 * v + v * (px + params.rho * params.xi * pv)
            dv = params.kappa * (params.theta - v) + v * (params.rho * params.xi * px + params.xi**2 * pv)
            return [dx, dv]

        state_sol = solve_ivp(
            state_rhs,
            (0.0, T),
            [0.0, params.v0],
            rtol=2e-11,
            atol=2e-13,
            max_step=T / max(n_steps, 40),
            dense_output=True,
        )
        if not state_sol.success:
            raise RuntimeError(state_sol.message)
        return state_sol.y[0, -1], p_sol, state_sol

    left, right = map(float, bracket)
    f_left = integrate_for_px(left)[0] - threshold
    f_right = integrate_for_px(right)[0] - threshold
    while f_left * f_right > 0.0:
        left *= 1.5
        if left < -60.0:
            raise RuntimeError("could not bracket the Hamiltonian shooting momentum")
        f_left = integrate_for_px(left)[0] - threshold

    px = brentq(lambda value: integrate_for_px(value)[0] - threshold, left, right, xtol=1e-13, rtol=1e-13)
    _, p_sol, state_sol = integrate_for_px(px, dense_output=True)

    t = np.linspace(0.0, T, n_steps + 1)
    x, v = state_sol.sol(t)
    pv = p_sol.sol(t)[0]
    px_path = np.full_like(t, px)
    v_safe = np.maximum(v, 1e-15)
    h1 = np.sqrt(v_safe) * (px + params.rho * params.xi * pv)
    h2 = np.sqrt(v_safe) * params.xi * rho_bar * pv
    action = np.trapezoid(0.5 * (h1**2 + h2**2), t)

    return {
        "t": t,
        "x": x,
        "v": v,
        "px": px_path,
        "pv": pv,
        "action": float(action),
        "success": bool(np.all(v > 0.0)),
        "message": "Hamiltonian shooting converged",
    }
