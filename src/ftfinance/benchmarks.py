import numpy as np
from .heston import simulate_heston, estimate_left_tail_probability
from .instanton import find_terminal_instanton
from .hamiltonian import solve_hamiltonian_instanton


def _ess_from_logweights(log_weights):
    m = np.max(log_weights)
    w = np.exp(log_weights - m)
    return (w.sum() ** 2) / np.sum(w**2)


def compare_tail_estimators(params, threshold=-0.5, T=1.0, n_steps=80, n_paths=50000, seed=7, control_scale=1.0, method="hamiltonian"):
    """Compare naive Monte Carlo with instanton-guided importance sampling."""
    if method == "hamiltonian":
        instanton = solve_hamiltonian_instanton(params, threshold, T=T, n_steps=n_steps)
    elif method == "discrete":
        instanton = find_terminal_instanton(params, threshold, T=T, n_steps=n_steps)
    else:
        raise ValueError("method must be 'hamiltonian' or 'discrete'")

    controls = control_scale * instanton["controls"]

    X0, _, _ = simulate_heston(params, T=T, n_steps=n_steps, n_paths=n_paths, seed=seed)
    p_mc, se_mc = estimate_left_tail_probability(X0[:, -1], threshold)

    X1, _, log_lr = simulate_heston(params, T=T, n_steps=n_steps, n_paths=n_paths, seed=seed + 1, controls=controls)
    p_is, se_is = estimate_left_tail_probability(X1[:, -1], threshold, log_weights=log_lr)

    variance_reduction = np.inf if se_is == 0 else (se_mc / se_is) ** 2
    return {
        "threshold": threshold,
        "p_mc": p_mc,
        "se_mc": se_mc,
        "p_is": p_is,
        "se_is": se_is,
        "variance_reduction": variance_reduction,
        "weight_ess": _ess_from_logweights(log_lr),
        "instanton_action": instanton["action"],
        "optimization_success": instanton["success"],
        "optimization_message": str(instanton["message"]),
        "x_star": instanton["x"],
        "v_star": instanton["v"],
        "controls": controls,
    }
