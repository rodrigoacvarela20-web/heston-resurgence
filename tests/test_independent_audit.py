"""Independent mathematical checks; not evidence of peer review."""
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from ftfinance.heston import HestonParams, estimate_left_tail_probability, simulate_heston
from ftfinance.heston_resurgence import heston_scaled_cgf


def test_affine_cgf_against_independently_integrated_complex_riccati():
    params = HestonParams(mu=0.01, kappa=1.6, theta=0.05, xi=0.42, rho=-0.45, v0=0.07)
    for p in (-1.25, -0.25, 0.3, 0.6 + 0.2j):
        def rhs(_, y):
            D = y[0]
            return [(p*p-p)/2 + (params.rho*params.xi*p-params.kappa)*D + params.xi**2*D**2/2,
                    params.mu*p + params.kappa*params.theta*D]
        sol = solve_ivp(rhs, (0, 0.85), [0j, 0j], rtol=2e-11, atol=1e-12)
        assert sol.success
        D, C = sol.y[:, -1]
        np.testing.assert_allclose(heston_scaled_cgf(p, params, T=0.85), C + params.v0*D, rtol=2e-9, atol=2e-10)


def test_weighted_estimator_ignores_huge_non_event_weights():
    estimate, se = estimate_left_tail_probability(np.array([2., -1.]), 0., np.array([1000., 0.]))
    assert estimate == pytest.approx(0.5)
    assert se == pytest.approx(0.5)


def test_estimator_rejects_mismatched_weights_and_single_path():
    with pytest.raises(ValueError):
        estimate_left_tail_probability([0., 1.], 0., [0.])
    with pytest.raises(ValueError):
        estimate_left_tail_probability([0.], 0.)


def test_parameters_and_controls_are_validated():
    with pytest.raises(ValueError):
        HestonParams(rho=1.01)
    with pytest.raises(ValueError):
        HestonParams(xi=0.)
    with pytest.raises(ValueError):
        simulate_heston(HestonParams(), controls=np.array([[np.nan, 0.]]))


def test_simulation_zero_controls_unit_weights_and_finite_paths():
    x, v, w = simulate_heston(HestonParams(), n_steps=10, n_paths=50, seed=42)
    assert x.shape == v.shape == (50, 11)
    assert np.all(np.isfinite(x)) and np.all(v >= 0) and np.all(w == 0)
