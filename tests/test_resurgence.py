import numpy as np

from ftfinance.resurgence import (
    GaussianTailModel,
    gaussian_tail_exact,
    gaussian_tail_borel_pade,
    gaussian_tail_fluctuation_coefficients,
    large_order_action_estimates,
)


def test_gaussian_borel_pade_matches_exact_tail():
    model = GaussianTailModel(mean_return=0.0, sigma=0.2, T=1.0, threshold=-0.3)
    for epsilon in (0.1, 0.2, 0.4):
        exact = float(gaussian_tail_exact(epsilon, model))
        resumed = float(np.real(gaussian_tail_borel_pade(epsilon, model, n_coeffs=18)))
        assert abs(resumed - exact) / exact < 2e-5


def test_large_order_recovers_borel_singularity():
    model = GaussianTailModel(mean_return=0.0, sigma=0.2, T=1.0, threshold=-0.3)
    coeffs = gaussian_tail_fluctuation_coefficients(50, model.instanton_action)
    estimates = large_order_action_estimates(coeffs)
    assert np.isclose(estimates[-1], -model.instanton_action, rtol=0.03)


def test_quartic_factor_borel_pade_matches_exact():
    from ftfinance.resurgence import quartic_factor_exact, quartic_factor_borel_pade

    for g in (0.01, 0.05, 0.15):
        exact = float(quartic_factor_exact(g))
        resumed = float(np.real(quartic_factor_borel_pade(g, n_coeffs=18)))
        assert abs(resumed - exact) / exact < 5e-6


def test_quartic_large_order_detects_complex_saddle():
    from ftfinance.resurgence import quartic_factor_coefficients, quartic_saddle_action

    coeffs = quartic_factor_coefficients(50)
    estimates = large_order_action_estimates(coeffs)
    assert np.isclose(estimates[-1], quartic_saddle_action(), rtol=0.04)


def test_heston_affine_rate_matches_hamiltonian_action():
    from ftfinance.heston import HestonParams
    from ftfinance.heston_resurgence import find_heston_tail_saddle
    from ftfinance.hamiltonian import solve_hamiltonian_shooting

    params = HestonParams(mu=0.0, kappa=2.0, theta=0.04, xi=0.45, rho=-0.7, v0=0.04)
    saddle = find_heston_tail_saddle(params, -0.25, T=1.0, dps=60)
    instanton = solve_hamiltonian_shooting(params, -0.25, T=1.0, n_steps=300)
    assert abs(instanton["action"] - saddle.rate) / saddle.rate < 5e-6
    assert abs(instanton["px"][0] - saddle.p_star) < 2e-8


def test_heston_large_order_recovers_nonperturbative_scale():
    from ftfinance.heston import HestonParams
    from ftfinance.heston_resurgence import (
        find_heston_tail_saddle,
        heston_large_order_extrapolation,
        heston_tail_fluctuation_coefficients,
    )

    params = HestonParams(mu=0.0, kappa=2.0, theta=0.04, xi=0.45, rho=-0.7, v0=0.04)
    saddle = find_heston_tail_saddle(params, -0.25, T=1.0, dps=60)
    coeffs, _ = heston_tail_fluctuation_coefficients(params, -0.25, T=1.0, n_terms=18, dps=75, saddle=saddle)
    estimate, _ = heston_large_order_extrapolation(coeffs, start_order=8, degree=2)
    assert np.isclose(estimate, -saddle.rate, rtol=8e-4)


def test_heston_borel_pade_reconstructs_affine_tail():
    from ftfinance.heston import HestonParams
    from ftfinance.heston_resurgence import (
        find_heston_tail_saddle,
        heston_tail_borel_pade_fluctuation,
        heston_tail_exact,
        heston_tail_fluctuation_coefficients,
    )

    params = HestonParams(mu=0.0, kappa=2.0, theta=0.04, xi=0.45, rho=-0.7, v0=0.04)
    saddle = find_heston_tail_saddle(params, -0.25, T=1.0, dps=60)
    coeffs, _ = heston_tail_fluctuation_coefficients(params, -0.25, T=1.0, n_terms=18, dps=75, saddle=saddle)
    _, exact_fluctuation = heston_tail_exact(0.2, params, -0.25, T=1.0, saddle=saddle)
    resumed = float(np.real(heston_tail_borel_pade_fluctuation(0.2, coeffs)))
    assert abs(resumed - exact_fluctuation) / exact_fluctuation < 2e-5


def test_heston_adjacent_sheet_fluctuation_sector_is_normalized():
    from ftfinance.heston import HestonParams
    from ftfinance.heston_resurgence import (
        find_heston_tail_saddle,
        heston_adjacent_sheet_fluctuation_coefficients,
    )

    params = HestonParams(mu=0.0, kappa=2.0, theta=0.04, xi=0.45, rho=-0.7, v0=0.04)
    saddle = find_heston_tail_saddle(params, -0.25, T=1.0, dps=60)
    sector = heston_adjacent_sheet_fluctuation_coefficients(params, -0.25, T=1.0, n_terms=3, dps=75, reference_saddle=saddle)
    coeffs = sector["coefficients"]
    assert np.isclose(coeffs[0].real, 1.0, atol=1e-12)
    assert abs(coeffs[0].imag) < 1e-12
    assert np.isclose(coeffs[1].real, 0.2792644914, rtol=2e-8)
    assert sector["singulant"].real > 0
    assert sector["singulant"].imag > 0


def test_effective_stokes_fit_recovers_synthetic_branch_strength():
    from ftfinance.heston_resurgence import fit_heston_effective_stokes_multiplier

    w1 = 0.53 + 0.12j
    orders = np.arange(60)
    model = np.zeros(60, dtype=complex)
    model[1:] = -1.0 / (orders[1:] * w1 ** orders[1:])
    K_true = 0.026 + 0.0015j
    values = 2 * np.real(K_true * model)
    ratio = -0.17j
    fit = fit_heston_effective_stokes_multiplier(values, model, ratio, w1, start_order=20, end_order=59)
    assert abs(fit["K"] - K_true) < 1e-10
    assert fit["nrmse"] < 1e-10


def test_conformal_pade_residue_matches_exact_rational_pole():
    from ftfinance.heston_resurgence import heston_conformal_pade_pole_residues

    rate = 1.0
    w0 = 0.3
    coeffs = np.array([w0 ** (-n) for n in range(4)], dtype=float)
    rows = heston_conformal_pade_pole_residues(coeffs, rate, n_coeffs=4, scale=0.5, dps=70)
    row = min(rows, key=lambda item: abs(item[0] - w0))
    w, zeta, residue = row
    expected_zeta = 4.0 * rate * w0 / (1.0 - w0) ** 2
    expected_residue = -w0 * 4.0 * rate * (1.0 + w0) / (1.0 - w0) ** 3
    assert abs(w - w0) < 1e-10
    assert abs(zeta - expected_zeta) < 1e-10
    assert abs(residue - expected_residue) < 1e-9


def test_heston_picard_lefschetz_wall_crossing():
    from ftfinance.heston import HestonParams
    from ftfinance.heston_resurgence import heston_picard_lefschetz_scan

    params = HestonParams(mu=0.0, kappa=2.0, theta=0.04, xi=0.45, rho=-0.7, v0=0.04)
    scan = heston_picard_lefschetz_scan(
        params,
        -0.25,
        T=1.0,
        contour_real=-1.0,
        angle_offsets=(-0.02, 0.0, 0.02),
        tau_max=60.0,
    )
    rows = {round(row["offset"], 2): row for row in scan["rows"]}
    assert rows[-0.02]["intersection_number"] == 0
    assert rows[0.02]["intersection_number"] == -1
    assert rows[0.02]["n_intersections"] == 1
    assert rows[0.0]["min_distance_to_physical_saddle"] < 0.01


def test_heston_adjacent_monodromy_survives_parameter_variation():
    import math
    from ftfinance.heston import HestonParams
    from ftfinance.heston_resurgence import find_heston_tail_saddle, find_heston_first_adjacent_sheet_saddle

    cases = [
        HestonParams(mu=0.0, kappa=2.0, theta=0.04, xi=0.8, rho=-0.7, v0=0.04),
        HestonParams(mu=0.0, kappa=0.75, theta=0.04, xi=0.45, rho=-0.7, v0=0.04),
    ]
    for params in cases:
        physical = find_heston_tail_saddle(params, -0.25, T=1.0, dps=55)
        adjacent = find_heston_first_adjacent_sheet_saddle(params, -0.25, T=1.0, dps=70, reference_saddle=physical)
        predicted = 2.0 * math.pi * params.kappa * params.theta / params.xi**2
        assert np.isclose(adjacent["singulant"].imag, predicted, rtol=2e-12, atol=2e-12)


def test_heston_generalization_detects_antistokes_sign_changes():
    from ftfinance.heston import HestonParams
    from ftfinance.heston_resurgence import find_heston_tail_saddle, find_heston_first_adjacent_sheet_saddle

    def delta_real(params, T=1.0):
        physical = find_heston_tail_saddle(params, -0.25, T=T, dps=45)
        adjacent = find_heston_first_adjacent_sheet_saddle(params, -0.25, T=T, dps=55, reference_saddle=physical, p_min=-140)
        return adjacent["singulant"].real

    base = dict(mu=0.0, kappa=2.0, theta=0.04, xi=0.45, rho=-0.7, v0=0.04)
    params = HestonParams(**base)
    assert delta_real(params, T=2.5) > 0
    assert delta_real(params, T=2.833333333333333) < 0

    p_low = HestonParams(**{**base, "kappa": 5.4523809523809526})
    p_high = HestonParams(**{**base, "kappa": 5.761904761904762})
    assert delta_real(p_low) > 0
    assert delta_real(p_high) < 0


def test_heston_fold_normal_form_and_airy_scaling():
    from ftfinance.heston import HestonParams
    from ftfinance.heston_uniform import find_heston_rho_fold, heston_fold_airy_coordinate
    p=HestonParams(mu=0.0,kappa=2.0,theta=0.04,xi=0.45,rho=-0.7,v0=0.04)
    fold=find_heston_rho_fold(p,-0.25,T=1.0,guess=(-13.47,0.03),dps=45)
    assert abs(fold.rho_c-0.02994951567746)<2e-10
    assert abs(fold.p_c+13.470910377298)<2e-9
    assert fold.alpha<0
    assert abs(fold.airy_slope+1.093268490501)<2e-9
    assert abs(heston_fold_airy_coordinate(fold.rho_c,0.05,fold))<1e-12


def test_heston_postcaustic_pair_stays_on_one_logarithmic_sheet():
    import math
    from ftfinance.heston import HestonParams
    from ftfinance.heston_uniform import find_heston_rho_fold, heston_adjacent_pair_through_fold

    base = dict(mu=0.0, kappa=2.0, theta=0.04, xi=0.45, v0=0.04)
    ref = HestonParams(rho=-0.7, **base)
    fold = find_heston_rho_fold(ref, -0.25, T=1.0, guess=(-13.47, 0.03), dps=45)
    params = HestonParams(rho=0.05, **base)
    pair = heston_adjacent_pair_through_fold(params, -0.25, T=1.0, fold=fold, dps=58)
    predicted = 2.0 * math.pi * params.kappa * params.theta / params.xi**2
    assert pair.p1.imag > 0 and pair.p2.imag < 0
    assert abs(pair.p1 - pair.p2.conjugate()) < 1e-9
    assert abs(pair.average_singulant.imag - predicted) < 1e-11
    assert pair.cfu_zeta.real < 0
    assert abs(pair.cfu_zeta.imag) < 1e-12


def test_airy_connection_matrices_are_unipotent():
    import numpy as np
    from ftfinance.heston_uniform import airy_stokes_matrices

    mats = airy_stokes_matrices()
    for key in ('wkb_upper', 'wkb_lower', 'thimble_upper', 'thimble_lower'):
        assert np.isclose(np.linalg.det(mats[key]), 1.0)
    assert mats['wkb_upper'][0,1] == 1j
    assert mats['wkb_lower'][1,0] == 1j
    assert mats['thimble_upper'][0,1] == 1
    assert mats['thimble_lower'][1,0] == 1


def test_postcaustic_complex_fluctuation_sectors_are_conjugate():
    from ftfinance.heston import HestonParams
    from ftfinance.heston_uniform import find_heston_rho_fold, heston_adjacent_pair_through_fold
    from ftfinance.heston_resurgence import heston_complex_saddle_fluctuation_coefficients

    base = dict(mu=0.0, kappa=2.0, theta=0.04, xi=0.45, v0=0.04)
    ref = HestonParams(rho=-0.7, **base)
    fold = find_heston_rho_fold(ref, -0.25, T=1.0, guess=(-13.47, 0.03), dps=42)
    params = HestonParams(rho=0.05, **base)
    pair = heston_adjacent_pair_through_fold(params, -0.25, T=1.0, fold=fold, dps=55)
    upper = heston_complex_saddle_fluctuation_coefficients(params, -0.25, pair.p1, n_terms=3, dps=75, radius=0.05)
    lower = heston_complex_saddle_fluctuation_coefficients(params, -0.25, pair.p2, n_terms=3, dps=75, radius=0.05)
    assert np.allclose(lower['coefficients'], np.conjugate(upper['coefficients']), rtol=2e-10, atol=2e-10)
    assert abs(lower['lambda_second'] - upper['lambda_second'].conjugate()) < 1e-10


def test_airy_uniform_reexpands_to_one_and_two_saddle_limits():
    from scipy.special import airy
    from ftfinance.heston_uniform import airy_one_saddle_asymptotic, airy_two_saddle_asymptotic

    zpos = 6.0
    zneg = -6.0
    exact_pos = airy(zpos)[0]
    exact_neg = airy(zneg)[0]
    approx_pos = airy_one_saddle_asymptotic(np.array([zpos]))[0]
    approx_neg = airy_two_saddle_asymptotic(np.array([zneg]))[0]
    assert abs((approx_pos - exact_pos) / exact_pos) < 0.01
    # Avoid normalising by Ai(-x), which can be arbitrarily close to a zero.
    envelope = 1.0 / (np.sqrt(np.pi) * abs(zneg) ** 0.25)
    assert abs(approx_neg - exact_neg) / envelope < 0.02
