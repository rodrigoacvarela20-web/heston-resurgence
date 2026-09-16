from pathlib import Path
import csv
import math
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from ftfinance.heston import HestonParams
from ftfinance.heston_uniform import (
    find_heston_rho_fold, heston_adjacent_pair_through_fold, airy_stokes_matrices,
    heston_fold_uniform_shape, heston_fold_uniform_asymptotic_shape,
)
from ftfinance.heston_resurgence import (
    find_heston_tail_saddle,
    heston_tail_fluctuation_coefficients,
    heston_conformal_borel_coefficients,
    heston_conformal_pade_poles,
    fit_heston_secondary_singulant,
    heston_complex_saddle_fluctuation_coefficients,
)

RESULTS = ROOT / 'results'; FIGURES = ROOT / 'figures'
RESULTS.mkdir(exist_ok=True); FIGURES.mkdir(exist_ok=True)
X = -0.25
BASE = dict(mu=0.0, kappa=2.0, theta=0.04, xi=0.45, v0=0.04)


def main():
    ref = HestonParams(rho=-0.7, **BASE)
    fold = find_heston_rho_fold(ref, X, T=1.0, guess=(-13.47, 0.03), dps=60)
    rhos = np.concatenate([
        np.linspace(fold.rho_c - 0.055, fold.rho_c - 0.002, 18),
        np.linspace(fold.rho_c + 0.0001, fold.rho_c + 0.16, 34),
    ])
    rows = []
    for rho in rhos:
        params = HestonParams(rho=float(rho), **BASE)
        pair = heston_adjacent_pair_through_fold(params, X, T=1.0, fold=fold, dps=68)
        rows.append([
            rho, pair.p1.real, pair.p1.imag, pair.p2.real, pair.p2.imag,
            pair.singulant_1.real, pair.singulant_1.imag,
            pair.singulant_2.real, pair.singulant_2.imag,
            pair.average_singulant.real, pair.average_singulant.imag,
            pair.cfu_zeta.real, pair.cfu_zeta.imag,
            fold.airy_slope * (rho - fold.rho_c), pair.monodromy,
        ])
    arr = np.asarray(rows, dtype=float)
    np.savetxt(
        RESULTS / 'heston_postcaustic_continuation.csv', arr, delimiter=',',
        header='rho,p1_re,p1_im,p2_re,p2_im,d1_re,d1_im,d2_re,d2_im,dbar_re,dbar_im,cfu_zeta_re,cfu_zeta_im,local_zeta,monodromy', comments=''
    )

    plt.figure(figsize=(7.6,5.0))
    plt.plot(arr[:,0], arr[:,1], label=r'${\rm Re}\,p_1$')
    plt.plot(arr[:,0], arr[:,3], '--', label=r'${\rm Re}\,p_2$')
    plt.plot(arr[:,0], arr[:,2], label=r'${\rm Im}\,p_1$')
    plt.plot(arr[:,0], arr[:,4], '--', label=r'${\rm Im}\,p_2$')
    plt.axvline(fold.rho_c, linewidth=.9)
    plt.xlabel(r'$\rho$'); plt.ylabel('continued saddle coordinates')
    plt.title('Real adjacent saddles continue into a complex-conjugate pair')
    plt.legend(fontsize=8, ncol=2); plt.tight_layout()
    plt.savefig(FIGURES / 'heston_postcaustic_saddle_continuation.png', dpi=240); plt.close()

    plt.figure(figsize=(7.6,5.0))
    plt.plot(arr[:,0], arr[:,6], label=r'${\rm Im}\,\Delta S_1$')
    plt.plot(arr[:,0], arr[:,8], '--', label=r'${\rm Im}\,\Delta S_2$ (lifted)')
    plt.plot(arr[:,0], arr[:,10], ':', linewidth=2, label=r'${\rm Im}\,\overline{\Delta S}$')
    plt.axhline(arr[0,14], linewidth=.8, label=r'$2\pi\kappa\theta/\xi^2$')
    plt.axvline(fold.rho_c, linewidth=.9)
    plt.xlabel(r'$\rho$'); plt.ylabel('imaginary action')
    plt.title('Full-sheet action continuation preserves the monodromy center')
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(FIGURES / 'heston_postcaustic_lifted_actions.png', dpi=240); plt.close()

    plt.figure(figsize=(7.6,5.0))
    plt.plot(arr[:,0], arr[:,11], label='exact CFU coordinate from saddle actions')
    plt.plot(arr[:,0], arr[:,13], '--', label='local linear fold coordinate')
    plt.axvline(fold.rho_c, linewidth=.9)
    plt.axhline(0, linewidth=.7)
    plt.xlabel(r'$\rho$'); plt.ylabel(r'$\zeta$')
    plt.title('CFU coordinate continues from positive to negative through the fold')
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(FIGURES / 'heston_cfu_coordinate_through_fold.png', dpi=240); plt.close()

    # Independent physical-sector Borel test beyond the fold.  The physical
    # perturbative series is real, so the two projected Borel singularities
    # occur as a conjugate pair even though the full-sheet actions are lifted
    # around the common logarithmic monodromy level.
    borel_rows = []
    fig, ax = plt.subplots(figsize=(7.6,5.4))
    for idx_case, (rho, marker) in enumerate([(0.04, 'o'), (0.05, 's'), (0.08, '^')]):
        color = ax._get_lines.get_next_color()
        params = HestonParams(rho=rho, **BASE)
        pair = heston_adjacent_pair_through_fold(params, X, T=1.0, fold=fold, dps=70)
        physical = find_heston_tail_saddle(params, X, T=1.0, dps=58)
        coeffs, _ = heston_tail_fluctuation_coefficients(params, X, T=1.0, n_terms=32, dps=105, saddle=physical)
        conformal = heston_conformal_borel_coefficients(coeffs, physical.rate, dps=105)
        _, zeta_poles = heston_conformal_pade_poles(conformal, physical.rate, n_coeffs=32, scale=0.55)
        target = pair.singulant_1
        idx = int(np.argmin(np.abs(zeta_poles - target)))
        pole = complex(zeta_poles[idx])
        rel = abs(pole-target)/abs(target)
        fit = fit_heston_secondary_singulant(conformal, physical.rate, target, start_order=20, end_order=31)
        fit_zeta = complex(fit["zeta"])
        fit_rel = abs(fit_zeta-target)/abs(target)
        borel_rows.append([rho, target.real, target.imag, pole.real, pole.imag, rel, fit_zeta.real, fit_zeta.imag, fit_rel, fit["weighted_rms"]])
        ax.scatter([pole.real, pole.real], [pole.imag, -pole.imag], marker=marker, s=42, color=color, facecolors='none', label=rf'Padé $\rho={rho:g}$')
        ax.scatter([fit_zeta.real, fit_zeta.real], [fit_zeta.imag, -fit_zeta.imag], marker='+', s=70, color=color, label='late-order fit' if idx_case == 0 else None)
        ax.scatter([target.real, target.real], [target.imag, -target.imag], marker='x', s=60, color=color, label='continued saddle' if idx_case == 0 else None)
    ax.axhline(0, linewidth=.7)
    ax.set_xlabel(r'${\rm Re}\,\zeta$'); ax.set_ylabel(r'${\rm Im}\,\zeta$')
    ax.set_title('Post-caustic Borel singularities track the projected complex saddles')
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(FIGURES / 'heston_postcaustic_borel_pair.png', dpi=240); plt.close(fig)
    np.savetxt(RESULTS / 'heston_postcaustic_borel_check.csv', np.asarray(borel_rows), delimiter=',',
               header='rho,target_re,target_im,pade_re,pade_im,pade_relative_error,latefit_re,latefit_im,latefit_relative_error,latefit_weighted_rms', comments='')


    # Cross-maturity spot checks of the same full-sheet continuation.
    cross_rows = []
    guess = (fold.p_c, fold.rho_c)
    for TT in (0.8, 1.0, 1.5, 2.0):
        fT = find_heston_rho_fold(ref, X, T=TT, guess=guess, dps=55)
        guess = (fT.p_c, fT.rho_c)
        rho = fT.rho_c + 0.03
        params = HestonParams(rho=rho, **BASE)
        pair = heston_adjacent_pair_through_fold(params, X, T=TT, fold=fT, dps=65)
        cross_rows.append([TT, fT.rho_c, rho, pair.average_singulant.imag, pair.monodromy, pair.cfu_zeta.real, pair.cfu_zeta.imag])
    np.savetxt(RESULTS / 'heston_postcaustic_cross_maturity.csv', np.asarray(cross_rows), delimiter=',',
               header='T,rho_c,rho_test,average_singulant_imag,monodromy,cfu_zeta_real,cfu_zeta_imag', comments='')

    # Local fluctuation sectors of the post-caustic complex pair.  Since all
    # model parameters are real, the projected pair must carry conjugate local
    # fluctuation series.  This tests the sector data, not just the saddle
    # locations.
    params_complex = HestonParams(rho=0.05, **BASE)
    pair_complex = heston_adjacent_pair_through_fold(params_complex, X, T=1.0, fold=fold, dps=68)
    upper = heston_complex_saddle_fluctuation_coefficients(params_complex, X, pair_complex.p1, T=1.0, n_terms=6, dps=95, radius=0.05)
    lower = heston_complex_saddle_fluctuation_coefficients(params_complex, X, pair_complex.p2, T=1.0, n_terms=6, dps=95, radius=0.05)
    fluct_rows = []
    for n, (cu, cl) in enumerate(zip(upper['coefficients'], lower['coefficients'])):
        fluct_rows.append([n, cu.real, cu.imag, cl.real, cl.imag, abs(cl-cu.conjugate())])
    np.savetxt(RESULTS / 'heston_postcaustic_complex_fluctuations.csv', np.asarray(fluct_rows), delimiter=',',
               header='order,upper_re,upper_im,lower_re,lower_im,conjugacy_error', comments='')

    # Exact Airy block versus its asymptotic saddle decompositions.  This is
    # the canonical connection test: one decaying real saddle for Z>0 and the
    # coherent sum of two complex saddles for Z<0.
    epsilon_airy = 0.004
    rho_grid = np.linspace(fold.rho_c - 0.13, fold.rho_c + 0.13, 700)
    exact_shape = []; asym_shape = []; zvals = []; regimes = []
    for rr in rho_grid:
        exact, _, z = heston_fold_uniform_shape(rr, epsilon_airy, fold)
        approx, z2, regime = heston_fold_uniform_asymptotic_shape(rr, epsilon_airy, fold)
        exact_shape.append(exact); asym_shape.append(approx); zvals.append(z); regimes.append(regime)
    exact_shape=np.asarray(exact_shape); asym_shape=np.asarray(asym_shape); zvals=np.asarray(zvals)
    plt.figure(figsize=(8.0,5.1))
    plt.plot(zvals, exact_shape, linewidth=1.8, label=r'uniform $\varepsilon^{1/3}{\rm Ai}(Z)$')
    mask_neg=zvals<=-3.0
    mask_pos=zvals>=3.0
    plt.plot(zvals[mask_neg], asym_shape[mask_neg], '--', linewidth=1.2, label='large-|Z| saddle re-expansion')
    plt.plot(zvals[mask_pos], asym_shape[mask_pos], '--', linewidth=1.2)
    plt.axvline(0, linewidth=.8)
    plt.xlabel(r'$Z=\zeta/\varepsilon^{2/3}$'); plt.ylabel('canonical fold amplitude')
    plt.title('Airy block re-expands into real saddles before the fold and complex saddles after it')
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(FIGURES / 'heston_postcaustic_airy_matching.png', dpi=240); plt.close()

    # Error away from the fold, normalized by a non-vanishing Airy envelope on
    # the oscillatory side to avoid artificial spikes at Ai zeros.
    good=np.abs(zvals)>=4.0
    err=[]
    for z,e,a in zip(zvals[good], exact_shape[good], asym_shape[good]):
        if z>0:
            denom=max(abs(e),1e-300)
        else:
            denom=(epsilon_airy**(1/3))/(np.sqrt(np.pi)*abs(z)**0.25)
        err.append(abs(a-e)/denom)
    airy_max_err=float(np.max(err))

    matrices = airy_stokes_matrices()
    with open(RESULTS / 'heston_airy_connection_matrices.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['basis','matrix'])
        for key, mat in matrices.items():
            w.writerow([key, np.array2string(mat, separator=' ')])

    # A compact summary for the manuscript and regression tests.
    max_center = float(np.max(np.abs(arr[:,10] - arr[:,14])))
    post = arr[arr[:,0] > fold.rho_c]
    max_zeta_im = float(np.max(np.abs(post[:,12])))
    borel_np = np.asarray(borel_rows)
    max_borel = float(np.max(borel_np[:,5]))
    max_latefit = float(np.max(borel_np[:,8]))
    max_fluct_conj = float(np.max(np.asarray(fluct_rows)[:,-1]))
    with open(RESULTS / 'heston_postcaustic_summary.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['quantity','value'])
        w.writerow(['rho_c',fold.rho_c]); w.writerow(['p_c',fold.p_c])
        w.writerow(['max_monodromy_center_error',max_center])
        w.writerow(['max_postcaustic_cfu_imaginary_part',max_zeta_im])
        w.writerow(['max_postcaustic_borel_relative_error_32_coeffs',max_borel])
        w.writerow(['max_postcaustic_latefit_relative_error_32_coeffs',max_latefit])
        w.writerow(['max_complex_fluctuation_conjugacy_error',max_fluct_conj])
        w.writerow(['max_airy_reexpansion_error_absZ_ge_4',airy_max_err])
    print('fold',fold.rho_c,fold.p_c)
    print('max monodromy-center error',max_center)
    print('max CFU Im after fold',max_zeta_im)
    print('max post-caustic Borel error',max_borel)
    print('max post-caustic late-fit error',max_latefit)
    print('max complex fluctuation conjugacy error',max_fluct_conj)
    print('max Airy re-expansion error |Z|>=4',airy_max_err)

if __name__ == '__main__':
    main()
