from pathlib import Path
import csv
import sys

import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np
from scipy.special import airy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from ftfinance.heston import HestonParams
from ftfinance.heston_uniform import find_heston_rho_fold, heston_fold_airy_coordinate, airy_one_saddle_asymptotic

RESULTS = ROOT / 'results'; FIGURES = ROOT / 'figures'
RESULTS.mkdir(exist_ok=True); FIGURES.mkdir(exist_ok=True)
BASE = HestonParams(mu=0.0, kappa=2.0, theta=0.04, xi=0.45, rho=-0.7, v0=0.04)
X = -0.25


def main():
    fold = find_heston_rho_fold(BASE, X, T=1.0, guess=(-13.47, 0.03), dps=75)

    # Verify the root and action 1/2 and 3/2 laws directly with the local
    # cubic normal form. These are the catastrophe-theory invariants that
    # justify replacing separate Gaussian saddles by one Airy sector.
    dr = np.logspace(-7, -2.4, 34)
    predicted_split = 2.0 * fold.split_coefficient * np.sqrt(dr)
    predicted_action = fold.action_split_coefficient * dr ** 1.5

    # High-precision exact roots/actions near the fold.
    old = mp.mp.dps; mp.mp.dps = 65
    exact_rows = []
    try:
        kappa=mp.mpf('2'); theta=mp.mpf('.04'); xi=mp.mpf('.45'); v0=mp.mpf('.04'); mu=mp.mpf('0'); T=mp.mpf('1'); x=mp.mpf('-.25')
        def lam(q,rho):
            B=kappa-rho*xi*q; gam=mp.sqrt(B*B-xi*xi*(q*q-q)); g=(B-gam)/(B+gam); e=mp.exp(-gam*T)
            D=(B-gam)/(xi*xi)*(1-e)/(1-g*e)
            C=mu*q*T+(kappa*theta/(xi*xi))*((B-gam)*T-2*mp.log((1-g*e)/(1-g)))
            return C+v0*D
        def F(q,rho): return mp.diff(lambda z:lam(z,rho),q)-x
        pc=mp.mpf(str(fold.p_c)); rc=mp.mpf(str(fold.rho_c)); Csplit=mp.mpf(str(fold.split_coefficient))
        for d in dr:
            dm=mp.mpf(str(d)); rho=rc-dm; shift=Csplit*mp.sqrt(dm)
            pminus=mp.findroot(lambda q:F(q,rho),(pc-shift*mp.mpf('1.02'),pc-shift*mp.mpf('.98')),tol=mp.mpf('1e-42'))
            pplus=mp.findroot(lambda q:F(q,rho),(pc+shift*mp.mpf('.98'),pc+shift*mp.mpf('1.02')),tol=mp.mpf('1e-42'))
            phiminus=lam(pminus,rho)-pminus*x; phiplus=lam(pplus,rho)-pplus*x
            exact_rows.append([float(d), float(mp.re(pplus-pminus)), float(abs(mp.re(phiplus-phiminus)))])
    finally:
        mp.mp.dps = old
    exact = np.asarray(exact_rows)

    with open(RESULTS/'heston_uniform_airy_fold_summary.csv','w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['quantity','value'])
        for key in ['rho_c','p_c','phi_prho','phi_ppp','alpha','split_coefficient','airy_slope','action_split_coefficient']:
            w.writerow([key,getattr(fold,key)])
    np.savetxt(RESULTS/'heston_uniform_airy_scaling.csv', exact, delimiter=',', header='rho_distance,exact_saddle_separation,exact_action_separation', comments='')

    plt.figure(figsize=(7.2,4.8))
    plt.loglog(exact[:,0], exact[:,1], 'o', ms=3.2, label='exact saddle separation')
    plt.loglog(dr, predicted_split, '--', linewidth=1.2, label=r'$2C_\rho(\rho_c-\rho)^{1/2}$')
    plt.xlabel(r'$\rho_c-\rho$'); plt.ylabel(r'$|p_+-p_-|$')
    plt.title('Square-root saddle coalescence at the Heston fold')
    plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(FIGURES/'heston_airy_saddle_scaling.png', dpi=220); plt.close()

    plt.figure(figsize=(7.2,4.8))
    plt.loglog(exact[:,0], exact[:,2], 'o', ms=3.2, label='exact action splitting')
    plt.loglog(dr, predicted_action, '--', linewidth=1.2, label=r'$C_S(\rho_c-\rho)^{3/2}$')
    plt.xlabel(r'$\rho_c-\rho$'); plt.ylabel(r'$|\Phi(p_+)-\Phi(p_-)|$')
    plt.title('Three-halves action law at the Heston fold')
    plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(FIGURES/'heston_airy_action_scaling.png', dpi=220); plt.close()

    # Show the actual Heston Airy boundary layer for several noise strengths.
    plt.figure(figsize=(7.4,5.0))
    for eps in (0.02,0.05,0.10):
        width = 5.0 * eps**(2/3) / abs(fold.airy_slope)
        rhos = np.linspace(fold.rho_c-width, fold.rho_c+width, 500)
        z = np.asarray([heston_fold_airy_coordinate(r,eps,fold) for r in rhos])
        Ai = airy(z)[0]
        # Normalize only by epsilon^(1/3); the common exponential is omitted.
        plt.plot((rhos-fold.rho_c)/eps**(2/3), Ai, label=rf'$\varepsilon={eps:g}$')
    plt.axvline(0, linewidth=.8)
    plt.xlabel(r'$(\rho-\rho_c)/\varepsilon^{2/3}$'); plt.ylabel(r'${\rm Ai}(\zeta/\varepsilon^{2/3})$')
    plt.title('Uniform Airy boundary layer through the Heston caustic')
    plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(FIGURES/'heston_uniform_airy_boundary_layer.png', dpi=220); plt.close()

    # Demonstrate why the Gaussian approximation fails: Ai is finite at z=0,
    # while its isolated-saddle asymptotic diverges as z^{-1/4}.
    z=np.logspace(-4,1.1,500); Ai=airy(z)[0]; gas=airy_one_saddle_asymptotic(z)
    plt.figure(figsize=(7.2,4.8)); plt.loglog(z,np.abs(Ai),label='uniform Airy'); plt.loglog(z,np.abs(gas),'--',label='isolated Gaussian saddle')
    plt.xlabel(r'$z=\zeta/\varepsilon^{2/3}$'); plt.ylabel('canonical amplitude')
    plt.title('Airy uniformization removes the fold divergence')
    plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(FIGURES/'heston_uniform_airy_vs_gaussian.png',dpi=220); plt.close()

    print('rho_c', fold.rho_c)
    print('p_c', fold.p_c)
    print('alpha', fold.alpha)
    print('airy_slope', fold.airy_slope)
    print('action_split_coefficient', fold.action_split_coefficient)

if __name__ == '__main__':
    main()
