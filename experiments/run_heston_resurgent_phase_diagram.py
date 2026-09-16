from pathlib import Path
import csv
import cmath
import sys

import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np
from scipy.optimize import brentq, root

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ftfinance.heston import HestonParams
from ftfinance.heston_resurgence import (
    _real_cgf_derivative,
    find_heston_first_adjacent_sheet_saddle,
    find_heston_tail_saddle,
    heston_negative_moment_poles,
    heston_phase_derivative,
    heston_scaled_cgf,
)
RESULTS=ROOT/'results';FIGURES=ROOT/'figures';RESULTS.mkdir(exist_ok=True);FIGURES.mkdir(exist_ok=True)
BASE=dict(mu=0.0,kappa=2.0,theta=0.04,xi=0.45,rho=-0.7,v0=0.04)
X=-0.25

def delta_real_T(T):
    p=HestonParams(**BASE); s=find_heston_tail_saddle(p,X,T=T,dps=45)
    a=find_heston_first_adjacent_sheet_saddle(p,X,T=T,dps=55,reference_saddle=s,p_min=-140)
    return a['singulant'].real

def delta_real_kappa(k):
    d=dict(BASE);d['kappa']=float(k);p=HestonParams(**d);s=find_heston_tail_saddle(p,X,dps=45)
    a=find_heston_first_adjacent_sheet_saddle(p,X,dps=55,reference_saddle=s,p_min=-140)
    return a['singulant'].real

def saddle_node_equations(z):
    pval,rho=z
    par=HestonParams(mu=0,kappa=2,theta=.04,xi=.45,rho=float(rho),v0=.04)
    f=_real_cgf_derivative(float(pval),par,T=1)-X
    h=1e-3*max(1,abs(float(pval)))
    fp=_real_cgf_derivative(float(pval)+h,par,T=1)-X
    fm=_real_cgf_derivative(float(pval)-h,par,T=1)-X
    return [f,(fp-fm)/(2*h)]

def real_adjacent_roots(rho):
    p=HestonParams(mu=0,kappa=2,theta=.04,xi=.45,rho=float(rho),v0=.04)
    poles=heston_negative_moment_poles(p,T=1,n_poles=2,p_min=-80)
    grid=np.linspace(poles[0]-2e-3,poles[1]+2e-3,4500)
    roots=[];prev=None
    for q in grid:
        try:v=_real_cgf_derivative(q,p,T=1)-X
        except Exception: prev=None;continue
        if not np.isfinite(v) or abs(v)>1e4:prev=None;continue
        if prev is not None and prev[1]*v<0:
            try:r=brentq(lambda u:_real_cgf_derivative(u,p,T=1)-X,q,prev[0],xtol=1e-11)
            except Exception:r=None
            if r is not None and (not roots or abs(r-roots[-1])>1e-3):roots.append(float(r))
        prev=(q,v)
    return roots[:2]

def complex_root(rho,guess):
    p=HestonParams(mu=0,kappa=2,theta=.04,xi=.45,rho=float(rho),v0=.04)
    def F(v):
        z=v[0]+1j*v[1]
        q=heston_phase_derivative(z,p,X,T=1,relative_step=2e-6)
        return [q.real,q.imag]
    out=root(F,[guess.real,guess.imag],tol=1e-10)
    if not out.success: raise RuntimeError(out.message)
    return complex(out.x[0],out.x[1])


def fold_normal_form(dps=70):
    """High-precision local normal form at the rho saddle-node.

    For F(p,rho)=Lambda_p(p;rho)-X, the fold obeys F=F_p=0.
    The leading local equation is

        delta p^2 = alpha delta rho,
        alpha = -2 F_rho / F_pp = -2 d_rho Lambda_p / Lambda_ppp.
    """
    old=mp.mp.dps; mp.mp.dps=dps
    try:
        kappa=mp.mpf('2');theta=mp.mpf('.04');xi=mp.mpf('.45');v0=mp.mpf('.04');mu=mp.mpf('0');TT=mp.mpf('1');xx=mp.mpf('-.25')
        def lam(q,rho):
            B=kappa-rho*xi*q
            gam=mp.sqrt(B*B-xi*xi*(q*q-q))
            g=(B-gam)/(B+gam)
            e=mp.exp(-gam*TT)
            D=(B-gam)/(xi*xi)*(1-e)/(1-g*e)
            C=mu*q*TT+(kappa*theta/(xi*xi))*((B-gam)*TT-2*mp.log((1-g*e)/(1-g)))
            return C+v0*D
        def F(q,rho): return mp.diff(lambda z:lam(z,rho),q)-xx
        def Fp(q,rho): return mp.diff(lambda z:lam(z,rho),q,2)
        pc,rc=mp.findroot((F,Fp),(mp.mpf('-13.4709'),mp.mpf('.02995')),tol=mp.mpf('1e-55'),maxsteps=100)
        Frho=mp.diff(lambda r:F(pc,r),rc)
        Fpp=mp.diff(lambda z:lam(z,rc),pc,3)
        alpha=-2*Frho/Fpp
        Cfold=mp.sqrt(-alpha)
        return dict(pc=pc,rhoc=rc,Frho=Frho,Lambda3=Fpp,alpha=alpha,C=Cfold,F=F)
    finally:
        mp.mp.dps=old


def build_fold_scaling_data():
    """Continue the two saddles through the fold and test square-root scaling."""
    old=mp.mp.dps; mp.mp.dps=65
    try:
        nf=fold_normal_form(dps=65);pc=nf['pc'];rc=nf['rhoc'];C=nf['C'];F=nf['F']
        deltas=[mp.mpf(v) for v in ('1e-7','2e-7','5e-7','1e-6','2e-6','5e-6','1e-5','2e-5','5e-5','1e-4','2e-4','5e-4','1e-3')]
        rows=[]
        for side in (-1,1):
            for dr in deltas:
                rho=rc+side*dr
                if side<0:
                    shift=C*mp.sqrt(dr)
                    for branch,guess in ((-1,pc-shift),(1,pc+shift)):
                        z=mp.findroot(lambda q:F(q,rho),(guess*(1-mp.mpf('1e-8')),guess*(1+mp.mpf('1e-8'))),tol=mp.mpf('1e-42'),maxsteps=60)
                        predicted=branch*C*mp.sqrt(dr)
                        rows.append([float(rho),float(mp.re(z)),float(mp.im(z)),branch,float(mp.re(predicted)),0.0,float(dr)])
                else:
                    guess=pc+1j*C*mp.sqrt(dr)
                    z=mp.findroot(lambda q:F(q,rho),(guess,guess*(1+mp.mpf('1e-8'))),tol=mp.mpf('1e-42'),maxsteps=60)
                    if mp.im(z)<0:z=mp.conj(z)
                    pred=C*mp.sqrt(dr)
                    rows.append([float(rho),float(mp.re(z)),float(mp.im(z)),2,0.0,float(pred),float(dr)])
                    rows.append([float(rho),float(mp.re(z)),-float(mp.im(z)),3,0.0,-float(pred),float(dr)])
        arr=np.asarray(rows,float)
        np.savetxt(RESULTS/'heston_rho_fold_scaling.csv',arr,delimiter=',',header='rho,p_real,p_imag,branch,predicted_real_shift,predicted_imag,abs_delta_rho',comments='')
        with open(RESULTS/'heston_rho_fold_normal_form.csv','w',encoding='utf-8') as f:
            f.write('quantity,value\n')
            for name,val in [('rho_c',rc),('p_c',pc),('F_rho',nf['Frho']),('Lambda_3',nf['Lambda3']),('alpha',nf['alpha']),('sqrt_minus_alpha',C)]:
                f.write(f'{name},{mp.nstr(val,40)}\n')

        # Scaled displacement makes the square-root law a horizontal limit.
        plt.figure(figsize=(7.2,4.8))
        left=arr[arr[:,3]<2]
        right=arr[arr[:,3]>=2]
        # absolute displacement from pc for left pair, imaginary displacement for right pair
        xleft=np.sqrt(np.maximum(float(rc)-left[:,0],0)); yleft=np.abs(left[:,1]-float(pc))
        xright=np.sqrt(np.maximum(right[:,0]-float(rc),0)); yright=np.abs(right[:,2])
        plt.plot(xleft,yleft,'o',ms=3,label=r'real splitting $|p-p_c|$')
        plt.plot(xright,yright,'s',ms=3,label=r'complex splitting $|\mathrm{Im}\,p|$')
        xmax=max(xleft.max(),xright.max()); xx=np.linspace(0,xmax,200)
        plt.plot(xx,float(C)*xx,'--',linewidth=1.1,label=rf'$3.217644464\sqrt{{|\rho-\rho_c|}}$')
        plt.xlabel(r'$\sqrt{|\rho-\rho_c|}$');plt.ylabel('saddle displacement')
        plt.title('Square-root fold scaling at the Heston correlation caustic')
        plt.legend(fontsize=8);plt.tight_layout();plt.savefig(FIGURES/'heston_rho_fold_scaling.png',dpi=220);plt.close()
        return nf,arr
    finally:
        mp.mp.dps=old

def read_csv(path):
    with open(path,newline='',encoding='utf-8') as f:return list(csv.DictReader(f))

def main():
    Tc=brentq(delta_real_T,2.5,3.0,xtol=2e-7)
    kc=brentq(delta_real_kappa,4.0,6.0,xtol=2e-7)
    sn=root(saddle_node_equations,[-13.47,.03],tol=1e-10)
    pc,rhoc=sn.x
    with open(RESULTS/'heston_phase_boundaries.csv','w',encoding='utf-8') as f:
        f.write('boundary,value,secondary_value,meaning\n')
        f.write(f'rho_saddle_node,{rhoc:.15g},{pc:.15g},first two adjacent real stationary points coalesce\n')
        f.write(f'T_antistokes,{Tc:.15g},0,Re DeltaS1 changes sign\n')
        f.write(f'kappa_antistokes,{kc:.15g},0,Re DeltaS1 changes sign\n')

    # Extended T and kappa scans for anti-Stokes plots.
    Tvals=np.linspace(.5,4.0,22); Trows=[]
    for T in Tvals:
        p=HestonParams(**BASE);s=find_heston_tail_saddle(p,X,T=T,dps=42);a=find_heston_first_adjacent_sheet_saddle(p,X,T=T,dps=52,reference_saddle=s,p_min=-150)
        Trows.append([T,a['singulant'].real,a['singulant'].imag])
    Kvals=np.linspace(.5,7.0,22); Krows=[]
    for k in Kvals:
        d=dict(BASE);d['kappa']=float(k);p=HestonParams(**d);s=find_heston_tail_saddle(p,X,dps=42);a=find_heston_first_adjacent_sheet_saddle(p,X,dps=52,reference_saddle=s,p_min=-150)
        Krows.append([k,a['singulant'].real,a['singulant'].imag])
    np.savetxt(RESULTS/'heston_antistokes_scans.csv',np.vstack([np.column_stack([np.zeros(len(Trows)),Trows]),np.column_stack([np.ones(len(Krows)),Krows])]),delimiter=',',header='scan,value,delta_real,delta_imag',comments='')

    plt.figure(figsize=(7.2,4.6))
    Trows=np.asarray(Trows);Krows=np.asarray(Krows)
    plt.plot(Trows[:,0],Trows[:,1],'o-',ms=3,label=r'vary $T$')
    plt.axvline(Tc,linestyle='--',linewidth=1,label=rf'$T_A={Tc:.3f}$')
    plt.axhline(0,linewidth=.8)
    plt.xlabel(r'$T$');plt.ylabel(r'${\rm Re}\,\Delta S_1$');plt.title('Maturity anti-Stokes boundary');plt.legend(fontsize=8);plt.tight_layout();plt.savefig(FIGURES/'heston_phase_boundary_T.png',dpi=220);plt.close()
    plt.figure(figsize=(7.2,4.6))
    plt.plot(Krows[:,0],Krows[:,1],'o-',ms=3,label=r'vary $\kappa$')
    plt.axvline(kc,linestyle='--',linewidth=1,label=rf'$\kappa_A={kc:.3f}$')
    plt.axhline(0,linewidth=.8)
    plt.xlabel(r'$\kappa$');plt.ylabel(r'${\rm Re}\,\Delta S_1$');plt.title('Mean-reversion anti-Stokes boundary');plt.legend(fontsize=8);plt.tight_layout();plt.savefig(FIGURES/'heston_phase_boundary_kappa.png',dpi=220);plt.close()

    # Saddle-node / complex continuation in rho.
    rhos_left=np.linspace(-.10,rhoc-.001,15); rho_rows=[]
    for rho in rhos_left:
        roots=real_adjacent_roots(rho)
        if len(roots)==2:
            rho_rows.append([rho,roots[0],0.0,0]);rho_rows.append([rho,roots[1],0.0,1])
    rhos_right=np.linspace(rhoc+.001,.30,18);guess=complex(pc,.05)
    for rho in rhos_right:
        z=complex_root(rho,guess)
        if z.imag<0:z=z.conjugate()
        guess=z
        rho_rows.append([rho,z.real,z.imag,2]);rho_rows.append([rho,z.real,-z.imag,3])
    np.savetxt(RESULTS/'heston_rho_saddle_continuation.csv',np.asarray(rho_rows),delimiter=',',header='rho,p_real,p_imag,branch',comments='')
    plt.figure(figsize=(7.4,4.8))
    arr=np.asarray(rho_rows)
    for branch,label in [(0,'real saddle 1'),(1,'real saddle 2'),(2,'complex $p_+$'),(3,'complex $p_-$')]:
        q=arr[arr[:,3]==branch];plt.plot(q[:,0],q[:,1],'-o',ms=2.5,label=label)
    plt.axvline(rhoc,linestyle='--',linewidth=1,label=rf'$\rho_c={rhoc:.4f}$')
    plt.xlabel(r'$\rho$');plt.ylabel(r'${\rm Re}\,p$');plt.title('Adjacent-saddle collision and complex continuation');plt.legend(fontsize=7);plt.tight_layout();plt.savefig(FIGURES/'heston_rho_saddle_node_real.png',dpi=220);plt.close()
    plt.figure(figsize=(7.4,4.8))
    for branch,label in [(2,'$p_+$'),(3,'$p_-$')]:
        q=arr[arr[:,3]==branch];plt.plot(q[:,0],q[:,2],'-o',ms=2.5,label=label)
    plt.axvline(rhoc,linestyle='--',linewidth=1,label=rf'$\rho_c={rhoc:.4f}$')
    plt.axhline(0,linewidth=.7);plt.xlabel(r'$\rho$');plt.ylabel(r'${\rm Im}\,p$');plt.title('Birth of a complex-conjugate adjacent-saddle pair');plt.legend(fontsize=8);plt.tight_layout();plt.savefig(FIGURES/'heston_rho_saddle_node_imag.png',dpi=220);plt.close()

    # Monodromy identity over the original 25-point real-saddle sweep.
    structural=read_csv(RESULTS/'heston_parameter_sweep_structure.csv')
    measured=np.asarray([float(r['delta_imag']) for r in structural]);pred=np.asarray([float(r['monodromy_pred']) for r in structural])
    plt.figure(figsize=(5.8,5.2));plt.scatter(pred,measured,s=34);lo=min(pred.min(),measured.min());hi=max(pred.max(),measured.max());plt.plot([lo,hi],[lo,hi],'--',linewidth=1);plt.xlabel(r'$2\pi\kappa\theta/\xi^2$');plt.ylabel(r'measured ${\rm Im}\,\Delta S_1$');plt.title('Exact logarithmic monodromy across the real-saddle regime');plt.tight_layout();plt.savefig(FIGURES/'heston_generalization_monodromy.png',dpi=220);plt.close()

    # Selected independent Borel locations against independently continued actions.
    high=read_csv(RESULTS/'heston_selected_highorder_32.csv')
    pts=[]
    for r in high:
        axis=r['axis'];value=float(r['value'])
        d=dict(BASE);threshold=X;T=1.0
        if axis=='xstar':threshold=value
        elif axis=='T':T=value
        else:d[axis]=value
        p=HestonParams(**d);s=find_heston_tail_saddle(p,threshold,T=T,dps=50);a=find_heston_first_adjacent_sheet_saddle(p,threshold,T=T,dps=60,reference_saddle=s,p_min=-120)
        pts.append([a['singulant'].real,a['singulant'].imag,float(r['pade_real']),float(r['pade_imag']),float(r['pade_rel_gap'])])
    pts=np.asarray(pts)
    plt.figure(figsize=(7.1,5.6));plt.scatter(pts[:,0],pts[:,1],s=46,label='adjacent-sheet action');plt.scatter(pts[:,2],pts[:,3],marker='x',s=55,label='32-term conformal Pade')
    for q in pts:plt.plot([q[0],q[2]],[q[1],q[3]],linewidth=.7)
    plt.xlabel(r'${\rm Re}\,\zeta$');plt.ylabel(r'${\rm Im}\,\zeta$');plt.title('Secondary Borel singularity tracks the moving adjacent action');plt.legend(fontsize=8);plt.tight_layout();plt.savefig(FIGURES/'heston_generalization_borel_tracking.png',dpi=220);plt.close()

    nf,fold_data=build_fold_scaling_data()
    print('fold coefficient',float(mp.re(nf['C'])))
    print('rho_c',rhoc,'p_c',pc)
    print('T_A',Tc,'kappa_A',kc)
    print('monodromy max rel error',max(abs(measured-pred)/abs(pred)))
    print('selected Borel median gap',np.median(pts[:,4]),'max gap',np.max(pts[:,4]))

if __name__=='__main__':main()
