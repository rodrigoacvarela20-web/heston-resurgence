from pathlib import Path
import cmath
import csv
import math
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from ftfinance.heston import HestonParams
from ftfinance.heston_resurgence import _real_cgf_derivative, heston_scaled_cgf
from ftfinance.heston_uniform import find_heston_rho_fold

RESULTS = ROOT / 'results'; FIGURES = ROOT / 'figures'
RESULTS.mkdir(exist_ok=True); FIGURES.mkdir(exist_ok=True)
BASE = dict(mu=0.0, kappa=2.0, theta=0.04, xi=0.45, v0=0.04)
X = -0.25


def physical_saddle_fast(params, T):
    right=-1e-8; left=-0.25
    f=lambda q:_real_cgf_derivative(q,params,T=T)-X
    fl=f(left)
    while fl>0:
        left*=1.4
        if left < -80: raise RuntimeError('could not bracket physical saddle')
        fl=f(left)
    return brentq(f,left,right,xtol=2e-11,rtol=2e-11)


def moment_poles_fast(params, T, p_min=-100.0, samples=2600):
    kappa=float(params.kappa);xi=float(params.xi);rho=float(params.rho)
    def den(p):
        p=complex(p);B=kappa-rho*xi*p;gamma=cmath.sqrt(B*B-xi*xi*(p*p-p))
        if gamma.real<0 or (abs(gamma.real)<1e-14 and gamma.imag<0):gamma=-gamma
        g=(B-gamma)/(B+gamma)
        return 1-g*cmath.exp(-gamma*T)
    grid=np.linspace(-0.25,p_min,int(samples));vals=np.asarray([den(q).imag for q in grid]);roots=[]
    for i in range(len(grid)-1):
        if vals[i]*vals[i+1]>=0:continue
        try:r=brentq(lambda q:den(q).imag,grid[i+1],grid[i],xtol=2e-11)
        except ValueError:continue
        if abs(den(r))>2e-5:continue
        if not roots or abs(r-roots[-1])>1e-4:roots.append(float(r))
        if len(roots)==2:return roots
    raise RuntimeError('first two moment poles not found')


def first_adjacent_fast(params, T):
    poles=moment_poles_fast(params,T)
    a,b=poles[0],poles[1]
    # Scan only the first logarithmic interval; by definition the first sign
    # change after the first pole is the adjacent sector used in the paper.
    margin=max(2e-4,2e-4*abs(a-b));grid=np.linspace(a-margin,b+margin,420)
    f=lambda q:_real_cgf_derivative(q,params,T=T)-X
    prev=None
    for q in grid:
        try:v=f(float(q))
        except Exception:prev=None;continue
        if not np.isfinite(v) or abs(v)>2e4:prev=None;continue
        if prev is not None and prev[1]*v<0:
            r=brentq(f,float(q),prev[0],xtol=2e-10,rtol=2e-10)
            return float(r),poles
        prev=(float(q),float(v))
    raise RuntimeError('no first-adjacent stationary point between the first two moment poles')


def fold_curve(T_values):
    base=HestonParams(rho=-.7,**BASE);values={};f0=find_heston_rho_fold(base,X,T=1.0,guess=(-13.47,.03),dps=50);values[1.0]=f0
    prev=(f0.p_c,f0.rho_c)
    for T in sorted([float(t) for t in T_values if t>1]):
        f=find_heston_rho_fold(base,X,T=T,guess=prev,dps=50);values[T]=f;prev=(f.p_c,f.rho_c)
    prev=(f0.p_c,f0.rho_c)
    for T in sorted([float(t) for t in T_values if t<1],reverse=True):
        f=find_heston_rho_fold(base,X,T=T,guess=prev,dps=50);values[T]=f;prev=(f.p_c,f.rho_c)
    return [values[float(t)] for t in T_values]


def zero_crossing(x,y):
    x=np.asarray(x);y=np.asarray(y);order=np.argsort(x);x=x[order];y=y[order]
    for i in range(len(x)-1):
        if y[i]*y[i+1]<0:return float(x[i]-y[i]*(x[i+1]-x[i])/(y[i+1]-y[i]))
    return np.nan


def main():
    Tvals=np.linspace(.7,3.2,21);rhovals=np.linspace(-.9,.32,43);folds=fold_curve(Tvals)
    foldarr=np.asarray([[T,f.rho_c,f.p_c] for T,f in zip(Tvals,folds)])
    np.savetxt(RESULTS/'heston_T_rho_caustic_curve.csv',foldarr,delimiter=',',header='T,rho_caustic,p_caustic',comments='')
    Re=np.full((len(Tvals),len(rhovals)),np.nan);Theta=np.full_like(Re,np.nan);P1=np.full_like(Re,np.nan);rows=[];anti=[]
    monodromy=2*np.pi*BASE['kappa']*BASE['theta']/BASE['xi']**2
    for i,(T,fold) in enumerate(zip(Tvals,folds)):
        xs=[];ys=[]
        for j,rho in enumerate(rhovals):
            if rho>=fold.rho_c-0.002:continue
            par=HestonParams(rho=float(rho),**BASE)
            try:p1,poles=first_adjacent_fast(par,float(T));p0=physical_saddle_fast(par,float(T))
            except RuntimeError:continue
            phi0=heston_scaled_cgf(p0,par,T=float(T))-p0*X;phi1=heston_scaled_cgf(p1,par,T=float(T))-p1*X;delta=phi0-phi1
            if delta.imag<0:delta=delta.conjugate()
            Re[i,j]=delta.real;Theta[i,j]=math.atan2(delta.imag,delta.real);P1[i,j]=p1
            rows.append([T,rho,p0,p1,poles[0],poles[1],delta.real,delta.imag,Theta[i,j]])
            xs.append(rho);ys.append(delta.real)
        z=zero_crossing(xs,ys)
        if np.isfinite(z):anti.append([T,z])
        print(f'T={T:.3f}: {len(xs)} points, anti={z}')
    np.savetxt(RESULTS/'heston_T_rho_phase_grid.csv',np.asarray(rows),delimiter=',',header='T,rho,p0,p1,pole1,pole2,delta_real,delta_imag,stokes_angle',comments='')
    anti=np.asarray(anti);np.savetxt(RESULTS/'heston_T_rho_antistokes_curve.csv',anti,delimiter=',',header='T,rho_antistokes',comments='')
    TT,RR=np.meshgrid(Tvals,rhovals,indexing='ij')
    plt.figure(figsize=(8,5.6));cf=plt.contourf(TT,RR,Re,levels=np.linspace(-1.2,3.2,25),extend='both');plt.colorbar(cf,label=r'${\rm Re}\,\Delta S_1$')
    plt.plot(foldarr[:,0],foldarr[:,1],'k-',lw=1.8,label=r'caustic $\Phi_p=\Phi_{pp}=0$')
    if anti.size:plt.plot(anti[:,0],anti[:,1],'w--',lw=2,label=r'anti-Stokes ${\rm Re}\,\Delta S_1=0$')
    plt.xlabel(r'$T$');plt.ylabel(r'$\rho$');plt.title('Global first-adjacent Heston resurgent phase portrait');plt.legend(fontsize=8);plt.tight_layout();plt.savefig(FIGURES/'heston_T_rho_resurgent_phase_portrait.png',dpi=240);plt.close()
    plt.figure(figsize=(8,5.6));cf=plt.contourf(TT,RR,Theta,levels=np.linspace(.35,2.9,24),extend='both');plt.colorbar(cf,label=r'$\vartheta_S=\arg\Delta S_1$')
    plt.plot(foldarr[:,0],foldarr[:,1],'k-',lw=1.8,label='caustic')
    if anti.size:plt.plot(anti[:,0],anti[:,1],'w--',lw=2,label='anti-Stokes')
    plt.xlabel(r'$T$');plt.ylabel(r'$\rho$');plt.title(r'Stokes-angle field of the first adjacent sector');plt.legend(fontsize=8);plt.tight_layout();plt.savefig(FIGURES/'heston_T_rho_stokes_angle.png',dpi=240);plt.close()
    # Interpolate the anti-Stokes crossing on the benchmark rho=-0.7 slice.
    Ta=np.nan
    if anti.size:
        for k in range(len(anti)-1):
            y0=anti[k,1]+.7;y1=anti[k+1,1]+.7
            if y0*y1<=0:Ta=anti[k,0]-y0*(anti[k+1,0]-anti[k,0])/(y1-y0);break
    with open(RESULTS/'heston_T_rho_phase_summary.csv','w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['quantity','value']);w.writerow(['monodromy',monodromy]);w.writerow(['anti_stokes_crossing_at_rho_minus_0p7',Ta]);w.writerow(['n_grid_points',len(rows)]);w.writerow(['n_antistokes_points',len(anti)])
    print('benchmark anti-Stokes T interpolation',Ta,'grid',len(rows))

if __name__=='__main__':main()
