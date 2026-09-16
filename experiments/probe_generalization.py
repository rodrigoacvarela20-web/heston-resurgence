from pathlib import Path
import sys, cmath, math
import numpy as np
from scipy.optimize import brentq
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from ftfinance.heston import HestonParams
from ftfinance.heston_resurgence import find_heston_tail_saddle, find_heston_adjacent_sheet_saddle, heston_scaled_cgf

def denominator(p, params, T):
    p=complex(p); k=float(params.kappa); xi=float(params.xi); rho=float(params.rho)
    B=k-rho*xi*p
    gamma=cmath.sqrt(B*B-xi*xi*(p*p-p))
    if gamma.real<0 or (abs(gamma.real)<1e-15 and gamma.imag<0): gamma=-gamma
    g=(B-gamma)/(B+gamma)
    return 1-g*cmath.exp(-gamma*T)

def find_pole(params,T):
    xs=-np.geomspace(0.5,80,3000)  # -0.5 to -80 decreasing
    vals=[denominator(x,params,T) for x in xs]
    roots=[]
    for i in range(len(xs)-1):
        a,b=xs[i],xs[i+1]
        fa,fb=vals[i].imag,vals[i+1].imag
        if fa==0 or fa*fb<0:
            try:
                r=brentq(lambda q: denominator(q,params,T).imag,b,a,xtol=1e-13,rtol=1e-13)
                res=abs(denominator(r,params,T))
                if res<1e-6: roots.append((abs(r),r,res))
            except Exception: pass
    if not roots: raise RuntimeError('no pole')
    return sorted(roots)[0][1]

def adjacent(params,x,T,pole,physical):
    # try continuation-like guesses proportional to pole
    guess_sets=[(pole*1.12,pole*1.35),(pole-0.5,pole-2.0),(pole*1.05,pole*1.8),(-7.5,-8.5)]
    err=[]
    for gs in guess_sets:
        try:
            out=find_heston_adjacent_sheet_saddle(params,x,T=T,dps=70,guesses=gs,reference_saddle=physical)
            p=out['p'].real
            if p < pole-1e-5:
                return out
        except Exception as e: err.append(str(e))
    raise RuntimeError('adjacent failed '+repr(err[-2:]))

base=dict(mu=0,kappa=2,theta=.04,xi=.45,rho=-.7,v0=.04)
axes={
'xstar':[-.15,-.20,-.25,-.30,-.35],
'T':[.5,.75,1,1.5,2],
'rho':[-.9,-.7,-.5,-.3,0.0],
'xi':[.25,.35,.45,.6,.8],
'kappa':[.75,1.25,2,3,4],
}
for axis,vals in axes.items():
    print('\nAXIS',axis)
    for val in vals:
        d=base.copy(); x=-.25; T=1
        if axis=='xstar': x=val
        elif axis=='T': T=val
        else: d[axis]=val
        params=HestonParams(**d)
        try:
            phys=find_heston_tail_saddle(params,x,T=T,dps=60)
            pole=find_pole(params,T)
            adj=adjacent(params,x,T,pole,phys)
            pred=2*math.pi*params.kappa*params.theta/(params.xi**2)
            print(f'{val:>6}: p*={phys.p_star: .5f} I={phys.rate:.5f} pole={pole:.5f} p1={adj["p"].real:.5f} dS={adj["singulant"].real:.4f}+{adj["singulant"].imag:.4f}i mon={pred:.4f}')
        except Exception as e:
            print(val,'FAIL',type(e).__name__,e)
