from pathlib import Path
import csv, math, sys
import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ftfinance.heston import HestonParams
from ftfinance.heston_resurgence import find_heston_tail_saddle, find_heston_first_adjacent_sheet_saddle, heston_adjacent_sheet_fluctuation_coefficients

BASE=dict(mu=0.0,kappa=2.0,theta=0.04,xi=0.45,rho=-0.7,v0=0.04)
XSTAR=-0.25; T=1.0
RESULTS=ROOT/'results'; FIGURES=ROOT/'figures'

def read_csv(path):
    with open(path,newline='',encoding='utf8') as f:
        return list(csv.DictReader(f))

def critical_rho():
    mp.mp.dps=70
    k=mp.mpf('2'); th=mp.mpf('.04'); xi=mp.mpf('.45'); v0=mp.mpf('.04'); mu=mp.mpf('0'); tt=mp.mpf('1'); x=mp.mpf('-.25')
    def L(p,rho):
        B=k-rho*xi*p
        gam=mp.sqrt(B*B-xi*xi*(p*p-p))
        g=(B-gam)/(B+gam); e=mp.e**(-gam*tt)
        D=(B-gam)/(xi*xi)*(1-e)/(1-g*e)
        C=mu*p*tt+(k*th/(xi*xi))*((B-gam)*tt-2*mp.log((1-g*e)/(1-g)))
        return C+v0*D
    def f1(p,r): return mp.diff(lambda q:L(q,r),p)-x
    def f2(p,r): return mp.diff(lambda q:L(q,r),p,2)
    p,r=mp.findroot((f1,f2),(-13.47,mp.mpf('.02995')),tol=mp.mpf('1e-50'),maxsteps=50)
    return float(mp.re(p)),float(mp.re(r))

def build_caustic():
    rhos=list(np.linspace(-0.30,0.0,13))+[0.005,0.010,0.015,0.020,0.025,0.027,0.028,0.029,0.0295,0.0299]
    rows=[]
    for rho in rhos:
        p=HestonParams(**{**BASE,'rho':float(rho)})
        physical=find_heston_tail_saddle(p,XSTAR,T=T,dps=60)
        adjacent=find_heston_first_adjacent_sheet_saddle(p,XSTAR,T=T,dps=85,reference_saddle=physical,p_min=-200)
        local=heston_adjacent_sheet_fluctuation_coefficients(p,XSTAR,T=T,n_terms=2,dps=90,reference_saddle=physical,adjacent=adjacent)
        rows.append({'rho':rho,'p1':adjacent['p'].real,'lambda2':local['lambda_second'].real,'delta_real':adjacent['singulant'].real,'delta_imag':adjacent['singulant'].imag})
    pc,rc=critical_rho()
    with open(RESULTS/'heston_rho_caustic.csv','w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=['rho','p1','lambda2','delta_real','delta_imag']);w.writeheader();w.writerows(rows)
    with open(RESULTS/'heston_rho_caustic_critical.csv','w',newline='',encoding='utf8') as f:
        w=csv.writer(f);w.writerow(['rho_critical','p_critical']);w.writerow([rc,pc])
    plt.figure(figsize=(6.4,4.3))
    plt.plot([r['rho'] for r in rows],[r['lambda2'] for r in rows],'o-')
    plt.axhline(0,linewidth=.8)
    plt.axvline(rc,linestyle='--',linewidth=.9,label=rf'$\rho_c={rc:.5f}$')
    plt.xlabel(r'$\rho$');plt.ylabel(r'$\Phi^{\prime\prime}(p_1)$')
    plt.title('Approach to the adjacent-saddle caustic')
    plt.legend(fontsize=8);plt.tight_layout();plt.savefig(FIGURES/'heston_generalization_rho_caustic.png',dpi=220);plt.close()
    return pc,rc

def make_summary_figures():
    structural=read_csv(RESULTS/'heston_parameter_sweep_structure.csv')
    # exact monodromy check across all 25 points
    obs=np.array([float(r['delta_imag']) for r in structural]); pred=np.array([float(r['monodromy_pred']) for r in structural])
    plt.figure(figsize=(5.2,4.6));plt.plot(pred,obs,'o');lo=min(pred.min(),obs.min());hi=max(pred.max(),obs.max());plt.plot([lo,hi],[lo,hi],'--');plt.xlabel(r'$2\pi\kappa\theta/\xi^2$');plt.ylabel(r'observed $\mathrm{Im}\,\Delta S_1$');plt.title('Monodromy identity across parameter sweeps');plt.tight_layout();plt.savefig(FIGURES/'heston_generalization_monodromy.png',dpi=220);plt.close()
    # topology jump
    labels=[f"{r['axis']}={float(r['value']):g}" for r in structural]
    below=np.array([int(float(r['intersection_below'])) for r in structural]);above=np.array([int(float(r['intersection_above'])) for r in structural])
    x=np.arange(len(labels));plt.figure(figsize=(10.8,4.5));plt.plot(x,below,'o',label='below Stokes wall');plt.plot(x,above,'s',label='above Stokes wall');plt.xticks(x,labels,rotation=70,ha='right',fontsize=6.5);plt.ylabel('oriented intersection number');plt.title('Picard-Lefschetz jump across 25 one-parameter sweep points');plt.legend(fontsize=8);plt.tight_layout();plt.savefig(FIGURES/'heston_generalization_topology.png',dpi=220);plt.close()
    # Borel-location diagnostics at extremes
    selected=read_csv(RESULTS/'heston_generalization_selected40.csv');labs=[f"{r['axis']}={float(r['value']):g}" for r in selected];gaps=np.array([float(r['pade_rel_gap']) for r in selected])*100
    plt.figure(figsize=(8.8,4.4));plt.plot(np.arange(len(labs)),gaps,'o-');plt.xticks(np.arange(len(labs)),labs,rotation=55,ha='right',fontsize=7);plt.ylabel('Padé-to-action gap (%)');plt.title('Secondary Borel singularity at parameter-sweep extremes (40 coefficients)');plt.tight_layout();plt.savefig(FIGURES/'heston_generalization_borel_extremes.png',dpi=220);plt.close()
    # Real action variation for each axis as separate plots
    axis_tex={'xstar':r'$x_\star$','T':r'$T$','rho':r'$\rho$','xi':r'$\xi$','kappa':r'$\kappa$'}
    for axis in axis_tex:
        rr=sorted([r for r in structural if r['axis']==axis],key=lambda r:float(r['value']))
        xv=[float(r['value']) for r in rr];re=[float(r['delta_real']) for r in rr];im=[float(r['delta_imag']) for r in rr]
        plt.figure(figsize=(5.8,4.2));plt.plot(xv,re,'o-',label=r'$\Re\,\Delta S_1$');plt.plot(xv,im,'s-',label=r'$\Im\,\Delta S_1$');plt.xlabel(axis_tex[axis]);plt.ylabel('singulant component');plt.title(f'Adjacent singulant versus {axis}');plt.legend(fontsize=8);plt.tight_layout();plt.savefig(FIGURES/f'heston_generalization_singulant_{axis}.png',dpi=220);plt.close()

if __name__=='__main__':
    pc,rc=build_caustic();make_summary_figures();
    structural=read_csv(RESULTS/'heston_parameter_sweep_structure.csv');selected=read_csv(RESULTS/'heston_generalization_selected40.csv')
    mon=max(abs(float(r['delta_imag'])-float(r['monodromy_pred'])) for r in structural)
    topo=sorted(set(int(float(r['intersection_above']))-int(float(r['intersection_below'])) for r in structural))
    gaps=[float(r['pade_rel_gap']) for r in selected]
    print('critical rho',rc,'pcrit',pc);print('max abs monodromy error',mon);print('topology jumps',topo);print('median selected Pade gap',float(np.median(gaps)),'max',max(gaps))
