from pathlib import Path
import sys,csv
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'experiments'))
from run_heston_parameter_generalization import high_order_case
cases=[('baseline',0),('xstar',-.15),('xstar',-.35),('T',.5),('T',2.0),('rho',-.9),('rho',0.0),('xi',.25),('xi',.8),('kappa',.75),('kappa',4.0)]
# baseline is encoded via xstar=-.25
payload=[]
for a,v in cases:
 if a=='baseline': payload.append(('xstar',-.25,32,'baseline'))
 else: payload.append((a,v,32,f'{a}_{v}'))
path=ROOT/'results'/'heston_selected_highorder_32.csv'
cols=['label','axis','value','n_terms','pade_real','pade_imag','pade_rel_gap','fit_real','fit_imag','fit_rel_gap','stokes_real','stokes_imag','stokes_distance_minus_one','chain_poles','error']
with open(path,'w',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
 for a,v,n,label in payload:
  print('RUN',label,flush=True)
  r=high_order_case((a,v,n));r={'label':label,**r};w.writerow(r);f.flush()
  print(' -> gap',r['pade_rel_gap'],'fitgap',r['fit_rel_gap'],'S',r['stokes_real'],r['stokes_imag'],'dist',r['stokes_distance_minus_one'],'err',r['error'],flush=True)
