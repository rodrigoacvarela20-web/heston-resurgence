from pathlib import Path
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from run_heston_parameter_generalization import AXES, high_order_case, write_csv, make_figures

def read_structure(path):
    import csv
    rows=[]
    with open(path,newline='',encoding='utf-8') as f:
        for r in csv.DictReader(f):
            out={}
            for k,v in r.items():
                if k=='axis': out[k]=v
                else:
                    try: out[k]=int(v) if k in ('selected_branch','intersection_below','intersection_above') else float(v)
                    except: out[k]=v
            rows.append(out)
    return rows

def main():
    n_terms=24; workers=2
    payloads=[(axis,value,n_terms) for axis,values in AXES.items() for value in values]
    rows=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        fs={pool.submit(high_order_case,p):p for p in payloads}
        for f in as_completed(fs):
            axis,value,_=fs[f]; r=f.result(); rows.append(r)
            print(axis,value,'ERR '+r['error'] if r['error'] else f"gap={r['pade_rel_gap']:.4g} Sdist={r['stokes_distance_minus_one']:.4g}",flush=True)
    rows.sort(key=lambda r:(list(AXES).index(r['axis']),r['value']))
    cols=['axis','value','n_terms','pade_real','pade_imag','pade_rel_gap','fit_real','fit_imag','fit_rel_gap','stokes_real','stokes_imag','stokes_distance_minus_one','chain_poles','error']
    write_csv(ROOT/'results'/'heston_parameter_sweep_resurgence.csv',rows,cols)
    structural=read_structure(ROOT/'results'/'heston_parameter_sweep_structure.csv')
    make_figures(structural,rows)
    valid=[r for r in rows if not r['error'] and np.isfinite(r['pade_rel_gap'])]
    print('SUMMARY valid',len(valid),'of',len(rows))
    print('median gap',np.median([r['pade_rel_gap'] for r in valid]),'max gap',max(r['pade_rel_gap'] for r in valid))
    print('median Sdist',np.median([r['stokes_distance_minus_one'] for r in valid]),'max Sdist',max(r['stokes_distance_minus_one'] for r in valid))
    worstg=max(valid,key=lambda r:r['pade_rel_gap']); worsts=max(valid,key=lambda r:r['stokes_distance_minus_one'])
    print('worst gap',worstg)
    print('worst stokes',worsts)
if __name__=='__main__': main()
