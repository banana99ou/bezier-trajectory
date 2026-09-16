import sys, json, csv, time, warnings, platform, hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0,'/Users/hyeon-yongjeong/code/bezier-trajectory')
from tools import build_tables as B
from tools.verify import harness_common as H
from tools.verify.sweep import DEGREES,N_SEGS,SCENARIOS
warnings.filterwarnings('ignore',message='c_KOZ .*')
out=Path('artifacts/degree_full_rerun_20260916')
meta={'commit':B._git('rev-parse','HEAD'),'dirty':B._git('status','--porcelain'),'python':platform.python_version(),'platform':platform.platform(),'scenarios':SCENARIOS,'degrees':DEGREES,'segments':N_SEGS,'repeats':15,'cache':False,'max_iter':1000,'tol':1e-8,'n_lin_seg':100,'trust_radius':'harness scenario defaults','started':time.strftime('%Y-%m-%dT%H:%M:%S%z')}
(out/'metadata.json').write_text(json.dumps(meta,indent=2))
rows=[]; started=time.time()
for scenario in SCENARIOS:
 for n in DEGREES:
  sc=H.make_scenario(scenario,N=n)
  for ns in N_SEGS:
   samples=[];metrics=[];errs=[]
   for repeat in range(15):
    try:
     P,info=H.run_rust(sc,n_seg=ns)
     samples.append(float(info['elapsed_time']))
     hull=float(info['final_hull_violation_km']);stop=int(info['scvx_stop_reason'])
     metrics.append({'certified':hull<=B.CERT_TOL and stop in B.PRINCIPLED_STOPS,'dense_probe_ok':bool(info['feasible']),'margin_km':float(info['min_radius'])-sc['r_e'],'ctrl_cost_ms2':float(info['mean_control_accel_ms2']),'objective':float(info['cost']),'iters':int(info['iterations']),'stop':stop,'hull_violation_km':hull})
    except Exception as e: errs.append({'repeat':repeat,'error':repr(e)})
   row={'scenario':scenario,'degree':n,'n_seg':ns,'trust_radius_km':sc['r0'],'completed_repeats':len(samples),'error_repeats':len(errs)}
   if metrics:
    row.update(metrics[-1]);row.update(runtime_min_s=min(samples),runtime_median_s=float(np.median(samples)),runtime_max_s=max(samples),consistent_metrics=all(m==metrics[0] for m in metrics),J_true=H.J_true(P,sc['T']))
    binds,tau=B._koz_binds(P);row.update(koz_binds=binds,tau_star=tau)
    np.save(out/f'{scenario}_N{n}_seg{ns}_control_points.npy',P)
   rows.append(row)
   with (out/'raw.jsonl').open('a') as f:f.write(json.dumps({'summary':row,'timings':samples,'metrics':metrics,'errors':errs})+'\n')
   keys=list(dict.fromkeys(k for r in rows for k in r))
   with (out/'results.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
   print(f"{len(rows)}/90 {scenario} N={n} seg={ns} certified={row.get('certified')} cost={row.get('ctrl_cost_ms2')} min_s={row.get('runtime_min_s')} stop={row.get('stop')} elapsed={time.time()-started:.0f}s",flush=True)
meta['elapsed_s']=time.time()-started;meta['finished']=time.strftime('%Y-%m-%dT%H:%M:%S%z');(out/'metadata.json').write_text(json.dumps(meta,indent=2))
print('DONE',out,flush=True)
