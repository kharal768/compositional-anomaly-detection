import numpy as np, json, sys
from kulldorff import scan_test
g=int(sys.argv[1]); B=int(sys.argv[2])
d=np.load("data/lansing.npz"); pts=np.column_stack([d['x'],d['y']]); lab=d['m'].astype(int); nm=str(d['levels'][g])
obs,p,w,c95,_=scan_test(pts,(lab==g).astype(int),B=B,rng=np.random.default_rng(g))
ctr=pts[w['centre']]; dist=np.sort(np.sqrt(((pts-ctr)**2).sum(1)))
r=dict(species=nm,LLR=float(obs),p=float(p),p_bonf6=float(min(1,6*p)),cases_in=w['cases'],window_n=w['size'],expected=w['expected'],centre=ctr.tolist(),radius=float(dist[w['size']-1]),B=B)
open(f"kd_lansing_{g}.json","w").write(json.dumps(r)); print(json.dumps(r))
