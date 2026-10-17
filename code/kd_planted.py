import numpy as np, json, sys
from kulldorff import scan_test
from coda import Sim, sample_labels
e=float(sys.argv[1]); res=[]
for seed in range(1,11):
    sim=Sim(n=2000,k=30,seed=seed,bg=[0.25]*4,center=(0.3,0.7))
    lab=sample_labels(sim.rng,sim.in_pts,4,e,None)
    obs,p,w,_,_=scan_test(sim.pts,(lab==0).astype(int),B=99,rng=np.random.default_rng(seed))
    ctr=sim.pts[w['centre']]; inside=np.hypot(*(ctr-np.array([0.3,0.7])))<=0.13
    res.append(dict(p=float(p),centre_in_hotspot=bool(inside)))
det=np.mean([r['p']<=0.05 and r['centre_in_hotspot'] for r in res])
json.dump(res,open(f"kd_planted_{e}.json","w"))
print(e,"scan detection (p<=0.05 and centre in hotspot):",det,"| p-values:",[round(r['p'],2) for r in res])
