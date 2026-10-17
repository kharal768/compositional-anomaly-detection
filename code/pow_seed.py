import numpy as np, json, sys
from coda import *
seed=int(sys.argv[1])
sim=Sim(n=2000,k=30,seed=seed)
raw=[sim.null_Y() for _ in range(300)]; mu=np.vstack(raw).mean(0); Si=np.linalg.inv(np.cov(np.vstack(raw).T))
N=np.array([t2(sim.null_Y(),mu,Si) for _ in range(300)]); crit=np.quantile(N,0.95,axis=0)
res={}
for e in (0.40,0.55,0.70,0.85):
    h=[];f=[]
    for _ in range(100):
        fl=t2(sim.alt_Y(e),mu,Si)>crit; h.append(fl[sim.in_grid].mean()); f.append(fl[~sim.in_grid].mean())
    res[str(e)]=[float(np.mean(h)),float(np.mean(f))]
open(f"pow_{seed}.json","w").write(json.dumps(res)); print(seed,res)
