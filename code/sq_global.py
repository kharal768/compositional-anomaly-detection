import numpy as np, json, warnings, logging; warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
from coda import *
from sq_bench import squidpy_stat
def tl(sim,w):
    bg=np.full(sim.D,1/sim.D); D=sim.D; u=sim.rng.random(sim.n)
    p_in=np.array([w]+[(1-w)/(D-1)]*(D-1))
    lb=np.searchsorted(np.cumsum(bg),u,side="right").clip(0,D-1)
    li=np.searchsorted(np.cumsum(p_in),u,side="right").clip(0,D-1)
    return np.where(sim.in_pts,li,lb)
out={}
for w in (0.40,0.55,0.70,0.85):
    rej=[]
    for seed in range(1,21):
        sim=Sim(n=2000,k=30,seed=seed,bg=[0.25]*4)
        # calibrate: max |z| over pairs under the null, 99 null draws, then test one alternative draw
        nulls=[np.abs(squidpy_stat(sim,sample_labels(sim.rng,sim.in_pts,sim.D,None,sim.bg),30)[0]).max() for _ in range(40)]
        c=np.quantile(nulls,0.95)
        z=np.abs(squidpy_stat(sim,tl(sim,w),30)[0]).max()
        rej.append(z>c)
    out[w]=float(np.mean(rej)); print(w, "detection rate over 20 patterns", np.mean(rej),flush=True)
json.dump(out,open("sq_global.json","w"))
