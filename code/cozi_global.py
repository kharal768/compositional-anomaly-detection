import numpy as np, json, warnings, logging; warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
from coda import Sim, sample_labels
from cozipy import run_cozi
def tl(sim,w):
    bg=np.full(sim.D,1/sim.D); D=sim.D; u=sim.rng.random(sim.n)
    p_in=np.array([w]+[(1-w)/(D-1)]*(D-1))
    lb=np.searchsorted(np.cumsum(bg),u,side="right").clip(0,D-1)
    li=np.searchsorted(np.cumsum(p_in),u,side="right").clip(0,D-1)
    return np.where(sim.in_pts,li,lb)
def stat(sim,labels):
    df=run_cozi(sim.pts,labels.astype(str),nbh_def="knn",n_neighbors=30,n_permutations=50,random_state=0,return_df=True)
    z=np.abs(df["zscore"].values); assert np.isfinite(z).all(), "non-finite COZI z-score"; return float(z.max())
out={}
for w in (0.40,0.55,0.70,0.85):
    rej=[]
    for seed in range(1,21):
        sim=Sim(n=2000,k=30,seed=seed,bg=[0.25]*4)
        nulls=[stat(sim,sample_labels(sim.rng,sim.in_pts,sim.D,None,sim.bg)) for _ in range(40)]
        c=np.quantile(nulls,0.95)
        rej.append(stat(sim,tl(sim,w))>c)
    out[w]=float(np.mean(rej)); print(w,"COZI global detection over 20 patterns",np.mean(rej),flush=True)
json.dump(out,open("cozi_global.json","w"))
