import numpy as np, json, sys, warnings, logging; warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
from coda import Sim, sample_labels
from cozipy import run_cozi
from scen2 import target_labels
w=float(sys.argv[1]); target=[0.40,0.10,0.30,0.20]
def stat(sim,labels):
    df=run_cozi(sim.pts,labels.astype(str),nbh_def="knn",n_neighbors=25,n_permutations=50,random_state=0,return_df=True)
    z=np.abs(df["zscore"].values); assert np.isfinite(z).all(); return float(z.max())
rej=[]
for seed in range(1,21):
    sim=Sim(n=3000,k=25,radius=0.15,seed=200+seed,bg=[0.40,0.30,0.20,0.10],center=(0.5,0.5))
    nulls=[stat(sim,sample_labels(sim.rng,sim.in_pts,sim.D,None,sim.bg)) for _ in range(40)]
    c=np.quantile(nulls,0.95)
    rej.append(stat(sim,target_labels(sim,w,target))>c)
print(w,"COZI global detection, scenario D, 20 patterns:",np.mean(rej),flush=True)
open(f"cozi_D_{w}.json","w").write(json.dumps(float(np.mean(rej))))
