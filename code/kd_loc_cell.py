import numpy as np, json, sys, time
from coda import Sim, sample_labels, t2
import kd_loc as K   # reuses sim, order, rad, llr_max_rows (module-level setup)
seed=int(sys.argv[1]); stage=sys.argv[2]; t0=time.time()
sim=K.sim; rad=K.rad; G=sim.grid.shape[0]
dgi=np.sqrt(((sim.grid[:,None,:]-sim.pts[None,:,:])**2).sum(-1))   # (G,n)
def cell_scores(lab, types):
    best=np.zeros(G)
    for t in types:
        M=K.llr_max_rows(lab,t)
        S=np.maximum.accumulate(M[:,::-1],axis=1)[:,::-1]          # suffix max over larger windows
        s=np.zeros(G)
        for i in range(sim.n):
            mmin=np.searchsorted(rad[i],dgi[:,i])                  # smallest window of centre i containing each cell
            ok=mmin<rad.shape[1]
            v=np.where(ok, S[i,np.minimum(mmin,rad.shape[1]-1)], 0.0)
            s=np.maximum(s,v)
        best=np.maximum(best,s)
    return best
MODES={"oracle":[0],"typeagnostic":[0,1,2,3]}
if stage=="calib":
    nulls={m:[] for m in MODES}
    for _ in range(199):
        lab=sample_labels(sim.rng,sim.in_pts,sim.D,None,None)
        for m,types in MODES.items(): nulls[m].append(cell_scores(lab,types).max())
    crit={m:float(np.quantile(nulls[m],0.95)) for m in MODES}
    json.dump(crit,open(f"kdcell_crit_{seed}.json","w")); print("crit",crit,round(time.time()-t0),"s")
else:
    e=float(stage); crit=json.load(open(f"kdcell_crit_{seed}.json"))
    out={}
    hs={m:[] for m in MODES}; fs={m:[] for m in MODES}
    for _ in range(10):
        lab=sample_labels(sim.rng,sim.in_pts,sim.D,e,None)
        for m,types in MODES.items():
            fl=cell_scores(lab,types)>=crit[m]
            hs[m].append(fl[sim.in_grid].mean()); fs[m].append(fl[~sim.in_grid].mean())
    out=dict(seed=seed,effect=e,**{m:[float(np.mean(hs[m])),float(np.mean(fs[m]))] for m in MODES})
    open(f"kdc_{seed}_{e}.json","w").write(json.dumps(out)); print(out, round(time.time()-t0),"s")
