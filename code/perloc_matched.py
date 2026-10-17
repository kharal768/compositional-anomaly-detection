import numpy as np, json, sys
from coda import Sim, sample_labels, t2
from perloc_nep import PerLocNEP
seed=int(sys.argv[1]); M=50; out={}
sim=Sim(n=2000,k=30,seed=seed); rng=sim.rng
raw=[sim.null_Y() for _ in range(300)]; mu=np.vstack(raw).mean(0); Si=np.linalg.inv(np.cov(np.vstack(raw).T))
N=np.array([t2(sim.null_Y(),mu,Si) for _ in range(300)]); cT=np.quantile(N,0.95,axis=0)
nep=PerLocNEP(sim); nep.calibrate(rng)
for e in (0.55,0.70,0.85):
    hT=[];hN=[];hM=[];areaT=[];areaN=[]
    for _ in range(M):
        lab=sample_labels(rng,sim.in_pts,sim.D,e,None)
        sT=t2(sim.Y(lab),mu,Si); fT=sT>cT
        sN=nep.stat(lab); fN=sN>nep.crit
        K=int(fT.sum())
        hT.append(fT[sim.in_grid].mean())
        hN.append(fN[sim.in_grid].mean()); areaN.append(fN.sum()); areaT.append(K)
        top=np.argsort(-sN)[:K]; fM=np.zeros_like(fN); fM[top]=True   # same flagged count as T2
        hM.append(fM[sim.in_grid].mean())
    out[str(e)]=dict(T2_power=float(np.mean(hT)),NEP_power=float(np.mean(hN)),NEP_power_matched_area=float(np.mean(hM)),
                     area_T2=float(np.mean(areaT)),area_NEP=float(np.mean(areaN)))
open(f"matched_{seed}.json","w").write(json.dumps(out)); print(seed,{e:{k:round(v,3) for k,v in d.items()} for e,d in out.items()})
