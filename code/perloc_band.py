import numpy as np, json, sys
from coda import Sim, sample_labels, t2
from perloc_nep import PerLocNEP
seed=int(sys.argv[1]); e=0.85; M=40
sim=Sim(n=2000,k=30,seed=seed); rng=sim.rng
raw=[sim.null_Y() for _ in range(300)]; mu=np.vstack(raw).mean(0); Si=np.linalg.inv(np.cov(np.vstack(raw).T))
N=np.array([t2(sim.null_Y(),mu,Si) for _ in range(300)]); cT=np.quantile(N,0.95,axis=0)
nep=PerLocNEP(sim); nep.calibrate(rng)
bands=[(0,0.0),(0.0,0.05),(0.05,0.10),(0.10,0.20),(0.20,9)]  # distance outside hotspot edge
dist=sim.dist_c  # >0 outside, <0 inside
T={b:[] for b in range(len(bands))}; NE={b:[] for b in range(len(bands))}
for _ in range(M):
    lab=sample_labels(rng,sim.in_pts,sim.D,e,None)
    fT=t2(sim.Y(lab),mu,Si)>cT; fN=nep.stat(lab)>nep.crit
    masks=[dist<=0,(dist>0)&(dist<=0.05),(dist>0.05)&(dist<=0.10),(dist>0.10)&(dist<=0.20),dist>0.20]
    for b,m in enumerate(masks):
        T[b].append(fT[m].mean()); NE[b].append(fN[m].mean())
res={"bands":["hotspot (power)","0-0.05","0.05-0.10","0.10-0.20",">0.20"],
     "T2":[float(np.mean(T[b])) for b in range(5)],"NEP":[float(np.mean(NE[b])) for b in range(5)]}
open(f"band_{seed}.json","w").write(json.dumps(res)); print(seed, "T2", np.round(res["T2"],3).tolist(), "NEP", np.round(res["NEP"],3).tolist())
