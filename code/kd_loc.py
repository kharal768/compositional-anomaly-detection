import numpy as np, json, sys, time
from scipy.spatial import cKDTree
from coda import Sim, sample_labels, t2
seed=int(sys.argv[1]); stage=sys.argv[2]; t0=time.time()
sim=Sim(n=2000,k=30,seed=seed,bg=[0.25]*4,center=(0.3,0.7))
n=sim.n; D=sim.D; mmax=n//2
D2=((sim.pts[:,None,:]-sim.pts[None,:,:])**2).sum(-1)
order=np.argsort(D2,axis=1,kind="stable")[:,:mmax]
rad=np.sqrt(np.take_along_axis(D2,order,axis=1))          # radius of window (centre i, size m)
m_arr=np.arange(1,mmax+1)[None,:]
def llr_max_rows(lab, t):
    cases=(lab==t).astype(float); C=cases.sum()
    c=np.cumsum(cases[order],axis=1); E=C*m_arr/n
    with np.errstate(divide="ignore",invalid="ignore"):
        t1=np.where(c>0,c*np.log(c/E),0.0); cc=C-c; ee=C-E
        t2_=np.where(cc>0,cc*np.log(cc/ee),0.0)
    return np.where(c>E,t1+t2_,0.0)               # (n,mmax)
def stat(lab, types):
    mats={t:llr_max_rows(lab,t) for t in types}
    return mats
def coverage(mats, crit):
    cov=np.zeros(sim.grid.shape[0],bool)
    for t,M in mats.items():
        sig=M>=crit
        Ri=np.where(sig.any(1), np.where(sig,rad,0).max(1), -1.0)
        ok=Ri>=0
        if not ok.any(): continue
        dd=np.sqrt(((sim.grid[:,None,:]-sim.pts[ok][None,:,:])**2).sum(-1))
        cov |= (dd<=Ri[ok][None,:]).any(1)
    return cov
MODES={"oracle":[0],"typeagnostic":[0,1,2,3]}
if __name__=="__main__":
    MODES={"oracle":[0],"typeagnostic":[0,1,2,3]}
    if stage=="calib":
        nulls={m:[] for m in MODES}
        for _ in range(199):
            lab=sample_labels(sim.rng,sim.in_pts,D,None,None)
            for m,types in MODES.items():
                nulls[m].append(max(llr_max_rows(lab,t).max() for t in types))
        crit={m:float(np.quantile(nulls[m],0.95)) for m in MODES}
        json.dump(crit,open(f"kdloc_crit_{seed}.json","w")); print("crit",crit, round(time.time()-t0),"s")
    else:
        e=float(stage); crit=json.load(open(f"kdloc_crit_{seed}.json"))
        raw=[sim.null_Y() for _ in range(300)]; mu=np.vstack(raw).mean(0); Si=np.linalg.inv(np.cov(np.vstack(raw).T))
        cT=np.quantile(np.array([t2(sim.null_Y(),mu,Si) for _ in range(300)]),0.95,axis=0)
        R={m:dict(h=[],f=[]) for m in list(MODES)+["T2"]}
        for _ in range(10):
            lab=sample_labels(sim.rng,sim.in_pts,D,e,None)
            for m,types in MODES.items():
                fl=coverage(stat(lab,types),crit[m])
                R[m]["h"].append(fl[sim.in_grid].mean()); R[m]["f"].append(fl[~sim.in_grid].mean())
            ft=t2(sim.Y(lab),mu,Si)>cT; R["T2"]["h"].append(ft[sim.in_grid].mean()); R["T2"]["f"].append(ft[~sim.in_grid].mean())
        out=dict(seed=seed,effect=e,**{m:[float(np.mean(v["h"])),float(np.mean(v["f"]))] for m,v in R.items()})
        open(f"kdl_{seed}_{e}.json","w").write(json.dumps(out)); print(out, round(time.time()-t0),"s")
