import numpy as np, time, json
from coda import *

def run_all(sim, effects, B_ref, B, M, B_cov, lam=None, insample=False, buf=None, eps=0.0):
    W = None if lam is None else smooth_matrix(sim.grid, lam, sim.dgrid)
    sm = (lambda Y: W @ Y) if W is not None else (lambda Y: Y)
    if not insample:
        raw = [sim.null_Y() for _ in range(B_ref)]
        mu = np.vstack(raw).mean(0)
        S = np.cov(np.vstack(raw).T) if W is None else np.cov(np.vstack([sm(sim.null_Y()) for _ in range(B_cov)]).T)
        Sinv = np.linalg.inv(S)
        stat = lambda Y: t2(sm(Y), mu, Sinv)
    else:
        def stat(Y):
            Z = sm(Y); return t2(Z, Z.mean(0), np.linalg.inv(np.cov(Z.T)))
    crit = np.quantile(np.array([stat(sim.null_Y()) for _ in range(B)]), 0.95, axis=0)
    out = {}
    far = (~sim.in_grid) if buf is None else ((~sim.in_grid) & (sim.dist_c > buf))
    for e in effects:
        h=np.empty(M); f1=np.empty(M); f2=np.empty(M)
        for m in range(M):
            f = stat(sim.alt_Y(e, eps)) > crit
            h[m]=f[sim.in_grid].mean(); f1[m]=f[~sim.in_grid].mean(); f2[m]=f[far].mean()
        out[e]=(h.mean(), f1.mean(), f2.mean())
    return out

if __name__ == "__main__":
    effects=[0.40,0.55,0.70,0.85]; NP=10
    cfgs = {"insample":dict(insample=True), "clean":dict(), "ewma040":dict(lam=0.40), "ewma015":dict(lam=0.15)}
    res = {c:[] for c in cfgs}
    t=time.time()
    for seed in range(1,NP+1):
        sim = Sim(n=2000,k=30,seed=seed)
        dl = sim.dgrid
        for c,kw in cfgs.items():
            kw=dict(kw)
            if "lam" in kw: kw["buf"]=dl*np.log(0.05)/np.log(1-kw["lam"])
            res[c].append(run_all(sim,effects,B_ref=300,B=300,M=100,B_cov=200,**kw))
        print("seed",seed,round(time.time()-t),flush=True)
    summ={}
    for c in cfgs:
        a=np.array([[ [r[e][j] for e in effects] for j in range(3)] for r in res[c]])  # (NP,3,4)
        summ[c]={"power":a[:,0].mean(0).tolist(),"power_sd":a[:,0].std(0,ddof=1).tolist(),
                 "fa":a[:,1].mean(0).tolist(),"fa_sd":a[:,1].std(0,ddof=1).tolist(),
                 "fa_buf":a[:,2].mean(0).tolist(),"fa_buf_sd":a[:,2].std(0,ddof=1).tolist()}
        print(c); 
        for k_,v in summ[c].items(): print("  ",k_,np.round(v,3))
    json.dump(summ,open("ablation.json","w"))
