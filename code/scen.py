import numpy as np, json, time
from coda import *

def local_props(sim, labels):
    return counts_from_labels(labels, sim.idx, sim.D) / sim.k

def shannon(p):
    q = np.where(p == 0, 1e-9, p)
    return -(q*np.log(q)).sum(1)

def make_stats(sim, B_ref):
    """reference moments for the four statistics from independent null draws"""
    Ys=[]; Ps=[]; Hs=[]
    for _ in range(B_ref):
        lab = sample_labels(sim.rng, sim.in_pts, sim.D, None, sim.bg)
        Ys.append(sim.Y(lab)); p=local_props(sim,lab); Ps.append(p); Hs.append(shannon(p))
    Yr=np.vstack(Ys); mu=Yr.mean(0); Si=np.linalg.inv(np.cov(Yr.T))
    Pr=np.vstack(Ps); pm=Pr.mean(0); ps=Pr.std(0,ddof=1)
    hm=np.concatenate(Hs).mean()
    def stats(lab):
        Y=sim.Y(lab); p=local_props(sim,lab)
        return {"T2":t2(Y,mu,Si),
                "oracle":np.abs(p[:,0]-pm[0])/ps[0],
                "maxtype":np.abs((p-pm)/ps).max(1),
                "shannon":np.abs(shannon(p)-hm)}
    return stats

def run_scenario(sim, effects, evenness=False, B_ref=300, B=300, M=100):
    stats = make_stats(sim, B_ref)
    names=["T2","oracle","maxtype","shannon"]
    nulls={n:[] for n in names}
    for _ in range(B):
        s=stats(sample_labels(sim.rng, sim.in_pts, sim.D, None, sim.bg))
        for n in names: nulls[n].append(s[n])
    crit={n:np.quantile(np.array(nulls[n]),0.95,axis=0) for n in names}
    out={}
    for e in effects:
        pw={n:[] for n in names}; fa={n:[] for n in names}
        for _ in range(M):
            if evenness:
                lab = even_labels(sim, e)
            else:
                lab = sample_labels(sim.rng, sim.in_pts, sim.D, e, sim.bg)
            s=stats(lab)
            for n in names:
                f=s[n]>crit[n]; pw[n].append(f[sim.in_grid].mean()); fa[n].append(f[~sim.in_grid].mean())
        out[e]={n:(np.mean(pw[n]),np.mean(fa[n])) for n in names}
    return out

def even_labels(sim, w):
    bg=np.asarray(sim.bg); D=sim.D; n=sim.n
    lab=np.empty(n,int)
    u=sim.rng.random(n)
    p_in=(1-w)*bg+w*np.full(D,1/D)
    cb=np.cumsum(bg); ci=np.cumsum(p_in)
    lab_bg=np.searchsorted(cb,u,side="right").clip(0,D-1)
    lab_in=np.searchsorted(ci,u,side="right").clip(0,D-1)
    return np.where(sim.in_pts, lab_in, lab_bg)

if __name__=="__main__":
    NP=10; t=time.time(); res={"A":[], "B":[]}
    effA=[0.40,0.55,0.70,0.85]; effB=[0.40,0.55,0.70,0.85,1.00]
    for seed in range(1,NP+1):
        sim=Sim(n=2000,k=30,seed=seed)
        res["A"].append(run_scenario(sim,effA))
        simB=Sim(n=3000,k=25,seed=100+seed,center=(0.5,0.5),radius=0.15,bg=[0.55,0.20,0.15,0.10])
        res["B"].append(run_scenario(simB,effB,evenness=True))
        print("seed",seed,round(time.time()-t),flush=True)
    summ={}
    for sc,effs in (("A",effA),("B",effB)):
        summ[sc]={}
        for n in ["T2","oracle","maxtype","shannon"]:
            P=np.array([[r[e][n][0] for e in effs] for r in res[sc]]); F=np.array([[r[e][n][1] for e in effs] for r in res[sc]])
            summ[sc][n]=dict(power=P.mean(0).tolist(),power_sd=P.std(0,ddof=1).tolist(),fa=F.mean(0).tolist(),fa_sd=F.std(0,ddof=1).tolist())
            print(sc,n,"power",np.round(P.mean(0),3),"sd",np.round(P.std(0,ddof=1),3),"fa",np.round(F.mean(0),3))
    json.dump(summ,open("scen.json","w"))
