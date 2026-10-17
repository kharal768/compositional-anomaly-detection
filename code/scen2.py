import numpy as np, json, time
from coda import *
from scen import make_stats, local_props, shannon
def target_labels(sim, w, target):
    bg=np.asarray(sim.bg); D=sim.D; n=sim.n
    u=sim.rng.random(n)
    p_in=(1-w)*bg+w*np.asarray(target)
    lab_bg=np.searchsorted(np.cumsum(bg),u,side="right").clip(0,D-1)
    lab_in=np.searchsorted(np.cumsum(p_in),u,side="right").clip(0,D-1)
    return np.where(sim.in_pts, lab_in, lab_bg)
def run(sim, ws, target, B_ref=300, B=300, M=100):
    stats=make_stats(sim,B_ref); names=["T2","oracle","maxtype","shannon"]
    nulls={n:[] for n in names}
    for _ in range(B):
        s=stats(sample_labels(sim.rng,sim.in_pts,sim.D,None,sim.bg))
        for n in names: nulls[n].append(s[n])
    crit={n:np.quantile(np.array(nulls[n]),0.95,axis=0) for n in names}
    out={}
    for w in ws:
        pw={n:[] for n in names}; fa={n:[] for n in names}
        for _ in range(M):
            s=stats(target_labels(sim,w,target))
            for n in names:
                f=s[n]>crit[n]; pw[n].append(f[sim.in_grid].mean()); fa[n].append(f[~sim.in_grid].mean())
        out[w]={n:(np.mean(pw[n]),np.mean(fa[n])) for n in names}
    return out
cfg={"C":dict(kw=dict(n=2000,k=30,radius=0.13,bg=[0.25]*4,center=(0.3,0.7)),target=[0.40,0.40,0.10,0.10],ws=[0.4,0.55,0.7,0.85,1.0]),
     "D":dict(kw=dict(n=3000,k=25,radius=0.15,bg=[0.40,0.30,0.20,0.10],center=(0.5,0.5)),target=[0.40,0.10,0.30,0.20],ws=[0.4,0.55,0.7,0.85,1.0])}
summ={}; t=time.time()
for sc,c in cfg.items():
    res=[]
    for seed in range(1,11):
        res.append(run(Sim(seed=200+seed,**c["kw"]),c["ws"],c["target"]))
    summ[sc]={}
    for n in ["T2","oracle","maxtype","shannon"]:
        P=np.array([[r[w][n][0] for w in c["ws"]] for r in res]); F=np.array([[r[w][n][1] for w in c["ws"]] for r in res])
        summ[sc][n]=dict(power=P.mean(0).tolist(),power_sd=P.std(0,ddof=1).tolist(),fa=F.mean(0).tolist(),fa_sd=F.std(0,ddof=1).tolist())
        print(sc,n,"power",np.round(P.mean(0),3),"sd",np.round(P.std(0,ddof=1),3),"fa",np.round(F.mean(0),3),flush=True)
    print(round(time.time()-t))
json.dump(summ,open("scen2.json","w"))
