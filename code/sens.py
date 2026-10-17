import numpy as np, time, json
from coda import *
from ablation import run_all
effects=[0.55,0.70,0.85]; NP=6
cfgs = [
 ("baseline",        dict(n=2000,k=30,radius=0.13), {}),
 ("n=1000",          dict(n=1000,k=30,radius=0.13), {}),
 ("n=4000",          dict(n=4000,k=30,radius=0.13), {}),
 ("k=15",            dict(n=2000,k=15,radius=0.13), {}),
 ("k=60",            dict(n=2000,k=60,radius=0.13), {}),
 ("radius=0.08",     dict(n=2000,k=30,radius=0.08), {}),
 ("radius=0.20",     dict(n=2000,k=30,radius=0.20), {}),
 ("label noise 10%", dict(n=2000,k=30,radius=0.13), dict(eps=0.10)),
 ("label noise 20%", dict(n=2000,k=30,radius=0.13), dict(eps=0.20)),
 ("skewed background",dict(n=2000,k=30,radius=0.13,bg=[0.55,0.20,0.15,0.10]), {}),
]
out={}
t=time.time()
for name,skw,kw in cfgs:
    P=[];F=[]
    for seed in range(1,NP+1):
        sim=Sim(seed=seed,**skw)
        r=run_all(sim,effects,B_ref=300,B=300,M=100,B_cov=100,**kw)
        P.append([r[e][0] for e in effects]); F.append([r[e][1] for e in effects])
    P=np.array(P);F=np.array(F)
    out[name]=dict(power=P.mean(0).tolist(),power_sd=P.std(0,ddof=1).tolist(),fa=F.mean(0).tolist(),fa_sd=F.std(0,ddof=1).tolist())
    print(name, np.round(P.mean(0),3), np.round(P.std(0,ddof=1),3), "FA", np.round(F.mean(0),3), round(time.time()-t), flush=True)
json.dump(out,open("sens.json","w"))
