import numpy as np, time
from coda import *
def run(name, drop=None, grid_m=30, ks=(10,20), disc=None, rect=None, B_ref=500, B=999, seed=0):
    d=np.load("data/"+name+".npz"); pts=np.column_stack([d['x'],d['y']]); lab=d['m'].astype(int); lev=list(d['levels'])
    if drop is not None:
        keep = lab!=drop; pts=pts[keep]; lab=lab[keep]; lev=[l for i,l in enumerate(lev) if i!=drop]
        lab = np.array([sorted(set(lab)).index(v) for v in lab])
    D=len(lev); V=helmert(D)
    if disc: 
        g=np.linspace(-disc*0.98,disc*0.98,grid_m); gx,gy=np.meshgrid(g,g); grid=np.column_stack([gx.ravel(),gy.ravel()])
        grid=grid[(grid**2).sum(1)<(0.98*disc)**2]
    else:
        lo,hi=rect; g=np.linspace(lo+0.02*(hi-lo),hi-0.02*(hi-lo),grid_m); gx,gy=np.meshgrid(g,g); grid=np.column_stack([gx.ravel(),gy.ravel()])
    out=[]
    for k in ks:
        for mode in ("D","k"):
            rng=np.random.default_rng(seed); idx=knn_index(pts,grid,k)
            f=lambda l: ilr_from_counts(counts_from_labels(l,idx,D),k,V,0.65,mode)
            z=(counts_from_labels(lab,idx,D)==0).any(1).mean()
            ref=np.vstack([f(rng.permutation(lab)) for _ in range(B_ref)]); mu=ref.mean(0); Si=np.linalg.inv(np.cov(ref.T))
            null=np.array([t2(f(rng.permutation(lab)),mu,Si) for _ in range(B)])
            obs=t2(f(lab),mu,Si); mx=null.max(1); crit=np.quantile(mx,.95); pc=np.quantile(null,.95,axis=0)
            p=(1+(mx>=obs.max()).sum())/(B+1)
            # naive in-sample for comparison
            def insample(l):
                Y=f(l); return t2(Y,Y.mean(0),np.linalg.inv(np.cov(Y.T)))
            nul2=np.array([insample(rng.permutation(lab)).max() for _ in range(B)]); o2=insample(lab).max()
            p2=(1+(nul2>=o2).sum())/(B+1)
            out.append(dict(data=name,n=len(lab),D=D,k=k,mode=mode,cells=len(grid),zfrac=round(z,3),max_obs=round(obs.max(),1),crit=round(crit,1),p=round(p,3),flagged=int((obs>pc).sum()),share=round((obs>pc).mean(),3),insample_max=round(o2,1),insample_crit=round(np.quantile(nul2,.95),1),insample_p=round(p2,3)))
            print(out[-1],flush=True)
    return out
res=[]
res+=run("hyytiala",drop=0,grid_m=25,ks=(10,20),rect=(0,20))
res+=run("sporophores",grid_m=30,ks=(15,30),disc=145)
import json; json.dump(res,open("realextra.json","w"))
