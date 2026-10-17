import numpy as np, json, time
from coda import *

# ---------- (a) multiple anomalies: single-step vs step-down ----------
def two_hotspot_labels(rng, pts, D, c1, r1, e1, c2, r2, e2):
    n=len(pts); u=rng.random(n); lab=np.searchsorted(np.cumsum(np.full(D,1/D)),u,side="right").clip(0,D-1)
    for c,r,e in ((c1,r1,e1),(c2,r2,e2)):
        m=((pts-np.array(c))**2).sum(1)<r*r; u2=rng.random(m.sum()); rest=(1-e)/(D-1)
        lab[m]=np.where(u2<e,0,1+np.floor((u2-e)/rest)).clip(0,D-1).astype(int)
    return lab
def stepdown(obs, N, alpha=0.05):
    G=len(obs); order=np.argsort(-obs); o=obs[order]
    Ns=N[:,order]; M=np.maximum.accumulate(Ns[:,::-1],axis=1)[:,::-1]
    p=(1+(M>=o[None,:]).sum(0))/(N.shape[0]+1); p=np.maximum.accumulate(p)
    rej=np.zeros(G,bool); rej[order]=p<=alpha; return rej
def exp_a(seed, B_ref=300, B=500, M=200):
    sim=Sim(n=2000,k=30,seed=seed); D=4
    c1,r1,e1=(0.3,0.7),0.10,0.85; c2,r2,e2=(0.7,0.3),0.10,0.60
    g1=((sim.grid-np.array(c1))**2).sum(1)<r1*r1; g2=((sim.grid-np.array(c2))**2).sum(1)<r2*r2
    raw=[sim.null_Y() for _ in range(B_ref)]; mu=np.vstack(raw).mean(0); Si=np.linalg.inv(np.cov(np.vstack(raw).T))
    N=np.array([t2(sim.null_Y(),mu,Si) for _ in range(B)]); c=np.quantile(N.max(1),0.95)
    out=dict(ss_strong=[],ss_weak=[],ss_weak_any=[],sd_strong=[],sd_weak=[],sd_weak_any=[],fw_ss=[],fw_sd=[])
    for _ in range(M):
        Y=sim.Y(two_hotspot_labels(sim.rng,sim.pts,D,c1,r1,e1,c2,r2,e2)); o=t2(Y,mu,Si)
        a=o>c; b=stepdown(o,N)
        out["ss_strong"].append(a[g1].mean()); out["ss_weak"].append(a[g2].mean()); out["ss_weak_any"].append(a[g2].any())
        out["sd_strong"].append(b[g1].mean()); out["sd_weak"].append(b[g2].mean()); out["sd_weak_any"].append(b[g2].any())
    for _ in range(M*2):
        o=t2(sim.null_Y(),mu,Si); out["fw_ss"].append((o>c).any()); out["fw_sd"].append(stepdown(o,N).any())
    return {k:float(np.mean(v)) for k,v in out.items()}

# ---------- (b) robustness with permutation null ----------
def perm_pipeline(pts, lab, D, k, rng, B_ref=150, B=200, gridm=25):
    grid,_=make_grid(gridm); idx=knn_index(pts,grid,k); V=helmert(D)
    f=lambda l: ilr_from_counts(counts_from_labels(l,idx,D),k,V)
    ref=np.vstack([f(rng.permutation(lab)) for _ in range(B_ref)]); mu=ref.mean(0); Si=np.linalg.inv(np.cov(ref.T))
    N=np.array([t2(f(rng.permutation(lab)),mu,Si) for _ in range(B)])
    obs=t2(f(lab),mu,Si); crit=np.quantile(N.max(1),0.95); pc=np.quantile(N,0.95,axis=0)
    return grid,(obs.max()>crit),(obs>pc)
def inhom_points(rng,n,g):
    out=[]
    while len(out)<n:
        x=rng.random(); y=rng.random()
        if rng.random()<(1+(g-1)*x)/g: out.append((x,y))
    return np.array(out)
def exp_b(M=60,n=2000,D=4,k=30):
    rng=np.random.default_rng(77); res={}
    for g in (1,4,10):
        rej=[];sh=[]
        for _ in range(M):
            pts=inhom_points(rng,n,g); lab=rng.integers(0,D,n)
            _,r,fl=perm_pipeline(pts,lab,D,k,rng); rej.append(r); sh.append(fl.mean())
        res[f"intensity ratio {g}:1, exchangeable marks"]=(float(np.mean(rej)),float(np.mean(sh)))
    rej=[];sh=[]
    for _ in range(M):
        pts=rng.random((n,2)); lab=rng.integers(0,D,n); flip=(lab==0)&(rng.random(n)<0.2); lab[flip]=1
        _,r,fl=perm_pipeline(pts,lab,D,k,rng); rej.append(r); sh.append(fl.mean())
    res["uniform type-dependent confusion (type 1 to 2 with prob 0.2)"]=(float(np.mean(rej)),float(np.mean(sh)))
    rej=[];sh=[];ins=[]
    for _ in range(M):
        pts=rng.random((n,2)); lab=rng.integers(0,D,n); patch=((pts-0.5)**2).sum(1)<0.15**2
        flip=patch&(lab==0)&(rng.random(n)<0.5); lab[flip]=1
        grid,r,fl=perm_pipeline(pts,lab,D,k,rng); rej.append(r); sh.append(fl.mean()); ins.append(fl[((grid-0.5)**2).sum(1)<0.15**2].mean())
    res["spatially clustered confusion patch (r=0.15, prob 0.5)"]=(float(np.mean(rej)),float(np.mean(sh)),float(np.mean(ins)))
    return res

# ---------- (c) power-based choice of k ----------
def exp_c(NP=4,ks=(10,20,30,45,60,90),rs=(0.08,0.13,0.20),eff=0.70):
    out={}
    for r in rs:
        for k in ks:
            P=[];F=[]
            for seed in range(1,NP+1):
                sim=Sim(n=2000,k=k,radius=r,seed=seed)
                raw=[sim.null_Y() for _ in range(250)]; mu=np.vstack(raw).mean(0); Si=np.linalg.inv(np.cov(np.vstack(raw).T))
                N=np.array([t2(sim.null_Y(),mu,Si) for _ in range(250)]); crit=np.quantile(N,0.95,axis=0)
                h=[];f=[]
                for _ in range(60):
                    fl=t2(sim.alt_Y(eff),mu,Si)>crit; h.append(fl[sim.in_grid].mean()); f.append(fl[~sim.in_grid].mean())
                P.append(np.mean(h)); F.append(np.mean(f))
            out[f"{r}|{k}"]=(float(np.mean(P)),float(np.mean(F)))
        print("r",r,{k:(round(out[f'{r}|{k}'][0],3),round(out[f'{r}|{k}'][1],3)) for k in ks},flush=True)
    return out
if __name__=="__main__":
    t=time.time()
    A=[exp_a(s) for s in range(1,7)]
    avg={k:float(np.mean([a[k] for a in A])) for k in A[0]}; sd={k:float(np.std([a[k] for a in A],ddof=1)) for k in A[0]}
    print("A",{k:round(v,3) for k,v in avg.items()}); print("Asd",{k:round(v,3) for k,v in sd.items()}, round(time.time()-t),flush=True)
    B=exp_b(); print("B",B, round(time.time()-t),flush=True)
    C=exp_c(); print(round(time.time()-t))
    json.dump(dict(A=avg,Asd=sd,B=B,C=C),open("extras.json","w"))
