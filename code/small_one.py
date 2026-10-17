import numpy as np, json, sys
from coda import *
name, k, seed, mode = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
d=np.load("data/"+name+".npz"); pts=np.column_stack([d['x'],d['y']]); lab=d['m'].astype(int); lev=list(d['levels'])
if name=="hyytiala":
    keep=lab!=0; pts=pts[keep]; lab=lab[keep]; lab=np.array([sorted(set(lab)).index(v) for v in lab]); lev=lev[1:]
D=len(set(lab)); V=helmert(D)
if name=="hyytiala":
    g=np.linspace(0.02*20,20-0.02*20,25); gx,gy=np.meshgrid(g,g); grid=np.column_stack([gx.ravel(),gy.ravel()])
else:
    g=np.linspace(-145*0.98,145*0.98,30); gx,gy=np.meshgrid(g,g); grid=np.column_stack([gx.ravel(),gy.ravel()]); grid=grid[(grid**2).sum(1)<(0.98*145)**2]
idx=knn_index(pts,grid,k)
f=lambda l: ilr_from_counts(counts_from_labels(l,idx,D),k,V,0.65,mode)
rng=np.random.default_rng(seed)
ref=np.vstack([f(rng.permutation(lab)) for _ in range(200)]); mu=ref.mean(0); Si=np.linalg.inv(np.cov(ref.T))
null=np.array([t2(f(rng.permutation(lab)),mu,Si).max() for _ in range(499)])
obs=t2(f(lab),mu,Si).max()
inn=[]
for _ in range(299):
    Y=f(rng.permutation(lab)); inn.append(t2(Y,Y.mean(0),np.linalg.inv(np.cov(Y.T))).max())
inn=np.array(inn); Yo=f(lab); o2=t2(Yo,Yo.mean(0),np.linalg.inv(np.cov(Yo.T))).max()
r=dict(data=name,k=k,mode=mode,seed=seed,p=float((1+(null>=obs).sum())/500),p_in=float((1+(inn>=o2).sum())/300))
print(json.dumps(r))
open(f"small_{name}_{k}_{mode}_{seed}.json","w").write(json.dumps(r))
