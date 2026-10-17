import numpy as np, time, resource
from coda import *
rng = np.random.default_rng(5)
D=4; k=30; G_m=35
grid,_ = make_grid(G_m); V=helmert(D)
res=[]
for n in (2_000, 20_000, 200_000, 1_000_000):
    pts = rng.random((n,2)); lab = rng.integers(0,D,n)
    t0=time.perf_counter(); idx = knn_index(pts, grid, k); t_idx=time.perf_counter()-t0
    B=100
    t0=time.perf_counter()
    for _ in range(B):
        l = rng.permutation(lab)
        Y = ilr_from_counts(counts_from_labels(l, idx, D), k, V)
    t_perm=(time.perf_counter()-t0)/B
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    res.append((n,t_idx,t_perm,rss)); print(n, round(t_idx,3), round(t_perm*1000,2),"ms/perm", round(rss),"MB", flush=True)
# masking formula check
p=3; rng=np.random.default_rng(1); G=2_000_000
for pi,d in ((0.053,20.0),(0.053,60.0),(0.2,50.0)):
    S0=np.eye(p); delta=np.zeros(p); delta[0]=np.sqrt(d)
    Y=rng.standard_normal((G,p)); a=rng.random(G)<pi; Y[a]+=delta
    m=Y.mean(0); S=np.cov(Y.T); Si=np.linalg.inv(S)
    T=np.einsum('ij,jk,ik->i',Y-m,Si,Y-m)
    pred=(1-pi)**2*d/(1+pi*(1-pi)*d)
    print("pi",pi,"d",d,"empirical anomalous mean T2:",round(T[a].mean()-0+0,2),"(includes noise df=",p,") ; deterministic part pred",round(pred,2),"; clean d=",d,"; bound (1-pi)/pi=",round((1-pi)/pi,2))
