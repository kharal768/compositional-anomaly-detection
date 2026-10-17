import numpy as np, time
from coda import *
d = np.load("data/lansing.npz")
pts = np.column_stack([d["x"], d["y"]]); lab = d["m"].astype(int); D = 6
grid,_ = make_grid(40)
V = helmert(D)
def feats(labels, idx, k, mode):
    c = counts_from_labels(labels, idx, D)
    if mode == "pseudo":
        x = (c + 0.5)/(k + 0.5*D); lx = np.log(x); return (lx - lx.mean(1,keepdims=True)) @ V
    return ilr_from_counts(c, k, V, 0.65, mode)
def analyse(k, mode, B_ref=500, B=999, seed=0):
    rng = np.random.default_rng(seed)
    idx = knn_index(pts, grid, k)
    zfrac = (counts_from_labels(lab, idx, D)==0).any(1).mean()
    ref = np.vstack([feats(rng.permutation(lab), idx, k, mode) for _ in range(B_ref)])
    mu = ref.mean(0); Sinv = np.linalg.inv(np.cov(ref.T))
    null = np.array([t2(feats(rng.permutation(lab), idx, k, mode), mu, Sinv) for _ in range(B)])
    obs = t2(feats(lab, idx, k, mode), mu, Sinv)
    mx = null.max(1); crit = np.quantile(mx, .95)
    pcrit = np.quantile(null, .95, axis=0)
    p = (1+(mx >= obs.max()).sum())/(B+1)
    return dict(k=k, mode=mode, zfrac=zfrac, max_obs=obs.max(), crit=crit, p=p, flagged=int((obs>pcrit).sum()))
if __name__=="__main__":
    t=time.time()
    for k in (15,40):
        for mode in ("D","k","pseudo"):
            r = analyse(k, mode)
            print({a:(round(b,4) if isinstance(b,float) else b) for a,b in r.items()}, round(time.time()-t), flush=True)
