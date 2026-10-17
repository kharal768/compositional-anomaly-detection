import numpy as np, json, sys, time
from scipy.spatial import cKDTree
from coda import Sim, sample_labels, t2

class PerLocNEP:
    """Per-location adaptation of COZI's conditional neighbour-preference z-score (per-location adaptation; not the published COZI output).
    For each grid cell g and type pair (a,b): among window points i of type a, the conditional
    neighbour count is #(i has >=1 type-b neighbour) normalised as in COZI; z is taken against a
    per-cell mean/sd from a separate pure-null reference batch. Score = max over pairs of |z|."""
    def __init__(self, sim, kgraph=30):
        self.sim=sim; D=sim.D; self.D=D
        _, nb = cKDTree(sim.pts).query(sim.pts, k=kgraph+1)
        self.nbr = nb[:,1:]                      # point-level neighbour graph
        self.win = sim.idx                       # grid-level windows (G,k)
    def scores(self, labels):
        D=self.D; oh=np.eye(D)[labels]          # (n,D)
        nbc = oh[self.nbr].sum(1)               # (n,D) neighbour-type counts per point
        lw = oh[self.win]                       # (G,k,D) window labels
        nw = nbc[self.win]                      # (G,k,D)
        obs = np.einsum('gka,gkb->gab', lw, nw)           # neighbour counts
        den = np.einsum('gka,gkb->gab', lw, (nw>0).astype(float))  # COZI-style conditional denominator
        return obs/np.maximum(den,1)            # (G,D,D) conditional ratio per cell
    def calibrate(self, rng, B_ref=200, B=300):
        R=[self.scores(sample_labels(rng,self.sim.in_pts,self.D,None,None)) for _ in range(B_ref)]
        R=np.array(R); self.mu=R.mean(0); self.sd=R.std(0,ddof=1)+1e-9
        Z=[]
        for _ in range(B):
            Z.append(self.stat(sample_labels(rng,self.sim.in_pts,self.D,None,None)))
        Z=np.array(Z); self.crit=np.quantile(Z,0.95,axis=0)
    def stat(self, labels):
        z=np.abs((self.scores(labels)-self.mu)/self.sd)
        D=self.D; mask=~np.eye(D,dtype=bool)
        return z[:,mask].max(1)                  # (G,)

def run(seed, effects=(0.40,0.55,0.70,0.85), M=60):
    sim=Sim(n=2000,k=30,seed=seed)
    rng=sim.rng
    # compositional T2 on the identical protocol
    raw=[sim.null_Y() for _ in range(300)]; mu=np.vstack(raw).mean(0); Si=np.linalg.inv(np.cov(np.vstack(raw).T))
    N=np.array([t2(sim.null_Y(),mu,Si) for _ in range(300)]); cT=np.quantile(N,0.95,axis=0)
    nep=PerLocNEP(sim); nep.calibrate(rng)
    out={}
    for e in effects:
        hT=[];fT=[];hN=[];fN=[]
        for _ in range(M):
            lab=sample_labels(rng,sim.in_pts,sim.D,e,None)
            fl=t2(sim.Y(lab),mu,Si)>cT; hT.append(fl[sim.in_grid].mean()); fT.append(fl[~sim.in_grid].mean())
            fn=nep.stat(lab)>nep.crit; hN.append(fn[sim.in_grid].mean()); fN.append(fn[~sim.in_grid].mean())
        out[str(e)]=dict(T2=[float(np.mean(hT)),float(np.mean(fT))],NEP=[float(np.mean(hN)),float(np.mean(fN))])
    return out

if __name__=="__main__":
    seed=int(sys.argv[1]); t=time.time()
    r=run(seed)
    open(f"perloc_{seed}.json","w").write(json.dumps(r))
    print(seed, {e:(round(v['T2'][0],3),round(v['NEP'][0],3),round(v['T2'][1],3),round(v['NEP'][1],3)) for e,v in r.items()}, round(time.time()-t), "s")
