import numpy as np, json, sys, warnings, logging, os; warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
from cozipy import run_cozi
from coda import Sim, sample_labels, t2
seed=int(sys.argv[1]); stage=sys.argv[2]
sim=Sim(n=2000,k=30,seed=seed,bg=[0.25]*4,center=(0.3,0.7))
rng=np.random.default_rng(seed+500); G=sim.grid.shape[0]
def window_stat(labels, cells):
    out=np.empty(len(cells))
    for j,g in enumerate(cells):
        w=sim.idx[g]
        df=run_cozi(sim.pts[w],labels[w].astype(str),nbh_def="knn",n_neighbors=6,n_permutations=50,random_state=0,return_df=True)
        z=np.abs(df["zscore"].values); out[j]=z.max() if np.isfinite(z).all() else np.nan
    return out
calib_cells=rng.choice(G,60,replace=False)
if stage=="calib":
    nv=[]
    for _ in range(200):
        nv.append(window_stat(sample_labels(sim.rng,sim.in_pts,sim.D,None,None),calib_cells))
    nv=np.array(nv); assert np.isfinite(nv).all()
    np.save(f"nv_{seed}.npy",nv); print("calib done, c_cozi=",np.quantile(nv.ravel(),0.95))
else:
    e=float(stage)
    nv=np.load(f"nv_{seed}.npy"); c=np.quantile(nv.ravel(),0.95)
    raw=[sim.null_Y() for _ in range(300)]; mu=np.vstack(raw).mean(0); Si=np.linalg.inv(np.cov(np.vstack(raw).T))
    N=np.array([t2(sim.null_Y(),mu,Si) for _ in range(300)]); cT=np.quantile(N,0.95,axis=0)
    hC=[];fC=[];hT=[];fT=[]
    for _ in range(10):
        lab=sample_labels(sim.rng,sim.in_pts,sim.D,e,None)
        fc=window_stat(lab,np.arange(G))>c; hC.append(fc[sim.in_grid].mean()); fC.append(fc[~sim.in_grid].mean())
        ft=t2(sim.Y(lab),mu,Si)>cT; hT.append(ft[sim.in_grid].mean()); fT.append(ft[~sim.in_grid].mean())
    r=dict(seed=seed,effect=e,c_cozi=float(c),COZI=[float(np.mean(hC)),float(np.mean(fC))],T2=[float(np.mean(hT)),float(np.mean(fT))])
    open(f"cl_{seed}_{e}.json","w").write(json.dumps(r)); print(r)
