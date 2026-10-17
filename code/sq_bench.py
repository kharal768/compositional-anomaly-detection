import numpy as np, json, sys, time, warnings
warnings.filterwarnings("ignore")
import anndata as ad, pandas as pd, squidpy as sq
from coda import *

def squidpy_stat(sim, labels, k):
    """Squidpy neighbourhood enrichment (permutation z-scores on the kNN graph)."""
    obs = pd.DataFrame({"type": pd.Categorical([f"t{v}" for v in labels])})
    a = ad.AnnData(X=np.zeros((sim.n, 1)), obs=obs)
    a.obsm["spatial"] = sim.pts
    sq.gr.spatial_neighbors(a, spatial_key="spatial", coord_type="generic", n_neighs=k)
    sq.gr.nhood_enrichment(a, cluster_key="type", n_perms=200, seed=0, show_progress_bar=False)
    return np.asarray(a.uns["type_nhood_enrichment"]["zscore"]), a.obs["type"].cat.categories

if __name__ == "__main__":
    t=time.time(); seed=int(sys.argv[1]); k=30
    sim = Sim(n=2000, k=k, seed=seed)
    rng = np.random.default_rng(seed)
    lab_null = sample_labels(sim.rng, sim.in_pts, sim.D, None, sim.bg)
    z0,_ = squidpy_stat(sim, lab_null, k)
    print("seed",seed,"null zscore range",np.round(z0.min(),2),np.round(z0.max(),2),"secs",round(time.time()-t,1),flush=True)
