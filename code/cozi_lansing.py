import numpy as np, json, sys, warnings, logging; warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
from cozipy import run_cozi
d=np.load("data/lansing.npz"); pts=np.column_stack([d['x'],d['y']]); lab=d['m'].astype(int); names=[str(x) for x in d['levels']]
k=int(sys.argv[1]); B=int(sys.argv[2])
def stat(labels, perms):
    df=run_cozi(pts,np.array(names)[labels],nbh_def="knn",n_neighbors=k,n_permutations=perms,random_state=0,return_df=True)
    z=np.abs(df["zscore"].values); assert np.isfinite(z).all(); return float(z.max()), df
obs,df=stat(lab,50)
rng=np.random.default_rng(0)
null=np.array([stat(rng.permutation(lab),50)[0] for _ in range(B)])
p=(1+(null>=obs).sum())/(B+1)
top=df.loc[df["zscore"].abs().idxmax()]
r=dict(k=k,max_abs_z=obs,p_global=float(p),B=B,top_pair=[top.index_cell_type,top.neighbor_cell_type,float(top.zscore)],n_pairs_abs_z_gt_2=int((df["zscore"].abs()>2).sum()),n_pairs=int(len(df)))
open(f"cozi_lansing_k{k}.json","w").write(json.dumps(r)); print(json.dumps(r))
