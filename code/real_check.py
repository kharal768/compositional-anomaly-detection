import os
import numpy as np, pandas as pd, json, warnings
warnings.filterwarnings("ignore")
from scipy.spatial import cKDTree
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from coda import Sim, sample_labels, ilr_from_counts, helmert, zero_replace

d = pd.read_csv(os.environ.get("CODEX_REGION_CSV", "data/B004_Ascending_subset.csv"))
epi = {"Enterocyte","CD66+ Enterocyte","MUC1+ Enterocyte","CD57+ Enterocyte","Goblet","Paneth","Neuroendocrine","Cycling TA","TA"}
imm = {"Plasma","CD4+ T cell","B","CD8+ T","M2 Macrophage","M1 Macrophage","DC","Neutrophil","CD7+ Immune","NK"}
stro= {"Smooth muscle","Endothelial","Stroma","Lymphatic"}
def comp(ct):
    if ct in epi: return 0
    if ct in imm: return 1
    if ct in stro: return 2
    return 3   # Nerve, ICC and other
d["cls"] = d["Cell Type"].map(comp)
print("compartment counts:", np.bincount(d["cls"], minlength=4).tolist())

# normalise coordinates to the unit square
xy = d[["x","y"]].to_numpy(float)
xy = (xy - xy.min(0)) / (xy.max(0) - xy.min(0))
lab_cls = d["cls"].to_numpy()
pub = d["Neighborhood"].to_numpy()

# query grid (25x25 inside the region) and k-NN windows, k = 25 as in the paper's classifier
k=25; D=4; V=helmert(D)
g=np.linspace(0.02,0.98,25); gx,gy=np.meshgrid(g,g); grid=np.column_stack([gx.ravel(),gy.ravel()])
tree=cKDTree(xy)
_, win = tree.query(grid, k=k)
counts = np.stack([np.bincount(lab_cls[w], minlength=D) for w in win]).astype(float)
Yreal = ilr_from_counts(counts, k, V)
_, nn = tree.query(grid, k=1)
pub_grid = pub[nn]
print("grid cells:", len(grid), "| zero windows:", round(float((counts==0).any(1).mean()),3))

# classifier: synthetic training as in Section 9 (balanced by oversampling)
rng=np.random.default_rng(7)
Ys=[];Ls=[]
bg=np.array([0.55,0.20,0.15,0.10])
for rep in range(20):
    s=Sim(n=3000,k=25,seed=500+rep,bg=bg,center=(0.5,0.5),radius=0.15)
    lab=sample_labels(s.rng,s.in_pts,4,None,bg)
    # dominance: majority type 0 raised to 0.85 inside disc; even-mixing: uniform inside disc
    lab_dom=lab.copy(); m=s.in_pts; lab_dom[m]=np.where(rng.random(m.sum())<0.85,0,rng.integers(1,4,m.sum()))
    lab_even=lab.copy(); lab_even[m]=rng.integers(0,4,m.sum())
    for L,c in ((lab,0),(lab_dom,1),(lab_even,2)):
        cnt=np.stack([np.bincount(L[w],minlength=4) for w in s.idx]).astype(float)
        Yc=ilr_from_counts(cnt,25,V)
        yl=np.where(s.in_grid,c if c else 0,0)
        if c==0: yl=np.zeros(len(yl),int)
        if c==1: yl=np.where(s.in_grid,1,0)
        if c==2: yl=np.where(s.in_grid,2,0)
        Ys.append(Yc); Ls.append(yl)
X=np.vstack(Ys); y=np.concatenate(Ls)
# oversample minority classes to equal size
idx=[np.where(y==c)[0] for c in range(3)]; m_=max(len(i) for i in idx)
sel=np.concatenate([rng.choice(i,m_,replace=True) for i in idx])
clf=MLPClassifier(hidden_layer_sizes=(6,),alpha=0.01,max_iter=500,random_state=0).fit(X[sel],y[sel])
pred=clf.predict(Yreal)
print("predicted class counts on region grid:", np.bincount(pred,minlength=3).tolist())

# baselines and agreement with published neighbourhoods
ari_clf=adjusted_rand_score(pub_grid, pred)
km=KMeans(n_clusters=len(np.unique(pub_grid)),n_init=10,random_state=0).fit_predict(Yreal)
ari_km=adjusted_rand_score(pub_grid, km)
ari_null=np.mean([adjusted_rand_score(pub_grid, rng.permutation(pred)) for _ in range(200)])
out=dict(n_grid=len(grid),n_neighbourhoods=int(len(np.unique(pub_grid))),
         ARI_classifier=float(ari_clf), ARI_kmeans_on_ILR=float(ari_km), ARI_permuted_mean=float(ari_null),
         predicted_counts=np.bincount(pred,minlength=3).tolist())
print(json.dumps(out,indent=1))
json.dump(out,open("real_check.json","w"))
