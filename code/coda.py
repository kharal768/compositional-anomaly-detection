import numpy as np
from scipy.spatial import cKDTree

def helmert(D):
    V = np.zeros((D, D-1))
    for j in range(1, D):
        V[:j, j-1] = 1.0
        V[j, j-1] = -j
        V[:, j-1] /= np.sqrt(j*(j+1))
    return V

def zero_replace(X, delta=0.65, mode="D", k=None):
    D = X.shape[1]
    Xr = X.copy()
    z = (X == 0)
    if not z.any():
        return Xr
    val = delta / D if mode == "D" else delta / k
    rows = np.where(z.any(1))[0]
    for i in rows:
        zi = z[i]
        s = val * zi.sum()
        Xr[i, zi] = val
        Xr[i, ~zi] = X[i, ~zi] * (1 - s)
    return Xr

def ilr_from_counts(counts, k, V, delta=0.65, zmode="D"):
    x = zero_replace(counts / k, delta, zmode, k)
    lx = np.log(x)
    clr = lx - lx.mean(1, keepdims=True)
    return clr @ V

def make_grid(m, lo=0.02, hi=0.98):
    g = np.linspace(lo, hi, m)
    X, Y = np.meshgrid(g, g, indexing="ij")
    # R expand.grid: x varies fastest
    gx, gy = np.meshgrid(g, g)
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    return grid, g[1]-g[0]

def knn_index(points, grid, k):
    _, idx = cKDTree(points).query(grid, k=k)
    return idx

def counts_from_labels(labels, idx, D):
    oh = np.eye(D)[labels]            # (n,D)
    return oh[idx].sum(1)             # (G,D)

def sample_labels(rng, inside, D, effect=None, bg=None):
    n = inside.shape[0]
    bg = np.full(D, 1.0/D) if bg is None else np.asarray(bg)
    u = rng.random(n)
    cum = np.cumsum(bg)
    lab = np.searchsorted(cum, u, side="right").clip(0, D-1)
    if effect is not None:
        m = inside
        u2 = rng.random(m.sum())
        e = effect
        rest = (1-e)/(D-1)
        l2 = np.where(u2 < e, 0, 1 + np.floor((u2 - e)/rest)).clip(0, D-1).astype(int)
        lab[m] = l2
    return lab

class Sim:
    def __init__(self, n=2000, D=4, k=30, gridm=35, center=(0.3,0.7), radius=0.13, seed=1, bg=None, delta=0.65, zmode="D"):
        self.rng = np.random.default_rng(seed)
        self.n, self.D, self.k = n, D, k
        self.pts = self.rng.random((n, 2))
        self.grid, self.dgrid = make_grid(gridm)
        self.idx = knn_index(self.pts, self.grid, k)
        c = np.array(center)
        self.in_pts = ((self.pts - c)**2).sum(1) < radius**2
        self.in_grid = ((self.grid - c)**2).sum(1) < radius**2
        self.dist_c = np.sqrt(((self.grid - c)**2).sum(1)) - radius
        self.V = helmert(D)
        self.bg = bg
        self.delta, self.zmode = delta, zmode

    def Y(self, labels):
        c = counts_from_labels(labels, self.idx, self.D)
        return ilr_from_counts(c, self.k, self.V, self.delta, self.zmode)

    def null_Y(self):
        return self.Y(sample_labels(self.rng, self.in_pts, self.D, None, self.bg))

    def alt_Y(self, effect, eps=0.0):
        lab = sample_labels(self.rng, self.in_pts, self.D, effect, self.bg)
        if eps > 0:
            flip = self.rng.random(self.n) < eps
            lab[flip] = self.rng.integers(0, self.D, flip.sum())
        return self.Y(lab)

def t2(Y, mu, Sinv):
    d = Y - mu
    return np.einsum("ij,jk,ik->i", d, Sinv, d)

def run_power(sim, effects, B_ref=500, B=500, M=300, W=None, B_cov=300, eps=0.0, insample=False):
    """Return dict effect -> (power, se, fa, se). W: smoothing matrix (EWMA) or None."""
    G = sim.grid.shape[0]
    sm = (lambda Y: W @ Y) if W is not None else (lambda Y: Y)
    if not insample:
        raw = [sim.null_Y() for _ in range(B_ref)]
        mu = np.vstack(raw).mean(0)
        if W is None:
            S = np.cov(np.vstack(raw).T)
        else:
            Z = [sm(sim.null_Y()) for _ in range(B_cov)]
            S = np.cov(np.vstack(Z).T)
        Sinv = np.linalg.inv(S)
        stat = lambda Y: t2(sm(Y), mu, Sinv)
    else:
        def stat(Y):
            Z = sm(Y)
            m = Z.mean(0); Sx = np.cov(Z.T)
            return t2(Z, m, np.linalg.inv(Sx))
    nulls = np.array([stat(sim.null_Y()) for _ in range(B)])
    crit = np.quantile(nulls, 0.95, axis=0)
    out = {}
    for e in effects:
        hits = np.empty(M); fa = np.empty(M)
        for m in range(M):
            f = stat(sim.alt_Y(e, eps)) > crit
            hits[m] = f[sim.in_grid].mean(); fa[m] = f[~sim.in_grid].mean()
        out[e] = (hits.mean(), hits.std(ddof=1)/np.sqrt(M), fa.mean(), fa.std(ddof=1)/np.sqrt(M))
    return out

def smooth_matrix(grid, lam, delta):
    Dm = np.sqrt(((grid[:,None,:]-grid[None,:,:])**2).sum(-1))
    W = (1-lam)**(Dm/delta)
    return W / W.sum(1, keepdims=True)
