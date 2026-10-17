import numpy as np
def scan_max(pts, cases, max_frac=0.5, order=None):
    """Kulldorff (1997) Poisson scan (implemented in Python; not SaTScan). Circular windows centred
    on each point, containing up to max_frac of the population. Returns max LLR and its window."""
    n=len(pts); C=cases.sum(); mmax=max(1,int(max_frac*n))
    if order is None:
        D2=((pts[:,None,:]-pts[None,:,:])**2).sum(-1)
        order=np.argsort(D2,axis=1,kind="stable")[:,:mmax]
    c=np.cumsum(cases[order],axis=1)            # cases in each window
    m=np.arange(1,mmax+1)[None,:]
    E=C*m/n
    with np.errstate(divide="ignore",invalid="ignore"):
        t1=np.where(c>0, c*np.log(c/E), 0.0)
        cc=C-c; ee=C-E
        t2=np.where(cc>0, cc*np.log(cc/ee), 0.0)
    llr=np.where(c>E, t1+t2, 0.0)
    i,j=np.unravel_index(np.argmax(llr),llr.shape)
    return llr[i,j], dict(centre=i, size=int(j+1), cases=int(c[i,j]), expected=float(C*(j+1)/n))
def scan_test(pts, cases, B=199, rng=None, max_frac=0.5):
    """Monte Carlo p-value by permuting case labels (keeps total cases fixed)."""
    rng=rng or np.random.default_rng(0)
    n=len(pts); D2=((pts[:,None,:]-pts[None,:,:])**2).sum(-1)
    order=np.argsort(D2,axis=1,kind="stable")[:,:max(1,int(max_frac*n))]
    obs,w=scan_max(pts,cases,max_frac,order)
    null=np.array([scan_max(pts,rng.permutation(cases),max_frac,order)[0] for _ in range(B)])
    p=(1+(null>=obs).sum())/(B+1)
    return obs,p,w,np.quantile(null,0.95),order
