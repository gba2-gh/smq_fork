"""Tie-exact common-anchor retrieval and fold-local cluster diagnostics."""
import numpy as np
import torch
from sklearn.metrics import adjusted_mutual_info_score
from script.seg import pipeline as PL


def tie_precision(dist, gallery_counts, eligible, k=10):
    """Expected class fraction under random ordering of ALL cutoff ties.

    Gallery items carry integer anchor multiplicities. Exact floating-point distance
    equality defines ties; no tolerance-based distance merging.
    """
    gc=gallery_counts.double(); mass=gc.sum(1)
    eligible=eligible & (mass[None,:]>0)
    d=dist.masked_fill(~eligible, float('inf'))
    vals,idx=torch.topk(d,min(k,d.shape[1]),dim=1,largest=False,sorted=True)
    masses=mass[idx]*torch.isfinite(vals)
    available=eligible.double()@mass
    target=available.clamp(max=k)
    cum=masses.cumsum(1)
    cutidx=(cum < target[:,None]).sum(1).clamp(max=vals.shape[1]-1)
    cutoff=vals.gather(1,cutidx[:,None])
    below=(d<cutoff)&eligible
    tied=(d==cutoff)&eligible
    lower=below.double()@gc
    tie=tied.double()@gc
    fraction=((target-lower.sum(1))/tie.sum(1).clamp(min=1)).clamp(0,1)
    prob=(lower+fraction[:,None]*tie)/target.clamp(min=1)[:,None]
    chance=(eligible.double()@gc)/available.clamp(min=1)[:,None]
    return prob,chance,available>0


def retrieval(V, rec, Q, groups, n_rec, centers=None, codes=None, chunk=256):
    C=Q.shape[1]; dev=PL.DEV
    out={k:np.zeros((n_rec,C)) for k in ('S','n','ch','saturated')}
    if centers is None:
        vt=torch.as_tensor(V,device=dev); qt=torch.as_tensor(Q,device=dev)
        gt=torch.as_tensor(groups,device=dev)
        for start in range(0,len(V),chunk):
            sl=slice(start,min(start+chunk,len(V)))
            distances=torch.cdist(vt[sl],vt)
            p,ch,valid=tie_precision(distances,qt,gt[sl,None]!=gt[None,:])
            for key,arr in (('S',p),('ch',ch),('n',torch.ones_like(p))):
                values=(qt[sl].double()*arr*valid[:,None]).cpu().numpy()
                np.add.at(out[key],rec[sl],values)
    else:
        K=len(centers)
        total=np.zeros((K,C)); np.add.at(total,codes,Q)
        d=PL.cdist_np(centers); d.fill_diagonal_(0.)
        # Prototypes with identical vectors also have mathematical zero distance.
        for i in range(K):
            identical=np.all(centers==centers[i],axis=1)
            d[i,torch.as_tensor(identical,device=dev)]=0.
        for group in np.unique(groups):
            ix=np.flatnonzero(groups==group)
            own=np.zeros_like(total); np.add.at(own,codes[ix],Q[ix])
            gallery=torch.as_tensor(total-own,device=dev)
            p,ch,valid=tie_precision(d,gallery,torch.ones_like(d,dtype=torch.bool))
            p=p.cpu().numpy();ch=ch.cpu().numpy();valid=valid.cpu().numpy()
            for key,arr in (('S',p),('ch',ch),('n',np.ones_like(p))):
                np.add.at(out[key],rec[ix],Q[ix]*arr[codes[ix]]*valid[codes[ix],None])
            sat=(total-own).sum(1)[codes[ix]]>=10
            np.add.at(out['saturated'],rec[ix],Q[ix]*sat[:,None])
    return out


def entropy(p):
    p=np.asarray(p,dtype=float);p=p[p>0];p=p/p.sum()
    return float(-(p*np.log(p)).sum())


def cluster_metrics(codes,y,K,C,weights=None):
    tab=np.zeros((K,C));np.add.at(tab,(codes,y),1 if weights is None else weights)
    pk=tab.sum(1);py=tab.sum(0);joint=entropy(tab.ravel());hy=entropy(py);hk=entropy(pk)
    h_y_k=max(0,joint-hk);h_k_y=max(0,joint-hy)
    if weights is None:
        ami=adjusted_mutual_info_score(y,codes)
    else:
        ami=adjusted_mutual_info_score(np.repeat(y,weights.astype(int)),np.repeat(codes,weights.astype(int)))
    return dict(contingency=tab,occupied=int((pk>0).sum()),effective_K=float(np.exp(hk)),
                H_label_given_code=h_y_k,H_code_given_label=h_k_y,
                homogeneity=1-h_y_k/hy if hy else 1.,completeness=1-h_k_y/hk if hk else 1.,
                AMI=float(ami),purity=float(tab.max(1).sum()/tab.sum()))


def self_test():
    dev=PL.DEV
    d=torch.tensor([[0.,1.,1.,1.]],device=dev)
    gc=torch.tensor([[1.,0.],[0.,5.],[3.,0.],[0.,1.]],device=dev)
    p,ch,v=tie_precision(d,gc,torch.ones_like(d,dtype=torch.bool),k=4)
    # one certain class-0 anchor, three draws from a 3:6 tie pool.
    np.testing.assert_allclose(p.cpu(),[[.5,.5]],atol=1e-10)
    e=torch.tensor([[False,True,True,True]],device=dev)
    p,_,_=tie_precision(d,gc,e,k=20)
    np.testing.assert_allclose(p.cpu(),[[1/3,2/3]],atol=1e-10)
    p,_,v=tie_precision(d,gc,torch.zeros_like(e),k=10)
    assert not v.any() and torch.isfinite(p).all()
    # Gallery-item permutations cannot alter tied results.
    ix=torch.tensor([3,1,0,2],device=dev)
    p,_,_=tie_precision(d[:,ix],gc[ix],torch.ones_like(e),k=4)
    np.testing.assert_allclose(p.cpu(),[[.5,.5]],atol=1e-10)
    print('tie retrieval checks passed',flush=True)

if __name__=='__main__':self_test()
