"""Loss-preserving row-space coordinates for redundant prototype inputs.

This is an internal linear solver parameterization, not a new representation.
Q has orthonormal columns spanning every standardized prototype. For W=UQ^T,
XW^T=(XQ)U^T and ||W||_F=||U||_F; discarded directions have zero data support.
"""
import numpy as np
import torch
from script.vocab.readout import DEV

def make_projectors(blocks,mu,sd):
    mu=np.asarray(mu).ravel();sd=np.asarray(sd).ravel();offset=0;out=[]
    for atoms in blocks:
        d=atoms.shape[1];end=offset+d
        normalized=(np.asarray(atoms,dtype=np.float64)-mu[offset:end])/sd[offset:end]
        _,s,vh=np.linalg.svd(normalized,full_matrices=False)
        rank=max(1,int((s>max(1e-12,s[0]*1e-10)).sum()))
        q=np.ascontiguousarray(vh[:rank].T)
        np.testing.assert_allclose(q.T@q,np.eye(rank),atol=1e-10)
        residual=np.linalg.norm(normalized-(normalized@q)@q.T)/max(np.linalg.norm(normalized),1.)
        assert residual<1e-8,residual
        out.append((offset,end,q,float(residual)));offset=end
    assert offset==len(mu)
    return out

def project(X,mu,sd,projectors):
    mu=np.asarray(mu).ravel();sd=np.asarray(sd).ravel();out=[]
    for a,b,q,_ in projectors:
        xt=torch.as_tensor(X[:,a:b],device=DEV,dtype=torch.float64)
        mt=torch.as_tensor(mu[a:b],device=DEV,dtype=torch.float64)
        st=torch.as_tensor(sd[a:b],device=DEV,dtype=torch.float64)
        qt=torch.as_tensor(q,device=DEV,dtype=torch.float64)
        out.append((((xt-mt)/st)@qt).cpu().numpy())
        del xt,mt,st,qt
    return np.concatenate(out,axis=1)

def info(projectors):
    return dict(block_ranks=[q.shape[1] for _,_,q,_ in projectors],
                maximum_dictionary_projection_error=max(e for _,_,_,e in projectors))

def self_test():
    rng=np.random.default_rng(543);atoms=rng.normal(size=(7,35));codes=rng.integers(0,7,size=(80,5))
    errors=[]
    for ordered in (False,True):
        X=atoms[codes].reshape(80,-1) if ordered else atoms[codes].mean(1)
        mu=X[:60].mean(0,keepdims=True);sd=X[:60].std(0,keepdims=True)
        blocks=[atoms]*5 if ordered else [atoms]
        projectors=make_projectors(blocks,mu,sd);xp=project(X,mu,sd,projectors)
        weights=rng.normal(size=(3,xp.shape[1]));full=np.zeros((3,X.shape[1]));offset=0
        for a,b,q,_ in projectors:
            r=q.shape[1];full[:,a:b]=weights[:,offset:offset+r]@q.T;offset+=r
        np.testing.assert_allclose(xp@weights.T,((X-mu)/sd)@full.T,atol=1e-10)
        np.testing.assert_allclose(np.sum(full**2),np.sum(weights**2),atol=1e-10)
        errors.append(dict(ordered=ordered,maximum_logit_error=float(np.abs(xp@weights.T-((X-mu)/sd)@full.T).max()),
                           penalty_error=float(abs(np.sum(full**2)-np.sum(weights**2)))))
    print('Prototype projection preserves logits and L2 penalty on training and held-out combinations',flush=True)
    return errors

if __name__=='__main__':self_test()
