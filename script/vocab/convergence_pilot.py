"""Training-objective-only numerical audit; never evaluates outer labels."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('MKL_NUM_THREADS','4')
import sys,time,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from script.vocab.common import OUT,PL,partitions,seg_vectors
from script.vocab.analyze import load
from script.vocab.readout import standardise_fit,_class_weights
torch.set_num_threads(4)
data=PL.Data('lara');folds,_=data.make_folds();edges=partitions(data,'fixed')
base=load(OUT/'exp2'/'lara_fixed_eight_f0_k1_s0.pkl');rec=base['rec']
center=np.concatenate([np.flatnonzero(rec==r)[2:-2] for r in range(data.n)])
neighbors=center[:,None]+np.arange(-2,3)[None,:]
fit=folds[rec[center]]!=0
X=np.concatenate([seg_vectors('eight',z,e,base['mu'],base['sd']) for z,e in zip(data.z,edges)])
X=X[neighbors[fit]].reshape(fit.sum(),-1);y=base['truth'][center[fit]]
mu,sd=standardise_fit(X);X=((X-mu)/sd).astype(np.float32)
previous_path=OUT/'archive'/'fp32_150'/'exp3'/'lara_f0_k0_s0_continuous.pkl'
if not previous_path.exists():previous_path=OUT/'exp3'/'lara_f0_k0_s0_continuous.pkl'
previous=load(previous_path)
l2=previous['selected']['config'];C=data.C;dev=PL.DEV
rows=[]
for dtype,limit,history,tol in [(torch.float32,150,20,1e-9),(torch.float64,1000,50,1e-12)]:
    t=time.time();xt=torch.tensor(X,device=dev,dtype=dtype);yt=torch.tensor(y,device=dev)
    cw=torch.tensor(_class_weights(y,C),dtype=dtype,device=dev)
    model=torch.nn.Linear(X.shape[1],C,dtype=dtype,device=dev)
    torch.nn.init.zeros_(model.weight);torch.nn.init.zeros_(model.bias)
    opt=torch.optim.LBFGS(model.parameters(),lr=1,max_iter=limit,history_size=history,line_search_fn='strong_wolfe',tolerance_grad=1e-6,tolerance_change=tol)
    def closure():
        opt.zero_grad()
        loss=torch.nn.functional.cross_entropy(model(xt),yt,weight=cw,reduction='sum')/cw[yt].sum()+.5*l2*model.weight.square().sum()
        loss.backward();return loss
    opt.step(closure)
    loss=float(closure().detach());gradient=float(max(p.grad.abs().max().item() for p in model.parameters()))
    row=dict(dtype=str(dtype),limit=limit,history=history,tolerance_change=tol,l2=l2,n=len(y),dimension=X.shape[1],
             iterations=opt.state[model.weight]['n_iter'],gradient_max=gradient,training_objective=loss,seconds=time.time()-t)
    rows.append(row);print(json.dumps(row),flush=True)
    del model,opt,xt,yt,cw
(OUT/'CONVERGENCE_PILOT.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
