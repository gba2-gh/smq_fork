"""Prespecified fixed-window context readouts; refuses incomplete primary experiments."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('MKL_NUM_THREADS','4')
import sys,time,pickle
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from script.vocab.common import OUT,PL,partitions,seg_vectors
from script.vocab.run_exp1 import save
from script.vocab.run_exp2 import check_time
from script.vocab import diagnostic as D,readout as M
torch.set_num_threads(4)

def assert_primary_complete():
    for ds,nf in (('lara',4),('babel1',5)):
        for f in range(nf):
            for rep in ('mean','eight'):
                for fam in ('linear','mlp'):
                    assert (OUT/'exp1'/f'{ds}_{rep}_f{f}_{fam}.pkl').exists()
                for kind in ('fixed','oracle'):
                    for mult in (1,4,16):
                        for seed in (0,1,2):
                            assert (OUT/'exp2'/f'{ds}_{kind}_{rep}_f{f}_k{mult}_s{seed}.pkl').exists()

def main(ds,only_fold=None):
    data=PL.Data(ds);folds,nf=data.make_folds();edges=partitions(data,'fixed')
    dest=OUT/'exp3';dest.mkdir(exist_ok=True)
    rec=np.concatenate([np.full(len(e)-1,r) for r,e in enumerate(edges)])
    center=np.concatenate([np.flatnonzero(rec==r)[2:-2] for r in range(data.n)])
    neighbors=center[:,None]+np.arange(-2,3)[None,:]
    assert np.all(rec[neighbors]==rec[center,None])
    for f in range(nf):
        if only_fold is not None and f!=only_fold:continue
        check_time()
        cp=OUT/'exp2'/f'{ds}_fixed_eight_f{f}_k1_s0.pkl'
        with open(cp,'rb') as h:base=pickle.load(h)
        y=base['truth'][center];rr=rec[center];fit=folds[rr]!=f;ev=~fit
        purity=np.concatenate([PL.seg_label_counts(l,e,data.C).max(1)/np.diff(e) for l,e in zip(data.labels,edges)])[center]
        def readout(X,tag,mult,seed,basis_blocks=None):
            path=dest/f'{ds}_f{f}_k{mult}_s{seed}_{tag}.pkl'
            if path.exists():return
            check_time();t=time.time()
            res=D.run(X[fit],y[fit],data.group[rr[fit]],X[ev],data.C,scale=tag not in ('center','histogram'),basis_blocks=basis_blocks,
                      progress_label=f'{ds}:f{f}:K{mult}:s{seed}:{tag}')
            pred=res['predictions'][0]
            res.update(dataset=ds,fold=f,mult=mult,seed=seed,input=tag,dimension=X.shape[1],rec=rr[ev],truth=y[ev],
                       conf=M.confusion_by_recording(y[ev],pred,rr[ev],data.n,data.C),
                       purity=purity[ev],total_windows=len(rec),eligible_windows=len(center),seconds=time.time()-t)
            save(path,res)
            print('CONTEXT',ds,f,mult,seed,tag,round(res['seconds'],1),'BA',round(M.bal_acc_f1(res['conf'].sum(0))[0],4),flush=True)
        # Same continuous input is reused for both K values and every clustering seed.
        path=dest/f'{ds}_f{f}_k0_s0_continuous.pkl'
        if not path.exists():
            X=np.concatenate([seg_vectors('eight',z,e,base['mu'],base['sd']) for z,e in zip(data.z,edges)])
            xc=X[neighbors].reshape(len(center),-1);del X
            readout(xc,'continuous',0,0);del xc
        for mult in (1,16):
            for seed in (0,1,2):
                with open(OUT/'exp2'/f'{ds}_fixed_eight_f{f}_k{mult}_s{seed}.pkl','rb') as h:cell=pickle.load(h)
                code=cell['codes'];proto=cell['centers'];K=len(proto)
                eye=np.eye(K,dtype=np.float32)
                readout(eye[code[center]],'center',mult,seed)
                readout(eye[code[neighbors]].mean(1),'histogram',mult,seed)
                readout(proto.astype(np.float64)[code[neighbors]].mean(1),'unordered',mult,seed,[proto])
                xp=proto[code[neighbors]].reshape(len(center),-1)
                readout(xp,'ordered',mult,seed,[proto]*5);del xp
    print('COMPLETE exp3',ds,flush=True)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',choices=('lara','babel1'))
    parser.add_argument('--fold',type=int)
    args=parser.parse_args()
    assert_primary_complete()
    for ds in ([args.dataset] if args.dataset else ('lara','babel1')):main(ds,args.fold)
