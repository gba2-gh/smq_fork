"""Run oracle segment diagnostics, resumably at outer-fold/readout granularity."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
import sys, time, pickle, json, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from script.vocab.common import OUT, PL, partitions, seg_vectors
from script.vocab import diagnostic as D
from script.vocab import readout as M
torch.set_num_threads(4)

def save(path, obj):
    tmp=path.with_suffix(f'.{os.getpid()}.tmp')
    with open(tmp,'wb') as f: pickle.dump(obj,f,protocol=4)
    tmp.replace(path)

def main(ds):
    t=time.time(); data=PL.Data(ds); folds,nf=data.make_folds(); edges=partitions(data,'oracle')
    dest=OUT/'exp1'; dest.mkdir(exist_ok=True)
    ys=[data.labels[r][e[:-1]] for r,e in enumerate(edges)]
    lengths=[np.diff(e) for e in edges]
    rec=np.concatenate([np.full(len(y),r) for r,y in enumerate(ys)])
    y=np.concatenate(ys); dur=np.concatenate(lengths)
    meta=dict(names=data.names,groups=data.group,folds=folds,class_names=data.class_names,
              anchor_hash=PL.anchors_hash(data.make_anchors(),data))
    save(dest/f'{ds}_meta.pkl',meta)
    for rep in ('mean','eight'):
        X=np.concatenate([seg_vectors(rep,z,e,0.,1.) for z,e in zip(data.z,edges)])
        for f in range(nf):
            fit=folds[rec]!=f; ev=~fit
            assert not set(data.group[rec[fit]]) & set(data.group[rec[ev]])
            for family in ('linear','mlp'):
                path=dest/f'{ds}_{rep}_f{f}_{family}.pkl'
                if path.exists(): continue
                start=time.time()
                result=D.run(X[fit],y[fit],data.group[rec[fit]],X[ev],data.C,family)
                result.update(rec=rec[ev],truth=y[ev],item_duration=dur[ev],fold=f,representation=rep,family=family)
                for tag,w in (('segment',None),('duration',dur[ev])):
                    result[tag]=np.asarray([M.confusion_by_recording(y[ev],p,rec[ev],data.n,data.C,w) for p in result['predictions']])
                    result['baseline_'+tag]=M.baseline_confusions(y[fit],y[ev],rec[ev],data.n,data.C,w)
                result['seconds']=time.time()-start
                save(path,result)
                print(ds,rep,f,family,'seconds',round(result['seconds'],1),'BA',M.bal_acc_f1(result['segment'].mean(0).sum(0))[0],flush=True)
        del X
    print('COMPLETE',ds,round(time.time()-t,1),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['lara','babel1','both'],default='both');a=p.parse_args()
    for ds in (('lara','babel1') if a.dataset=='both' else (a.dataset,)):main(ds)
