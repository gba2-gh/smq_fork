"""Factorial vocabulary experiment; each fitted cell saved immediately."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('MKL_NUM_THREADS','4')
import sys,time,pickle,argparse,hashlib
from pathlib import Path
from datetime import datetime,timezone
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
from script.vocab.common import OUT,PL,R,partitions,seg_vectors,seg_majority_label
from script.vocab.run_exp1 import save
from script.vocab import diagnostic as D,readout as M,evaluate as E
torch.set_num_threads(4)
DEADLINE=float(os.environ.get('SMQ_VOCAB_DEADLINE',datetime(2026,9,21,20,50,tzinfo=timezone.utc).timestamp()))

def check_time():
    if time.time()>DEADLINE:raise SystemExit('Primary execution cutoff; preserve cells and report partial results.')

def main(ds,only_fold=None):
    data=PL.Data(ds);folds,nf=data.make_folds();anchors=data.make_anchors()
    dest=OUT/'exp2';dest.mkdir(exist_ok=True)
    save(dest/f'{ds}_meta.pkl',dict(names=data.names,groups=data.group,folds=folds,class_names=data.class_names,
                                   anchor_hash=PL.anchors_hash(anchors,data),fps=data.fps))
    edges_by={kind:partitions(data,kind) for kind in ('fixed','oracle')}
    for f in range(nf):
        if only_fold is not None and f!=only_fold:continue
        check_time();fitrec=np.flatnonzero(folds!=f);evrec=np.flatnonzero(folds==f)
        mu,sd=R.fit_scaler([data.z[r] for r in fitrec])
        for kind,edges in edges_by.items():
            rec=np.concatenate([np.full(len(e)-1,r) for r,e in enumerate(edges)])
            y=np.concatenate([seg_majority_label(l,e,data.C)[0] for l,e in zip(data.labels,edges)])
            dur=np.concatenate([np.diff(e) for e in edges]);fit=folds[rec]!=f;ev=~fit
            Q=[]
            for r,e in enumerate(edges):
                counts=np.zeros((len(e)-1,data.C));sid=np.searchsorted(e,anchors[r],side='right')-1
                np.add.at(counts,(sid,data.labels[r][anchors[r]]),1);Q.append(counts)
            Q=np.concatenate(Q);keep=ev&(Q.sum(1)>0)
            for rep in ('mean','eight'):
                base=f'{ds}_{kind}_{rep}_f{f}'
                needed=any(not (dest/f'{base}_k{mult}_s{seed}.pkl').exists() for mult in (1,4,16) for seed in (0,1,2))
                if not needed:continue
                check_time();t=time.time()
                X=np.concatenate([seg_vectors(rep,z,e,mu,sd) for z,e in zip(data.z,edges)])
                xf=X[fit]
                continuous_path=dest/f'{base}_continuous.pkl'
                if not continuous_path.exists():
                    cont=E.retrieval(X[keep],rec[keep],Q[keep],data.group[rec[keep]],data.n)
                    save(continuous_path,dict(retrieval=cont,seconds=time.time()-t))
                print('FEATURES',base,'nfit',len(xf),'dim',xf.shape[1],'seconds',round(time.time()-t,1),flush=True)
                for mult in (1,4,16):
                    K=data.C*mult
                    for seed in (0,1,2):
                        check_time();path=dest/f'{base}_k{mult}_s{seed}.pkl'
                        if path.exists():continue
                        t=time.time()
                        with threadpool_limits(limits=4):
                            km=KMeans(n_clusters=K,n_init=5,max_iter=300,random_state=seed).fit(xf)
                        centers=km.cluster_centers_.astype(np.float32)
                        code=PL.cdist_np(X,centers).argmin(1).cpu().numpy()
                        result=dict(dataset=ds,kind=kind,rep=rep,fold=f,mult=mult,K=K,seed=seed,
                                    centers=centers,codes=code,rec=rec,truth=y,duration=dur,mu=mu,sd=sd,
                                    inertia=km.inertia_,n_iter=km.n_iter_,fit_n=int(fit.sum()),eval_n=int(ev.sum()))
                        result['retrieval']=E.retrieval(None,rec[keep],Q[keep],data.group[rec[keep]],data.n,centers,code[keep])
                        result['cluster_segment']=E.cluster_metrics(code[ev],y[ev],K,data.C)
                        # Duration-weighted majority segment label, separate from anchor labels.
                        result['cluster_duration']=E.cluster_metrics(code[ev],y[ev],K,data.C,dur[ev])
                        result['fit_occupied']=int(len(np.unique(code[fit])))
                        onehot=np.eye(K,dtype=np.float32)
                        ro=D.run(onehot[code[fit]],y[fit],data.group[rec[fit]],onehot[code[ev]],data.C,scale=False)
                        pred=ro.pop('predictions')[0];result['readout']=ro
                        result['predictions']=pred
                        for tag,w in (('segment',None),('duration',dur[ev])):
                            result['conf_'+tag]=M.confusion_by_recording(y[ev],pred,rec[ev],data.n,data.C,w)
                        result['class_coverage']=dict(predicted=int(len(np.unique(pred))),recalled=int(len(np.unique(y[ev][pred==y[ev]]))),total=data.C)
                        if kind=='oracle':
                            trans=np.zeros((data.n,data.C,data.C,2),dtype=np.int64)
                            for r in evrec:
                                ix=np.flatnonzero(rec==r);a=y[ix[:-1]];b=y[ix[1:]]
                                retain=(code[ix[:-1]]!=code[ix[1:]]).astype(int)
                                assert np.all(a!=b)
                                np.add.at(trans,(np.full(len(a),r),a,b,retain),1)
                            result['transitions']=trans
                        result['seconds']=time.time()-t
                        result['discrete_payload_sha256']=hashlib.sha256(result['centers'].tobytes()+result['codes'].tobytes()+result['retrieval']['S'].tobytes()).hexdigest()
                        save(path,result)
                        print('CELL',base,'K',K,'seed',seed,'seconds',round(result['seconds'],1),'AMI',round(result['cluster_segment']['AMI'],4),flush=True)
                del X,xf
    print('COMPLETE exp2',ds,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['lara','babel1','both'],default='both');p.add_argument('--fold',type=int);a=p.parse_args()
    E.self_test()
    for ds in (('lara','babel1') if a.dataset=='both' else (a.dataset,)):main(ds,a.fold)
