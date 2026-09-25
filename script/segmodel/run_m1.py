"""Run M1 only. It deliberately does not contain HMM/HSMM or supervised selection."""
from __future__ import annotations
import argparse, hashlib, json, os, platform, time
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from script.segmodel.core import *
from script.exp2round.q1q2.core import load_dataset, score, map_predictions, majority_code_predictions, sha256_file, CHECKPOINTS, D7_REFERENCE

def append(path,row):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',encoding='utf8') as f:f.write(json.dumps(row)+'\n')

def expanded(edges, labels, lengths, width):
    out=[]
    for e,lab,lens in zip(edges,labels,lengths):
        y=np.empty(int(lens.sum()),np.int32)
        for a,b,c in zip(e[:-1],e[1:],lab):
            start=int(np.sum(lens[:a])); end=int(np.sum(lens[:b])); y[start:end]=c
        out.append(y)
    return out

def gate(data,ds):
    pred=majority_code_predictions(data.gts,data.codes); mapped=map_predictions(data.gts,pred,'hungarian'); met=score(data.gts,mapped)
    return met, abs(met['MoF']-D7_REFERENCE[ds])<=1.5

def run(ds,standardize,k,t_mult,seed,transductive=True):
    data=load_dataset(ds); arrays=per_recording_zscore(data.arrays) if standardize else data.arrays
    n=len(arrays); fit=np.arange(n)
    prepared=prepare(arrays,WINDOW[ds],fit); features=prepared.transformed
    km=KMeans(n_clusters=k,init='k-means++',n_init=5,max_iter=300,tol=1e-4,algorithm='lloyd',random_state=seed).fit(prepared.fit_features)
    tau=base_temperature(prepared.fit_features,km.cluster_centers_)
    if tau is None: raise RuntimeError('no positive temperature gap')
    p=posterior(features,km.cluster_centers_,tau*t_mult)
    # The label-free calibration target is one segment per elapsed second.
    target=sum(int(np.ceil(len(x) / FPS[ds])) for x in arrays)
    max_windows=max(1, int(np.floor(10 * FPS[ds] / WINDOW[ds])))
    init=fit_centroids(p,prepared.real_lengths,CLASSES[ds],seed)
    costs=[candidate_costs(histogram_prefix(x,l),FPS[ds],max_windows,init)[0] for x,l in zip(p,prepared.real_lengths)]
    lam,ldiag=calibrate_lambda(costs,target)
    centroids,edges,labels,trace=fit_segmental(p,prepared.real_lengths,WINDOW[ds],FPS[ds],CLASSES[ds],seed,lam)
    raw=expanded(edges,labels,prepared.real_lengths,WINDOW[ds]); mapped=map_predictions(data.gts,raw,'hungarian'); metrics=score(data.gts,mapped)
    nseg=sum(len(x) for x in labels); gtseg=sum(len(rle(x)[0]) for x in data.gts)
    unseen=0; total=0
    for q in p:
        top=q.argmax(1); total+=len(top); unseen+=np.count_nonzero(np.bincount(top,minlength=k)==0) # retained for schema; corrected below in inductive runner
    tag=f'{ds}_std{int(standardize)}_k{k}_t{t_mult:g}_s{seed}'
    model=OUT/'models'/f'{tag}.npz'; save_npz(model,centroids=centroids,prototypes=km.cluster_centers_,lambda_value=np.asarray(lam),fit_ids=prepared.fit_ids,trace=np.asarray([json.dumps(x) for x in trace]))
    for name,y,e in zip(data.names,mapped,edges):save_npz(OUT/'predictions'/f'{tag}_{name}.npz',labels=y,window_edges=e*WINDOW[ds])
    return dict(dataset=ds,protocol='transductive',standardize_per_recording=standardize,representation='soft',window=WINDOW[ds],num_units=k,temperature_multiplier=t_mult,seed=seed,lambda_value=lam,**ldiag,**metrics,predicted_segments=nseg,gt_segments=gtseg,segment_count_ratio=nseg/gtseg,unseen_top_unit_fraction=None,status='degenerate' if not .5<=nseg/gtseg<=2 else 'complete',rounds=len(trace),runtime_seconds=None,model=str(model))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--datasets',nargs='+',default=['hugadb','lara']);ap.add_argument('--primary',action='store_true');ap.add_argument('--all',action='store_true');ap.add_argument('--fresh',action='store_true');ap.add_argument('--deadline-hours',type=float,default=6.0);args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True); started=time.time(); cells=OUT/'cells.jsonl'
    if args.fresh and cells.exists(): cells.unlink()
    manifest={'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'milestone':1,'threads':min(8,os.cpu_count() or 1),'deadline_hours':args.deadline_hours,'checkpoint_sha256':{d:sha256_file(CHECKPOINTS[d]) for d in args.datasets},'python':platform.python_version()}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    grid=[(500,1.)] if args.primary or not args.all else [(k,t) for k in (200,500,1000) for t in (1.,2.)]
    for ds in args.datasets:
        data=load_dataset(ds); d7,passed=gate(data,ds);append(cells,dict(dataset=ds,method='d7',seed=None,status='complete' if passed else 'failed_gate',**d7))
        if not passed: continue
        for std in (False,True):
            for k,t in grid:
                for seed in SEEDS:
                    if time.time() - started >= args.deadline_hours * 3600:
                        append(cells,dict(dataset=ds,protocol='transductive',standardize_per_recording=std,num_units=k,temperature_multiplier=t,seed=seed,status='not_run',reason='compute deadline'))
                        continue
                    started_cell=time.time()
                    try:
                        row=run(ds,std,k,t,seed);row['runtime_seconds']=time.time()-started_cell
                    except Exception as e: row=dict(dataset=ds,protocol='transductive',standardize_per_recording=std,num_units=k,temperature_multiplier=t,seed=seed,status='failed',reason=repr(e))
                    append(cells,row);print(json.dumps(row),flush=True)
    manifest['completed_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());manifest['elapsed_seconds']=time.time()-started;(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__': main()
