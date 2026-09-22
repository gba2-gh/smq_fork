"""Independent small numerical checks, artifact audit, and provenance manifest."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('MKL_NUM_THREADS','4')
import sys,pickle,json,hashlib,platform,subprocess,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch,sklearn,scipy
from script.vocab.common import OUT,ROOT,PL,mean_vectors
from script.vocab.evaluate import tie_precision,self_test
from script.vocab.analyze import load,csvout
from script.vocab import readout as D
torch.set_num_threads(4)

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def numerical_checks():
    self_test();rng=np.random.default_rng(901)
    for trial in range(100):
        G=17;C=4;N=5;k=int(rng.integers(1,20))
        distances=rng.integers(0,6,size=(N,G)).astype(np.float32)
        counts=rng.integers(0,5,size=(G,C));eligible=rng.random((N,G))>.3
        actual,ch,valid=tie_precision(torch.tensor(distances,device=PL.DEV),torch.tensor(counts,device=PL.DEV),torch.tensor(eligible,device=PL.DEV),k)
        expected=np.zeros((N,C))
        for q in range(N):
            labels=[];d=[]
            for g in range(G):
                if not eligible[q,g]:continue
                for c in range(C):
                    labels.extend([c]*int(counts[g,c]));d.extend([distances[q,g]]*int(counts[g,c]))
            labels=np.array(labels);d=np.array(d);den=min(k,len(labels))
            if den:
                cutoff=np.sort(d)[den-1];before=d<cutoff;ties=d==cutoff
                expected[q]=(np.bincount(labels[before],minlength=C)+(den-before.sum())*np.bincount(labels[ties],minlength=C)/ties.sum())/den
        np.testing.assert_allclose(actual.cpu().numpy(),expected,atol=1e-12)
    z=rng.normal(size=(23,7)).astype(np.float32);edges=np.array([0,1,4,15,23]);mu=rng.normal(size=7).astype(np.float32);sd=np.ones(7,dtype=np.float32)*2
    actual=mean_vectors(z,edges,mu,sd)
    expected=np.stack([((z[a:b]-mu)/sd).mean(0) for a,b in zip(edges[:-1],edges[1:])])
    np.testing.assert_allclose(actual,expected,atol=2e-7)
    from script.vocab.linear_projection import self_test as projection_check
    projection=projection_check()
    (OUT/'NUMERICAL_CHECKS.json').write_text(json.dumps(dict(expanded_anchor_cases=100,empty_gallery_and_group_exclusion='passed',
                                                            mean_pooling='passed',prototype_projection=projection),indent=2),encoding='utf-8')
    print('100 expanded-anchor tie references and mean-pooling checks passed',flush=True)

def manifest():
    obj=dict(start_utc='2026-09-21T13:36:57Z',deadline_utc='2026-09-21T21:36:57Z',python=platform.python_version(),
             numpy=np.__version__,torch=torch.__version__,sklearn=sklearn.__version__,scipy=scipy.__version__,
             device=torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu',threads=4,
             git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
             working_tree_note='Existing unrelated modified and untracked files preserved. Source hashes identify this run.',
             source_sha256={str(p.relative_to(ROOT)):sha(p) for folder in ('script/vocab','script/seg') for p in sorted((ROOT/folder).glob('*.py'))},datasets={})
    for ds in ('lara','babel1'):
        cache=ROOT/'results'/'seg'/'cache'/ds
        paths=sorted(cache.glob('*.npy'))
        meta=load(OUT/'exp1'/f'{ds}_meta.pkl')
        frame_hashes={p.name:sha(p) for p in paths}
        label_hashes={name:sha(ROOT/'data'/ds/'groundTruth'/f'{name}.txt') for name in meta['names']}
        obj['datasets'][ds]=dict(recordings=len(paths),cache_bytes=sum(p.stat().st_size for p in paths),
                                  cache_meta_sha256=sha(cache/'_meta.npz'),
                                  frame_file_sha256=frame_hashes,annotation_file_sha256=label_hashes,
                                  checkpoint_sha256=sha(ROOT/'models'/'pretrained'/f'{ds}.model'),
                                  anchor_sha256=meta['anchor_hash'],classes=meta['class_names'],
                                  folds=meta['folds'].tolist(),groups=meta['groups'].tolist(),names=meta['names'])
    (OUT/'MANIFEST.json').write_text(json.dumps(obj,indent=2),encoding='utf-8')

def audit(require_complete=False):
    report=dict(numerical_checks='passed',experiments={},issues=[])
    solver_rows=[]
    for exp,expected in ((1,36),(2,324),(3,225)):
        folder=OUT/f'exp{exp}'
        paths=sorted(folder.glob('*_k*_s*.pkl')) if exp!=1 else sorted(folder.glob('*_f*_*.pkl'))
        if exp==2:paths=[p for p in paths if not p.name.endswith('_continuous.pkl')]
        report['experiments'][str(exp)]=dict(completed=len(paths),expected=expected)
        if require_complete:assert len(paths)==expected,(exp,len(paths),expected)
        for p in paths:
            cell=load(p)
            ro=cell['readout'] if exp==2 else cell
            if exp!=1 or cell['family']=='linear':
                assert ro.get('solver')==D.LINEAR_SOLVER,(p.name,ro.get('solver'))
                optimization=ro['optimization'][0]
                solver_rows.append(dict(experiment=exp,file=p.name,solver_dimension=ro['solver_dimension'],
                                        iterations=optimization['iterations'],gradient_max=optimization['gradient_max'],
                                        training_objective=optimization['training_objective'],
                                        tuning_fits_at_cap=sum(x['iterations']>=D.LIN_ITERS for x in ro['tuning'])))
                if optimization['iterations']>=D.LIN_ITERS:
                    report['issues'].append(dict(file=p.name,note='Selected linear refit reached its iteration cap.',gradient_max=optimization['gradient_max']))
                if ro.get('projection') is not None:
                    assert ro['projection']['maximum_dictionary_projection_error']<1e-8
            if exp==1:
                meta=load(folder/f"{p.name.split('_')[0]}_meta.pkl")
                forbidden=set(meta['groups'][meta['folds']==cell['fold']])
                assert not forbidden & (set(cell['validation_groups'])|set(cell['training_groups']))
                assert len(cell['rec'])==len(cell['truth'])==len(cell['item_duration'])
                for conf in cell['segment']:assert conf.sum()==len(cell['truth'])
                for conf in cell['duration']:assert conf.sum()==cell['item_duration'].sum()
                C=len(meta['class_names']);n=len(meta['names'])
                for s,pred in enumerate(cell['predictions']):
                    index=(cell['rec']*C+cell['truth'])*C+pred
                    for key,w in [('segment',None),('duration',cell['item_duration'])]:
                        independent=np.bincount(index,weights=w,minlength=n*C*C).reshape(n,C,C)
                        np.testing.assert_array_equal(independent,cell[key][s])
                assert not set(cell['validation_groups'])&set(cell['training_groups'])
            elif exp==2:
                meta=load(folder/f"{cell['dataset']}_meta.pkl");ev=meta['folds'][cell['rec']]==cell['fold']
                forbidden=set(meta['groups'][meta['folds']==cell['fold']])
                assert not forbidden & (set(cell['readout']['validation_groups'])|set(cell['readout']['training_groups']))
                assert cell['conf_segment'].sum()==ev.sum()
                assert cell['conf_duration'].sum()==cell['duration'][ev].sum()
                C=len(meta['class_names']);n=len(meta['names']);index=(cell['rec'][ev]*C+cell['truth'][ev])*C+cell['predictions']
                for key,w in [('segment',None),('duration',cell['duration'][ev])]:
                    independent=np.bincount(index,weights=w,minlength=n*C*C).reshape(n,C,C)
                    np.testing.assert_array_equal(independent,cell['conf_'+key])
                assert not set(cell['readout']['validation_groups'])&set(cell['readout']['training_groups'])
                assert np.isfinite(cell['centers']).all() and len(cell['centers'])==cell['K']
                assert cell['codes'].min()>=0 and cell['codes'].max()<cell['K']
                assert cell['discrete_payload_sha256']==hashlib.sha256(cell['centers'].tobytes()+cell['codes'].tobytes()+cell['retrieval']['S'].tobytes()).hexdigest()
                if cell['n_iter']>=300:report['issues'].append(dict(file=p.name,note='KMeans reached its maximum iteration count; convergence is not established.'))
                if cell['kind']=='oracle':
                    evrec=np.unique(cell['rec'][ev]);assert cell['transitions'].sum()==ev.sum()-len(evrec)
            else:
                meta=load(OUT/'exp2'/f"{cell['dataset']}_meta.pkl")
                reference=load(folder/f"{cell['dataset']}_f{cell['fold']}_k0_s0_continuous.pkl")
                for field in ('rec','truth','purity'):
                    np.testing.assert_array_equal(cell[field],reference[field])
                assert cell['eligible_windows']==reference['eligible_windows']
                assert cell['total_windows']==reference['total_windows']
                forbidden=set(meta['groups'][meta['folds']==cell['fold']])
                assert not forbidden & (set(cell['validation_groups'])|set(cell['training_groups']))
                assert cell['conf'].sum()==len(cell['truth'])
                C=len(meta['class_names']);n=len(meta['names']);index=(cell['rec']*C+cell['truth'])*C+cell['predictions'][0]
                independent=np.bincount(index,minlength=n*C*C).reshape(n,C,C)
                np.testing.assert_array_equal(independent,cell['conf'])
                assert not set(cell['validation_groups'])&set(cell['training_groups'])
    report['complete']=all(v['completed']==v['expected'] for v in report['experiments'].values())
    sensitivities=[]
    for path in sorted((OUT/'convergence_sensitivity').glob('*.pkl')):
        alternative=load(path);cell=load(OUT/'exp3'/path.name);info=alternative['summary']
        assert info['independent_objective']<=info['original_objective']+1e-8
        meta=load(OUT/'exp2'/f"{cell['dataset']}_meta.pkl");C=len(meta['class_names']);n=len(meta['names'])
        index=(cell['rec']*C+cell['truth'])*C+alternative['predictions']
        np.testing.assert_array_equal(np.bincount(index,minlength=n*C*C).reshape(n,C,C),alternative['conf'])
        assert np.isclose(info['prediction_agreement'],np.mean(alternative['predictions']==cell['predictions'][0]))
        sensitivities.append(info)
    report['convergence_sensitivity']=dict(refits=len(sensitivities),
                                         gradient_below_1e_8=sum(s['independent_gradient']<1e-8 for s in sensitivities))
    csvout('SOLVER_AUDIT.csv',solver_rows)
    report['linear_solver']=dict(refits=len(solver_rows),at_iteration_cap=sum(r['iterations']>=D.LIN_ITERS for r in solver_rows),
                                maximum_gradient=max([r['gradient_max'] for r in solver_rows],default=0),
                                tuning_fits_at_cap=sum(r['tuning_fits_at_cap'] for r in solver_rows))
    (OUT/'VERIFICATION.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--complete',action='store_true');p.add_argument('--artifacts-only',action='store_true');p.add_argument('--check-numerics',action='store_true');a=p.parse_args()
    if not a.artifacts_only:
        numerical_checks();manifest()
    else:
        if a.check_numerics:numerical_checks()
        previous=json.loads((OUT/'MANIFEST.json').read_text(encoding='utf-8'))
        assert all('frame_file_sha256' in d for d in previous['datasets'].values())
        previous['source_sha256']={str(p.relative_to(ROOT)):sha(p) for folder in ('script/vocab','script/seg') for p in sorted((ROOT/folder).glob('*.py'))}
        (OUT/'MANIFEST.json').write_text(json.dumps(previous,indent=2),encoding='utf-8')
    audit(a.complete)
