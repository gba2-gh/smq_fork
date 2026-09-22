"""Paired group-bootstrap summaries and exportable confusion matrices."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('MKL_NUM_THREADS','4')
import sys,pickle,csv,json,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from script.vocab.common import OUT,FIG
from script.vocab import readout as R
from script.seg import stats as S

def load(p):
    with open(p,'rb') as f:return pickle.load(f)

def csvout(name,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with open(OUT/name,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)

def interval(x):
    point,lo,hi=S.ci(x)
    return dict(value=point,lo=lo,hi=hi)

def confusion_rows(meta,conf,condition):
    return [dict(**condition,true=a,predicted=b,count=conf[i,j]) for i,a in enumerate(meta['class_names']) for j,b in enumerate(meta['class_names'])]

def readout_stats(confs,W):
    # Seeds are averaged after computing the nonlinear metric, not treated as independent groups.
    stats=[R.metrics_from_weighted(c,W) for c in confs]
    return {'balanced_accuracy':np.mean([s[0] for s in stats],axis=0),
            'macro_f1':np.mean([s[1] for s in stats],axis=0)}

def exp1():
    rows=[];paired=[];confrows=[];recalls=[];tuning=[];seedrows=[]
    for ds,nf in (('lara',4),('babel1',5)):
        meta=load(OUT/'exp1'/f'{ds}_meta.pkl');W=S.make_weights(meta['groups']);stats={}
        for rep in ('mean','eight'):
            for family in ('linear','mlp'):
                files=[OUT/'exp1'/f'{ds}_{rep}_f{f}_{family}.pkl' for f in range(nf)]
                if not all(p.exists() for p in files):continue
                cells=[load(p) for p in files]
                for f,cell in enumerate(cells):
                    tuning.append(dict(dataset=ds,representation=rep,family=family,fold=f,parameters=cell['parameters'],
                                       selected=json.dumps(cell['selected']),validation_groups=json.dumps(cell['validation_groups'].tolist())))
                for weight in ('segment','duration'):
                    conf=sum(c[weight] for c in cells)
                    stats[rep,family,weight]=readout_stats(conf,W)
                    cond=dict(dataset=ds,representation=rep,family=family,weighting=weight)
                    for metric,x in stats[rep,family,weight].items():rows.append(dict(**cond,metric=metric,**interval(x)))
                    confrows+=confusion_rows(meta,conf.mean(0).sum(0),cond)
                    for seed,c in enumerate(conf):
                        ba,f1=R.bal_acc_f1(c.sum(0));seedrows.append(dict(**cond,seed=seed,balanced_accuracy=ba,macro_f1=f1))
                    recalls += [dict(**cond,action=label,recall=value) for label,value in zip(meta['class_names'],R.per_class_recall(conf.mean(0).sum(0)))]
                    if rep=='mean' and family=='linear':
                        for baseline in ('majority','prior','uniform'):
                            bc=sum(c['baseline_'+weight][baseline] for c in cells)
                            bs=readout_stats(bc[None],W)
                            for metric,x in bs.items():rows.append(dict(dataset=ds,representation='baseline',family=baseline,weighting=weight,metric=metric,**interval(x)))
        for weight in ('segment','duration'):
            comparisons=[(('eight',fam,weight),('mean',fam,weight),'eight_minus_mean_'+fam) for fam in ('linear','mlp')]
            comparisons += [((rep,'mlp',weight),(rep,'linear',weight),'mlp_minus_linear_'+rep) for rep in ('mean','eight')]
            for a,b,name in comparisons:
                if a in stats and b in stats:
                    for metric in stats[a]:paired.append(dict(dataset=ds,comparison=name,weighting=weight,metric=metric,**interval(stats[a][metric]-stats[b][metric])))
    for name,x in [('exp1_summary.csv',rows),('exp1_paired.csv',paired),('exp1_confusions.csv',confrows),('exp1_recalls.csv',recalls),('exp1_tuning.csv',tuning),('exp1_seeds.csv',seedrows)]:csvout(name,x)
    print('Exp1 summary rows',len(rows),flush=True)

def retrieval_stats(arrays,W,classes):
    arr={k:np.stack([a[k] for a in arrays],axis=1) for k in ('S','n','ch')}
    return S.cb_retrieval(W,{f'r_{k}':v for k,v in arr.items()},'r',classes)

def exp2():
    rows=[];seedrows=[];foldrows=[];paired=[];transrows=[];confrows=[];reuse=[];draws={}
    for ds,nf in (('lara',4),('babel1',5)):
        meta=load(OUT/'exp2'/f'{ds}_meta.pkl');W=S.make_weights(meta['groups']);n=len(meta['names'])
        dsstats={};old=load(OUT.parent/'seg'/'raw'/f'exp1_{ds}.pkl')
        assert meta['anchor_hash']==old['meta']['checks']['anchors_sha256']
        assert np.array_equal(meta['folds'],old['meta']['fold_of_rec'])
        assert meta['names']==old['meta']['names']
        assert np.array_equal(meta['groups'],old['meta']['group'])
        reference=old['results']['fixed'][0]['ca_q_n'];classes=np.flatnonzero(reference.sum(0)>=20)
        for rep in ('mean','eight'):
            for kind in ('fixed','oracle'):
                continuous=[]
                for f in range(nf):
                    p=OUT/'exp2'/f'{ds}_{kind}_{rep}_f{f}_continuous.pkl'
                    if p.exists():continuous.append(load(p)['retrieval'])
                if len(continuous)!=nf:continue
                cont={k:sum(a[k] for a in continuous) for k in continuous[0]}
                np.testing.assert_array_equal(cont['n'],reference)
                if rep=='eight':
                    previous=old['results'][kind][0]
                    original={k:previous['ca_cont_'+k] for k in ('S','n','ch')}
                    oldp=retrieval_stats([original],W[:1],classes)['p'][0,0]
                    newp=retrieval_stats([cont],W[:1],classes)['p'][0,0]
                    discrepancy=float(np.abs(cont['S']-original['S']).max())
                    matches=all(np.allclose(cont[k],original[k],rtol=0,atol=1e-9) for k in ('S','n','ch'))
                    reuse.append(dict(dataset=ds,boundaries=kind,mode='continuous',seed=0,old_p10=oldp,new_p10=newp,difference=newp-oldp,
                                      max_record_class_score_sum_difference=discrepancy,source='historical_verified' if matches else 'new_tie_exact_evaluation'))
                    if matches:cont.update(original)
                cs=retrieval_stats([cont],W,classes)['p'][:,0]
                dsstats[rep,kind,'continuous']=cs
                rows.append(dict(dataset=ds,representation=rep,boundaries=kind,K_multiple=0,metric='continuous_p10',**interval(cs)))
                for mult in (1,4,16):
                    files=[OUT/'exp2'/f'{ds}_{kind}_{rep}_f{f}_k{mult}_s{s}.pkl' for s in (0,1,2) for f in range(nf)]
                    if not all(p.exists() for p in files):continue
                    # Only one condition in memory; retained arrays are small.
                    cells=[[load(OUT/'exp2'/f'{ds}_{kind}_{rep}_f{f}_k{mult}_s{s}.pkl') for f in range(nf)] for s in (0,1,2)]
                    cond=dict(dataset=ds,representation=rep,boundaries=kind,K_multiple=mult)
                    ras=[{k:sum(c['retrieval'][k] for c in member) for k in member[0]['retrieval']} for member in cells]
                    for a in ras:np.testing.assert_array_equal(a['n'],reference)
                    if mult==1 and rep=='eight':
                        for seed,a in enumerate(ras):
                            prev=old['results'][kind][seed]
                            original=dict(S=prev['ca_q_S'],n=prev['ca_q_n'],ch=prev['ca_q_ch'])
                            oldp=retrieval_stats([original],W[:1],classes)['p'][0,0]
                            newp=retrieval_stats([a],W[:1],classes)['p'][0,0]
                            discrepancy=float(np.abs(a['S']-original['S']).max())
                            matches=all(np.allclose(a[k],original[k],rtol=0,atol=1e-9) for k in ('S','n','ch'))
                            reuse.append(dict(dataset=ds,boundaries=kind,mode='prototype',seed=seed,old_p10=oldp,new_p10=newp,difference=newp-oldp,
                                              max_record_class_score_sum_difference=discrepancy,
                                              source='historical_verified' if matches else 'new_tie_exact_evaluation'))
                            if matches:
                                # Reuse the historical K=C endpoint after validating compatibility;
                                # refitted centers are still needed for the newly requested diagnostics.
                                a.update(original)
                    ret=retrieval_stats(ras,W,classes);q=ret['p'].mean(1);dsstats[rep,kind,mult]=q
                    rows.append(dict(**cond,metric='prototype_p10',**interval(q)))
                    rows.append(dict(**cond,metric='retrieval_chance',**interval(ret['chance'].mean(1))))
                    rows.append(dict(**cond,metric='continuous_minus_prototype',**interval(cs-q)))
                    sat=np.stack([a['saturated'].sum(1) for a in ras],1);den=reference.sum(1)
                    saturation=((W@sat)/(W@den)[:,None]).mean(1)
                    rows.append(dict(**cond,metric='same_code_at_least_10',**interval(saturation)))
                    for weighting in ('segment','duration'):
                        cf=np.stack([sum(c['conf_'+weighting] for c in member) for member in cells])
                        st=readout_stats(cf,W)
                        for metric,x in st.items():rows.append(dict(**cond,metric='code_'+weighting+'_'+metric,**interval(x)))
                        confrows+=confusion_rows(meta,cf.mean(0).sum(0),dict(**cond,weighting=weighting))
                        for seed,member in enumerate(cells):
                            ba,f1=R.bal_acc_f1(cf[seed].sum(0));total=cf[seed].sum(0)
                            metrics=dict(balanced_accuracy=ba,macro_f1=f1,predicted_classes=int((total.sum(0)>0).sum()),recalled_classes=int((np.diag(total)>0).sum()),
                                         prototype_p10=ret['p'][0,seed],same_code_at_least_10=sat[:,seed].sum()/den.sum())
                            for field in ('predicted','recalled'):
                                coverage=[c['class_coverage'][field] for c in member]
                                metrics['mean_fold_'+field+'_classes']=float(np.mean(coverage))
                                metrics['min_fold_'+field+'_classes']=int(min(coverage))
                                metrics['max_fold_'+field+'_classes']=int(max(coverage))
                            if kind=='oracle':
                                transition=sum(c['transitions'] for c in member).sum(0)
                                support=transition.sum(-1);ok=support>0
                                metrics['transition_retention']=float(transition[...,1].sum()/support.sum())
                                metrics['macro_transition_retention']=float((transition[...,1][ok]/support[ok]).mean())
                            cm=[c['cluster_'+weighting] for c in member]
                            for key in cm[0]:
                                if key!='contingency':metrics[key]=float(np.mean([m[key] for m in cm]))
                            seedrows.append(dict(**cond,weighting=weighting,seed=seed,**metrics))
                            for f,c in enumerate(member):
                                foldrows.append(dict(**cond,weighting=weighting,seed=seed,fold=f,fit_n=c['fit_n'],eval_n=c['eval_n'],
                                                     fit_occupied=c['fit_occupied'],n_iter=c['n_iter'],seconds=c['seconds'],
                                                     predicted_classes=c['class_coverage']['predicted'],recalled_classes=c['class_coverage']['recalled'],
                                                     readout_l2=c['readout']['selected']['config'],readout_inner_BA=c['readout']['selected']['ba'],
                                                     **{k:v for k,v in c['cluster_'+weighting].items() if k!='contingency'}))
                    if kind=='oracle':
                        ts=np.stack([sum(c['transitions'] for c in member) for member in cells])
                        all_draws=[];macro_draws=[]
                        for seed,t in enumerate(ts):
                            tb=np.einsum('br,rijc->bijc',W,t);denom=tb.sum(-1);nt=tb[...,1]
                            valid=denom[0]>0
                            all_draws.append(nt.sum((1,2))/denom.sum((1,2)))
                            with np.errstate(invalid='ignore',divide='ignore'):macro_draws.append(np.nanmean(np.where(valid,nt/denom,np.nan),axis=(1,2)))
                            for i,a in enumerate(meta['class_names']):
                                for j,b in enumerate(meta['class_names']):
                                    if i!=j:transrows.append(dict(**cond,seed=seed,from_action=a,to_action=b,collapsed=t[:,i,j,0].sum(),retained=t[:,i,j,1].sum()))
                        for metric,x in [('transition_retention',np.mean(all_draws,0)),('macro_transition_retention',np.mean(macro_draws,0))]:rows.append(dict(**cond,metric=metric,**interval(x)))
                    del cells
            if all((rep,k,m) in dsstats for k in ('fixed','oracle') for m in (1,4,16)):
                for mult in (1,4,16):
                    paired.append(dict(dataset=ds,representation=rep,comparison=f'oracle_minus_fixed_K{mult}',**interval(dsstats[rep,'oracle',mult]-dsstats[rep,'fixed',mult])))
                interaction=(dsstats[rep,'oracle',16]-dsstats[rep,'fixed',16])-(dsstats[rep,'oracle',1]-dsstats[rep,'fixed',1])
                paired.append(dict(dataset=ds,representation=rep,comparison='primary_interaction',**interval(interaction)))
                for kind in ('fixed','oracle'):
                    paired.append(dict(dataset=ds,representation=rep,comparison=kind+'_K16_minus_K1',**interval(dsstats[rep,kind,16]-dsstats[rep,kind,1])))
        draws[ds]=dsstats
    for name,x in [('exp2_summary.csv',rows),('exp2_seeds.csv',seedrows),('exp2_folds.csv',foldrows),('exp2_paired.csv',paired),('exp2_transitions.csv',transrows),('exp2_confusions.csv',confrows),('reuse_audit.csv',reuse)]:csvout(name,x)
    with open(OUT/'retrieval_draws.pkl','wb') as f:pickle.dump(draws,f)
    print('Exp2 summary rows',len(rows),flush=True)

def exp3():
    rows=[];paired=[];params=[];confrows=[];seedrows=[];population={}
    for ds,nf in (('lara',4),('babel1',5)):
        base=load(OUT/'exp2'/f'{ds}_fixed_eight_f0_k1_s0.pkl')
        rec=base['rec'];y=base['truth']
        recordings=np.split(np.arange(len(rec)),np.flatnonzero(np.diff(rec))+1)
        centers=np.concatenate([indices[2:-2] for indices in recordings])
        population[ds]=dict(eligible_windows=len(centers),total_windows=len(rec),
                            neighbor_label_agreement=float((y[centers[:,None]+np.array([-2,-1,1,2])]==y[centers,None]).mean()),
                            five_windows_same_label=float((y[centers[:,None]+np.arange(-2,3)]==y[centers,None]).all(1).mean()))
        meta=load(OUT/'exp2'/f'{ds}_meta.pkl');W=S.make_weights(meta['groups']);stats={}
        for mult in (0,1,16):
            for tag in (('continuous',) if mult==0 else ('center','histogram','unordered','ordered')):
                seeds=(0,) if mult==0 else (0,1,2)
                files=[OUT/'exp3'/f'{ds}_f{f}_k{mult}_s{s}_{tag}.pkl' for s in seeds for f in range(nf)]
                if not all(p.exists() for p in files):continue
                cells=[[load(OUT/'exp3'/f'{ds}_f{f}_k{mult}_s{s}_{tag}.pkl') for f in range(nf)] for s in seeds]
                conf=np.stack([sum(c['conf'] for c in member) for member in cells]);st=readout_stats(conf,W);stats[mult,tag]=st
                cond=dict(dataset=ds,K_multiple=mult,input=tag)
                for seed,c in zip(seeds,conf):
                    ba,f1=R.bal_acc_f1(c.sum(0));seedrows.append(dict(**cond,seed=seed,balanced_accuracy=ba,macro_f1=f1))
                for metric,x in st.items():rows.append(dict(**cond,metric=metric,**interval(x)))
                confrows+=confusion_rows(meta,conf.mean(0).sum(0),cond)
                for seed,member in zip(seeds,cells):
                    for f,c in enumerate(member):params.append(dict(**cond,seed=seed,fold=f,dimension=c['dimension'],parameters=c['parameters'],
                                                                   selected=json.dumps(c['selected']),n_eval=len(c['truth']),mean_purity=float(c['purity'].mean()),
                                                                   solver_dimension=c.get('solver_dimension',c['dimension']),optimizer_iterations=c['optimization'][0]['iterations'],
                                                                   solver_parameters=c.get('solver_parameters',c['parameters']),
                                                                   optimizer_gradient_max=c['optimization'][0]['gradient_max'],
                                                                   eligible_windows=c['eligible_windows'],total_windows=c['total_windows']))
        for mult in (1,16):
            for a,b in [('ordered','unordered'),('ordered','center'),('unordered','center'),('histogram','center'),('continuous','ordered')]:
                ka=(0 if a=='continuous' else mult,a);kb=(mult,b)
                if ka in stats and kb in stats:
                    for metric in stats[ka]:paired.append(dict(dataset=ds,K_multiple=mult,comparison=a+'_minus_'+b,metric=metric,**interval(stats[ka][metric]-stats[kb][metric])))
    for name,x in [('exp3_summary.csv',rows),('exp3_paired.csv',paired),('exp3_parameters.csv',params),('exp3_confusions.csv',confrows),('exp3_seeds.csv',seedrows)]:csvout(name,x)
    (OUT/'CONTEXT_POPULATION.json').write_text(json.dumps(population,indent=2),encoding='utf-8')
    print('Exp3 summary rows',len(rows),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--experiment',choices=['1','2','3','all'],default='all');a=p.parse_args()
    for num,fn in [('1',exp1),('2',exp2),('3',exp3)]:
        if a.experiment in (num,'all'):fn()
