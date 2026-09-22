"""Numerical sensitivity of capped selected prototype-context fits.

Fixed selected hyperparameters; no re-selection and no replacement of primary
predictions. An independent float64 L-BFGS-B implementation checks the same
class-balanced softmax + L2 objective. Outer predictions are compared only after
optimization terminates using training-objective criteria.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('MKL_NUM_THREADS','4')
import sys,time,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from scipy.optimize import minimize,OptimizeResult
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
from script.vocab.common import OUT
from script.vocab.analyze import load,csvout,readout_stats,interval,S
from script.vocab.run_exp1 import save
from script.vocab import readout as R,linear_projection as LP
torch.set_num_threads(4)

def independent_fit(X,y,C,l2,warm=None):
    X=np.asarray(X,dtype=np.float64);d=X.shape[1]
    cw=R._class_weights(y,C);weight=cw[y]/cw[y].sum()
    def objective(theta):
        pars=theta.reshape(C,d+1);W=pars[:,:d];b=pars[:,d]
        logits=X@W.T+b;logz=logsumexp(logits,axis=1)
        loss=np.sum(weight*(logz-logits[np.arange(len(y)),y]))+.5*l2*np.sum(W*W)
        residual=np.exp(logits-logz[:,None]);residual[np.arange(len(y)),y]-=1
        residual*=weight[:,None]
        gradient=np.column_stack((residual.T@X+l2*W,residual.sum(0)))
        return float(loss),gradient.ravel()
    initial=np.zeros(C*(d+1))
    # Independent central differences check the analytic gradient before fitting.
    rng=np.random.default_rng(717);probe=rng.normal(scale=.01,size=initial.shape)
    _,gradient=objective(probe)
    for j in rng.choice(len(probe),size=min(8,len(probe)),replace=False):
        plus=probe.copy();minus=probe.copy();plus[j]+=1e-5;minus[j]-=1e-5
        numerical=(objective(plus)[0]-objective(minus)[0])/2e-5
        np.testing.assert_allclose(numerical,gradient[j],atol=1e-7,rtol=1e-5)
    if warm is None:
        result=minimize(objective,initial,jac=True,method='L-BFGS-B',
                        options=dict(maxiter=10000,maxcor=50,gtol=1e-8,ftol=1e-13,maxls=40))
    else:
        theta=warm['parameters'].ravel();loss,gradient=objective(theta);old=warm['summary']
        np.testing.assert_allclose(loss,old['independent_objective'],atol=1e-10,rtol=0)
        result=OptimizeResult(x=theta,fun=loss,jac=gradient,nit=old['independent_iterations'],
                              success=old['success'],message=old['termination'])
    result.lbfgs_objective=float(result.fun);result.lbfgs_gradient=float(np.abs(result.jac).max())
    result.newton_steps=0
    # Small coarse-vocabulary systems allow an exact Hessian check/refinement.
    # Fix the last intercept to remove softmax's common-intercept null direction.
    if len(initial)-1<=2500:
        augmented=np.column_stack((X,np.ones(len(X))))
        for iteration in range(40):
            loss,gradient=objective(result.x)
            if np.abs(gradient).max()<1e-8:break
            pars=result.x.reshape(C,d+1);logits=augmented@pars.T
            prob=np.exp(logits-logsumexp(logits,axis=1)[:,None])
            H=np.zeros((len(initial),len(initial)))
            for a in range(C):
                for b in range(a,C):
                    factor=weight*prob[:,a]*((1. if a==b else 0.)-prob[:,b])
                    block=augmented.T@(augmented*factor[:,None])
                    if a==b:block[np.arange(d),np.arange(d)]+=l2
                    ia=slice(a*(d+1),(a+1)*(d+1));ib=slice(b*(d+1),(b+1)*(d+1))
                    H[ia,ib]=block;H[ib,ia]=block.T
            if iteration==0:
                for j in rng.choice(len(initial)-1,size=3,replace=False):
                    plus=result.x.copy();minus=result.x.copy();plus[j]+=1e-5;minus[j]-=1e-5
                    numerical=(objective(plus)[1]-objective(minus)[1])/2e-5
                    np.testing.assert_allclose(numerical,H[:,j],atol=1e-7,rtol=1e-5)
            direction=np.zeros_like(initial)
            direction[:-1]=np.linalg.solve(H[:-1,:-1],gradient[:-1])
            descent=float(gradient@direction);assert descent>=-1e-12
            step=1.
            for _ in range(40):
                candidate=result.x-step*direction;candidate_loss,_=objective(candidate)
                if candidate_loss<=loss-1e-4*step*descent+1e-14:break
                step*=.5
            else:break
            result.x=candidate;result.newton_steps+=1
        result.fun,result.jac=objective(result.x)
        result.success=bool(np.abs(result.jac).max()<1e-8)
        result.message='Exact-Hessian refinement: '+('gradient tolerance satisfied' if result.success else 'stopping limit reached')
    return result,result.x.reshape(C,d+1)

def check(path,warm=None):
    cell=load(path);ds=cell['dataset'];f=cell['fold'];mult=cell['mult'];seed=cell['seed'];tag=cell['input']
    assert tag in ('ordered','unordered'),(path.name,'This sensitivity handles prototype inputs only.')
    base=load(OUT/'exp2'/f'{ds}_fixed_eight_f{f}_k{mult}_s{seed}.pkl')
    meta=load(OUT/'exp2'/f'{ds}_meta.pkl');rec=base['rec'];proto=base['centers'];C=len(meta['class_names'])
    center=np.concatenate([v[2:-2] for v in np.split(np.arange(len(rec)),np.flatnonzero(np.diff(rec))+1)])
    neighbors=center[:,None]+np.arange(-2,3);codes=base['codes'][neighbors]
    fit=meta['folds'][rec[center]]!=f;y=base['truth'][center]
    X=proto[codes].reshape(len(center),-1) if tag=='ordered' else proto.astype(np.float64)[codes].mean(1)
    mu,sd=R.standardise_fit(X[fit]);blocks=[proto]*5 if tag=='ordered' else [proto]
    projectors=LP.make_projectors(blocks,mu,sd)
    xt=LP.project(X[fit],mu,sd,projectors);xe=LP.project(X[~fit],mu,sd,projectors);del X
    np.testing.assert_array_equal(y[~fit],cell['truth'])
    started=time.time();l2=cell['selected']['config']
    with threadpool_limits(limits=4):result,parameters=independent_fit(xt,y[fit],C,l2,warm)
    pred=(xe@parameters[:,:-1].T+parameters[:,-1]).argmax(1)
    conf=R.confusion_by_recording(y[~fit],pred,rec[center][~fit],len(meta['names']),C)
    oldba,oldf1=R.bal_acc_f1(cell['conf'].sum(0));ba,f1=R.bal_acc_f1(conf.sum(0))
    row=dict(file=path.name,dataset=ds,fold=f,K_multiple=mult,seed=seed,input=tag,l2=l2,
             original_iterations=cell['optimization'][0]['iterations'],original_objective=cell['optimization'][0]['training_objective'],
             original_gradient=cell['optimization'][0]['gradient_max'],independent_iterations=int(result.nit),
             lbfgs_objective=result.lbfgs_objective,lbfgs_gradient=result.lbfgs_gradient,newton_steps=result.newton_steps,
             independent_objective=float(result.fun),independent_gradient=float(np.abs(result.jac).max()),
             success=bool(result.success),termination=str(result.message),
             prediction_agreement=float((pred==cell['predictions'][0]).mean()),
             fold_BA_difference=ba-oldba,fold_macro_F1_difference=f1-oldf1,seconds=time.time()-started)
    payload=dict(summary=row,conf=conf,predictions=pred,parameters=parameters)
    if warm is not None:payload['previous_stage']=warm
    save(OUT/'convergence_sensitivity'/path.name,payload)
    print(json.dumps(row),flush=True)

def summarize():
    alternatives=[load(p) for p in sorted((OUT/'convergence_sensitivity').glob('*.pkl'))]
    conditions=sorted({(a['summary']['dataset'],a['summary']['K_multiple'],a['summary']['input']) for a in alternatives})
    rows=[]
    for ds,mult,tag in conditions:
        nf=4 if ds=='lara' else 5
        paths=[[OUT/'exp3'/f'{ds}_f{f}_k{mult}_s{s}_{tag}.pkl' for f in range(nf)] for s in range(3)]
        if not all(p.exists() for member in paths for p in member):continue
        original=np.stack([sum(load(p)['conf'] for p in member) for member in paths]);refined=original.copy()
        for alternative in alternatives:
            info=alternative['summary']
            if (info['dataset'],info['K_multiple'],info['input'])!=(ds,mult,tag):continue
            refined[info['seed']]+=alternative['conf']-load(OUT/'exp3'/info['file'])['conf']
        meta=load(OUT/'exp2'/f'{ds}_meta.pkl');weights=S.make_weights(meta['groups'])
        old=readout_stats(original,weights);new=readout_stats(refined,weights)
        for metric in old:
            rows.append(dict(dataset=ds,K_multiple=mult,input=tag,metric=metric,
                             published_value=old[metric][0],refined_value=new[metric][0],
                             **interval(new[metric]-old[metric])))
    csvout('CONVERGENCE_SENSITIVITY_AGGREGATE.csv',rows)
    # Recompute every paired context contrast with converged replacements, using
    # identical group bootstrap draws. Hyperparameter choices remain untouched.
    lookup={a['summary']['file']:a['conf'] for a in alternatives};paired=[]
    for ds,nf in [('lara',4),('babel1',5)]:
        meta=load(OUT/'exp2'/f'{ds}_meta.pkl');weights=S.make_weights(meta['groups']);stats={}
        for mult in (0,1,16):
            for tag in (('continuous',) if mult==0 else ('center','histogram','unordered','ordered')):
                seeds=[0] if mult==0 else range(3)
                files=[[OUT/'exp3'/f'{ds}_f{f}_k{mult}_s{s}_{tag}.pkl' for f in range(nf)] for s in seeds]
                if not all(p.exists() for member in files for p in member):continue
                original=[];refined=[]
                for member in files:
                    cells=[load(p)['conf'] for p in member]
                    original.append(sum(cells))
                    refined.append(sum(lookup.get(p.name,c) for p,c in zip(member,cells)))
                stats[mult,tag]=(readout_stats(np.stack(original),weights),readout_stats(np.stack(refined),weights))
        for mult in (1,16):
            for a,b in [('ordered','unordered'),('ordered','center'),('unordered','center'),('histogram','center'),('continuous','ordered')]:
                ka=(0 if a=='continuous' else mult,a);kb=(mult,b)
                if ka not in stats or kb not in stats:continue
                olda,newa=stats[ka];oldb,newb=stats[kb]
                for metric in olda:
                    published=olda[metric][0]-oldb[metric][0];difference=newa[metric]-newb[metric]
                    paired.append(dict(dataset=ds,K_multiple=mult,comparison=a+'_minus_'+b,metric=metric,
                                       published_value=published,change_from_published=float(difference[0]-published),
                                       **interval(difference)))
    csvout('CONVERGENCE_SENSITIVITY_PAIRED.csv',paired)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--refine',action='store_true');args=parser.parse_args()
    dest=OUT/'convergence_sensitivity';dest.mkdir(exist_ok=True)
    for path in sorted((OUT/'exp3').glob('*.pkl')):
        cell=load(path)
        if cell['optimization'][0]['iterations']>=R.LIN_ITERS:
            saved=dest/path.name
            if not saved.exists():check(path)
            elif args.refine:check(path,load(saved))
    csvout('CONVERGENCE_SENSITIVITY.csv',[load(p)['summary'] for p in sorted(dest.glob('*.pkl'))])
    summarize()
