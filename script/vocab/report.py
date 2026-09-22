"""Render the completed analysis tables and figures; interpretation is separately authored."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('MKL_NUM_THREADS','4')
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from script.vocab.common import OUT,FIG
from script.vocab.analyze import load

def read(name):return pd.read_csv(OUT/name,keep_default_na=False)
def ds_label(ds):return {'lara':'LARa','babel1':'BABEL-1'}[ds]
def ci(row):return f"{100*row.value:.1f} [{100*row.lo:.1f}, {100*row.hi:.1f}]"
def val(df,**filters):
    for k,v in filters.items():df=df[df[k]==v]
    assert len(df)==1,(filters,len(df))
    return df.iloc[0]
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])+'\n'

def figures():
    e1=read('exp1_summary.csv');e2=read('exp2_summary.csv');e3=read('exp3_summary.csv') if (OUT/'exp3_summary.csv').exists() else None
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':130})
    fig,axs=plt.subplots(1,2,figsize=(10,4),sharey=True)
    for ax,ds in zip(axs,('lara','babel1')):
        for shift,fam,color in [(-.15,'linear','#31688e'),(.15,'mlp','#35b779')]:
            rr=[val(e1,dataset=ds,representation=rep,family=fam,weighting='segment',metric='balanced_accuracy') for rep in ('mean','eight')]
            y=np.array([r.value for r in rr]);lo=np.array([r.lo for r in rr]);hi=np.array([r.hi for r in rr]);x=np.arange(2)+shift
            ax.errorbar(x,y*100,yerr=np.stack([y-lo,hi-y])*100,fmt='o',capsize=4,label=fam,color=color)
        ax.axhline(12.5 if ds=='lara' else 20,color='gray',linestyle=':',label='majority BA')
        ax.set(title=ds_label(ds),xticks=[0,1],xticklabels=['Mean','Eight bins'],ylim=(0,85),ylabel='Balanced accuracy (%)')
        ax.legend(frameon=False)
    fig.suptitle('Oracle segment supervised diagnostics — paired group bootstrap intervals');fig.tight_layout();fig.savefig(FIG/'exp1_diagnostics.png');plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(11,7),sharex=True,sharey=True)
    for i,ds in enumerate(('lara','babel1')):
        for j,rep in enumerate(('mean','eight')):
            ax=axs[i,j]
            for kind,color in [('fixed','#31688e'),('oracle','#d95f02')]:
                rr=[val(e2,dataset=ds,representation=rep,boundaries=kind,K_multiple=k,metric='prototype_p10') for k in (1,4,16)]
                y=np.array([r.value for r in rr]);lo=np.array([r.lo for r in rr]);hi=np.array([r.hi for r in rr])
                ax.errorbar([0,1,2],100*y,yerr=np.stack([y-lo,hi-y])*100,label=kind+' prototypes',color=color,marker='o',capsize=3)
                cont=val(e2,dataset=ds,representation=rep,boundaries=kind,K_multiple=0,metric='continuous_p10')
                ax.axhline(100*cont.value,color=color,linestyle='--',label=kind+' continuous')
            chance=val(e2,dataset=ds,representation=rep,boundaries='fixed',K_multiple=1,metric='retrieval_chance')
            ax.axhline(100*chance.value,color='gray',linestyle=':',label='gallery-label chance')
            ax.set(title=f'{ds_label(ds)} / {"eight bins" if rep=="eight" else rep}',xticks=[0,1,2],xticklabels=['C','4C','16C'],ylabel='Class-balanced precision@10 (%)')
    handles,labels=axs[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=5,frameon=False,fontsize=8)
    fig.suptitle('Common anchors: vocabulary size × segment boundaries');fig.tight_layout(rect=(0,.065,1,.96));fig.savefig(FIG/'exp2_retrieval.png');plt.close(fig)
    transitions=read('exp2_transitions.csv')
    conf=read('exp1_confusions.csv')
    fig,axs=plt.subplots(2,2,figsize=(12,10))
    for i,ds in enumerate(('lara','babel1')):
        names=list(dict.fromkeys(conf[conf.dataset==ds].true))
        labels=[s.replace('Handling','H.').replace('wards','') for s in names]
        for j,k in enumerate((1,16)):
            q=transitions[(transitions.dataset==ds)&(transitions.representation=='eight')&(transitions.K_multiple==k)]
            q=q.groupby(['from_action','to_action'])[['collapsed','retained']].mean()
            retained=q.retained.unstack().reindex(index=names,columns=names).to_numpy()
            support=(q.collapsed+q.retained).unstack().reindex(index=names,columns=names).to_numpy()
            with np.errstate(invalid='ignore',divide='ignore'):rate=retained/support
            ax=axs[i,j];im=ax.imshow(rate,vmin=0,vmax=1,cmap='viridis')
            ax.set(xticks=range(len(names)),yticks=range(len(names)),xticklabels=labels,yticklabels=labels,
                   title=f'{ds_label(ds)} eight-bin K={k}C',xlabel='Next action',ylabel='Previous action')
            plt.setp(ax.get_xticklabels(),rotation=45,ha='right')
            for a in range(len(names)):
                for b in range(len(names)):
                    label=f'{100*rate[a,b]:.0f}%\nn={support[a,b]:.0f}' if np.isfinite(rate[a,b]) else '—'
                    ax.text(b,a,label,ha='center',va='center',fontsize=6,color='white' if np.isfinite(rate[a,b]) and rate[a,b]<.55 else 'black')
            fig.colorbar(im,ax=ax,fraction=.046,pad=.04)
    fig.suptitle('Oracle transition retention: seed-mean rates; n counts unique annotated transitions')
    fig.tight_layout();fig.savefig(FIG/'exp2_transitions.png');plt.close(fig)
    if e3 is not None and len(e3)==36:
        fig,axs=plt.subplots(1,2,figsize=(12,4.5),sharey=True)
        tags=['center','histogram','unordered','ordered']
        for ax,ds in zip(axs,('lara','babel1')):
            for shift,k,color in [(-.15,1,'#31688e'),(.15,16,'#d95f02')]:
                rr=[val(e3,dataset=ds,K_multiple=k,input=tag,metric='balanced_accuracy') for tag in tags]
                y=np.array([r.value for r in rr]);lo=np.array([r.lo for r in rr]);hi=np.array([r.hi for r in rr])
                ax.errorbar(np.arange(4)+shift,100*y,yerr=np.stack([y-lo,hi-y])*100,fmt='o',label=f'K={k}C',capsize=3,color=color)
            r=val(e3,dataset=ds,K_multiple=0,input='continuous',metric='balanced_accuracy')
            ax.axhspan(100*r.lo,100*r.hi,color='#35b779',alpha=.10)
            ax.axhline(100*r.value,color='#35b779',linestyle='--',label='continuous context')
            ax.set(title=ds_label(ds),xticks=np.arange(4),xticklabels=['Centre','Histogram','Mean prototype','Ordered'],ylabel='Balanced accuracy (%)')
        handles,labels=axs[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False,fontsize=9)
        fig.suptitle('Fixed-window context diagnostics');fig.tight_layout(rect=(0,.08,1,.96));fig.savefig(FIG/'exp3_context.png');plt.close(fig)
    conf=read('exp1_confusions.csv')
    fig,axs=plt.subplots(2,2,figsize=(12,10))
    for i,ds in enumerate(('lara','babel1')):
        for j,fam in enumerate(('linear','mlp')):
            q=conf[(conf.dataset==ds)&(conf.representation=='eight')&(conf.family==fam)&(conf.weighting=='segment')]
            names=list(dict.fromkeys(q.true));matrix=q.pivot(index='true',columns='predicted',values='count').loc[names,names].to_numpy();matrix/=matrix.sum(1,keepdims=True)
            ax=axs[i,j];im=ax.imshow(matrix,vmin=0,vmax=1,cmap='Blues')
            labels=[s.replace('Handling','H.').replace('wards','') for s in names]
            ax.set(xticks=range(len(names)),yticks=range(len(names)),xticklabels=labels,yticklabels=labels,title=ds_label(ds)+' eight-bin '+fam,xlabel='Predicted',ylabel='True')
            plt.setp(ax.get_xticklabels(),rotation=45,ha='right')
            for a in range(len(names)):
                for b in range(len(names)):ax.text(b,a,f'{matrix[a,b]:.2f}',ha='center',va='center',fontsize=7,color='white' if matrix[a,b]>.55 else 'black')
    fig.suptitle('Oracle segment confusion matrices (rows normalized)');fig.tight_layout();fig.savefig(FIG/'exp1_confusions.png');plt.close(fig)

def report():
    e1=read('exp1_summary.csv');p1=read('exp1_paired.csv');e2=read('exp2_summary.csv');p2=read('exp2_paired.csv');seeds=read('exp2_seeds.csv')
    context_complete=(OUT/'exp3_summary.csv').exists() and len(read('exp3_summary.csv'))==36
    audit=json.loads((OUT/'VERIFICATION.json').read_text(encoding='utf-8'))
    solver=audit['linear_solver']
    counts=', '.join(f"Exp{k}: {v['completed']}/{v['expected']}" for k,v in audit['experiments'].items())
    audit_note=(f"**Artifact verification:** {counts}. Saved predictions independently reconstruct their confusion matrices; inner/outer group separation, code payload hashes, and prototype projection residuals are checked. "
                f"The selected linear refits number {solver['refits']}; {solver['at_iteration_cap']} reached the 2,000-iteration cap. "
                f"Their largest final absolute gradient is {solver['maximum_gradient']:.3g}. "
                f"Across regularization selection, {solver['tuning_fits_at_cap']} candidate fits reached the cap; tuning therefore remains a bounded optimization diagnostic. "
                f"The audit records {len(audit['issues'])} issues in [VERIFICATION.json](VERIFICATION.json); per-fit optimization details are in [SOLVER_AUDIT.csv](SOLVER_AUDIT.csv). "
                "Independent recomputation of fitting-only frame scalers passed for all 324 clustering cells across nine folds ([SCALER_AUDIT.json](SCALER_AUDIT.json)).")
    if (OUT/'CONVERGENCE_SENSITIVITY.csv').exists():
        sensitivity=read('CONVERGENCE_SENSITIVITY.csv')
        audit_note+=(f" **Capped-fit sensitivity:** {len(sensitivity)} selected prototype-context fits were checked at their fixed selected regularization with an independent implementation of the same objective; "
                     f"{int((sensitivity.independent_gradient<1e-8).sum())} reached a maximum gradient below 1e-8. "
                     f"The largest affected-fold changes were {100*sensitivity.fold_BA_difference.abs().max():.4f} BA points and {100*sensitivity.fold_macro_F1_difference.abs().max():.4f} macro-F1 points; minimum prediction agreement was {100*sensitivity.prediction_agreement.min():.4f}%. "
                     "Small systems additionally use an exact-Hessian refinement, with gradients and Hessians checked by finite differences. Primary results retain the original bounded fits. Details: [CONVERGENCE_SENSITIVITY.csv](CONVERGENCE_SENSITIVITY.csv).")
        if (OUT/'CONVERGENCE_SENSITIVITY_AGGREGATE.csv').exists():
            sensitivity_aggregate=read('CONVERGENCE_SENSITIVITY_AGGREGATE.csv')
            audit_note+=f" Replacing the flagged predictions for sensitivity changes any affected pooled seed-mean endpoint by at most {100*sensitivity_aggregate.value.abs().max():.5f} points ([paired aggregate sensitivity](CONVERGENCE_SENSITIVITY_AGGREGATE.csv))."
    title='# Action vocabulary and local context: complete experiment report' if context_complete else '# Action vocabulary: primary results (context experiment pending)'
    lines=[title,'',
           'This separately requested follow-up examines category collapse after [the segmentation study](../../DECISION.md). Experiment numbering is local to this report. It tests vocabulary size, Euclidean clustering geometry, weak segment features, and information accessible through neighboring codes. It concerns action-level representations; it does not test sub-action primitive discovery.','',
           'The design and numerical amendments are in [PROTOCOL.md](PROTOCOL.md), exact environment, source hashes, folds and anchors in [MANIFEST.json](MANIFEST.json), and completion checks in [VERIFICATION.json](VERIFICATION.json). Classification, retrieval and retention scores below are percentages; their differences are percentage points. Homogeneity, completeness and AMI use their usual unit scale; entropies are in nats. Bracketed ranges are paired-group bootstrap 95% intervals unless specified otherwise.','']
    interpretation=OUT/'INTERPRETATION.md'
    if interpretation.exists():lines += [interpretation.read_text(encoding='utf-8'),'']
    inventory=[]
    for ds,nf in [('lara',4),('babel1',5)]:
        durations=np.concatenate([load(OUT/'exp1'/f'{ds}_mean_f{f}_linear.pkl')['item_duration'] for f in range(nf)])
        fixed=load(OUT/'exp2'/f'{ds}_fixed_mean_f0_k1_s0.pkl')
        fps=50 if ds=='lara' else 30
        inventory.append([ds,len(fixed['rec']),len(durations),f'{np.median(durations)/fps:.2f}',f'{durations.sum()/fps/3600:.2f}'])
    lines += ['## Design and implementation assessment','',table(['Dataset','Fixed-window tokens','Oracle intervals','Median oracle duration (s)','Recorded hours'],inventory),'',
      '- **Data and exposure:** LARa: 439 recordings, 16 participants, four participant-separated folds, C=8. BABEL-1: 1,960 recordings, five recording-separated folds, C=5. Its participant identities are unavailable. The released encoder was trained transductively on all recordings; these results are not an inductive encoder evaluation.',
      '- **Representations:** mean latent (352/400 dimensions) and eight temporal bins (2,816/3,200 dimensions). The latter uses the preceding study’s area-average/interpolation rule. Oracle intervals use annotation boundaries and are diagnostic, not deployable segmentation. Latent receptive fields can cross those boundaries.',
      '- **Fitting separation:** frame scaling and k-means see outer fitting recordings only. Every readout tunes regularization on a group-disjoint inner 25% split. Readout feature scalers fit inner training during tuning, then outer training for refit. The code readout conditions on an unsupervised outer-fit codebook that includes inner-validation features. No outer evaluation labels select configurations.',
      '- **Readouts:** class-balanced L2 multinomial logistic regression (five regularization settings; float64 L-BFGS, up to 2,000 iterations, history 50) and one-hidden-layer 128-unit MLP (six decay/dropout settings, up to 60 epochs, patience 10). Both representations receive identical tuning budgets. The selected MLP epoch is refitted with three initialization seeds. These are supervised diagnostics, not unsupervised algorithms or mathematical upper bounds.',
      '- **Numerical correction:** preliminary dense linear fits hit a 150-iteration limit with material gradients. A training-objective-only [convergence check](CONVERGENCE_PILOT.json) motivated the more accurate solver; no outer scores selected the correction. All linear diagnostics were refreshed consistently. The initial readouts are archived in `archive/fp32_150/`. Prototype readouts use orthonormal row-space coordinates internally, preserving logits and the L2 penalty; the input representation and classifier family remain the same. Projection equivalence and optimization diagnostics are audited.',
      '- **Weighting:** main classifier and cluster analyses give each segment equal weight. Duration-weighted classifier results are separate evaluation summaries of the same trained models. Duration-weighted clustering consistency uses each segment’s majority label, which differs from the frame-label mixture for impure fixed windows. Common-anchor retrieval uses actual labels at the fixed timestamps, not majority labels.',
      '- **Clustering:** K=C,4C,16C; seeds 0,1,2; sklearn KMeans n_init=5, max_iter=300. No Hungarian matching or reduction back to C. Original codebooks were not retained, so fitting is required for new assignments, transition metrics and context inputs; matching historical eight-bin K=C retrieval is checked in [reuse_audit.csv](reuse_audit.csv).',
      '- **Retrieval:** identical timestamp populations across all conditions, with galleries restricted to the held-out fold and excluding the query participant (LARa) or recording (BABEL). Precision@10 uses exact expected random tie-breaking across all anchors at the cutoff. A corrected tie implementation was independently checked against 100 explicitly expanded-anchor examples. This addresses the earlier evaluator’s arbitrary cutoff selection between tied items.',
      '- **Uncertainty:** 2,000 paired resamples of participants/recordings, shared across conditions; clustering/MLP seeds are averaged rather than counted as independent samples. Intervals condition on fitted models and fixed retrieval galleries; galleries and models are not rebuilt within each resample. Multiple endpoints are descriptive; no multiplicity-adjusted discovery claim is made.',
      '- **Audit fix:** an initial artifact key collision overwrote per-item duration metadata with the duration confusion matrix. Raw interval durations were restored from annotations and validated; predictions and all summary scores were unaffected.',
      '',audit_note,'',
      '**Historical reuse:** 8 of 12 matching eight-bin K=C prototype endpoints were exactly compatible and reused. The largest discrepancy among the remaining re-evaluations was 0.0234 percentage points; continuous endpoint differences were at most 0.00042 points. New assignments still required refitting because historical codebooks were not saved. Both exact tie handling and numerical/refitting differences can contribute; this audit does not isolate their causes.','',
      '## Experiment 1 — Accessible action information','',
      '**Research question:** can a small supervised diagnostic separate oracle actions from frozen segment features, and does eight-bin representation help?','',
      'Majority-class and class-prior baselines use outer fitting-segment label counts only. The prior baseline is evaluated analytically through its expected confusion matrix, avoiding additional sampling noise. Duration weighting changes evaluation weights, not the fitted baseline or classifier.','',
      '![Supervised diagnostics](../../figures/vocab/exp1_diagnostics.png)','',
      '### Primary segment-weighted scores','']
    rr=[]
    for ds in ('lara','babel1'):
        for rep,fam in [('baseline','majority'),('baseline','prior'),('mean','linear'),('mean','mlp'),('eight','linear'),('eight','mlp')]:
            rr.append([ds,rep,fam,ci(val(e1,dataset=ds,representation=rep,family=fam,weighting='segment',metric='balanced_accuracy')),ci(val(e1,dataset=ds,representation=rep,family=fam,weighting='segment',metric='macro_f1'))])
    lines += [table(['Dataset','Representation','Readout','Balanced accuracy','Macro-F1'],rr),'### Paired representation/readout differences','',table(['Dataset','Contrast','BA difference','Macro-F1 difference'],[[ds,comp,ci(val(p1,dataset=ds,comparison=comp,weighting='segment',metric='balanced_accuracy')),ci(val(p1,dataset=ds,comparison=comp,weighting='segment',metric='macro_f1'))] for ds in ('lara','babel1') for comp in ('eight_minus_mean_linear','eight_minus_mean_mlp','mlp_minus_linear_mean','mlp_minus_linear_eight')]),
      '### Duration-weighted scores (separate endpoint)','',table(['Dataset','Representation','Readout','Balanced accuracy','Macro-F1'],[[ds,rep,fam,ci(val(e1,dataset=ds,representation=rep,family=fam,weighting='duration',metric='balanced_accuracy')),ci(val(e1,dataset=ds,representation=rep,family=fam,weighting='duration',metric='macro_f1'))] for ds in ('lara','babel1') for rep in ('mean','eight') for fam in ('linear','mlp')]),
      '### Class-level performance','', '![Confusion matrices](../../figures/vocab/exp1_confusions.png)','',
      'All four readouts’ per-class recall and complete segment/duration confusion matrices are in [exp1_recalls.csv](exp1_recalls.csv) and [exp1_confusions.csv](exp1_confusions.csv). MLP confusion counts average initialization seeds and can therefore be fractional. Seed-level scores and selected hyperparameters are in [exp1_seeds.csv](exp1_seeds.csv) and [exp1_tuning.csv](exp1_tuning.csv).','',
      '**Analysis:** substantial action information is accessible in both datasets. Eight bins improve LARa BA and macro-F1; BABEL’s eight-bin macro-F1 improvement is clearer than its BA improvement. BABEL benefits from the nonlinear readout. LARa’s eight-bin MLP does not improve BA over linear. This weakens a blanket “features contain no action information” explanation, while leaving class-specific and representation limitations. Eight-bin gains combine temporal detail and higher dimension; they do not isolate temporal order. A negative MLP comparison does not establish absent information.','',
      '## Experiment 2 — Vocabulary size × meaningful boundaries','',
      '**Research question:** does expanding the vocabulary preferentially recover distinctions when segments follow actual action boundaries, or merely increase code differences and fragmentation?','',
      '![Vocabulary retrieval](../../figures/vocab/exp2_retrieval.png)','',
      '### Retrieval and continuous–quantized gap','']
    rr=[]
    for ds in ('lara','babel1'):
        for rep in ('mean','eight'):
            for kind in ('fixed','oracle'):
                filt=dict(dataset=ds,representation=rep,boundaries=kind)
                rr.append([ds,rep,kind,ci(val(e2,**filt,K_multiple=0,metric='continuous_p10'))]+[ci(val(e2,**filt,K_multiple=k,metric='prototype_p10')) for k in (1,4,16)]+[ci(val(e2,**filt,K_multiple=16,metric='continuous_minus_prototype'))])
    lines += [table(['Dataset','Features','Boundaries','Continuous','K=C','K=4C','K=16C','Continuous−16C'],rr),'### Primary paired interaction','',table(['Dataset','Features','Interaction (16C gain in oracle−fixed gap)'],[[ds,rep,ci(val(p2,dataset=ds,representation=rep,comparison='primary_interaction'))] for ds in ('lara','babel1') for rep in ('mean','eight')]),
      'The paired oracle−fixed contrasts at each K and K=16C−C gains within each boundary condition are in [exp2_paired.csv](exp2_paired.csv).','',
      '### Oracle transition preservation and vocabulary occupancy','']
    rr=[]
    for ds in ('lara','babel1'):
        for rep in ('mean','eight'):
            for k in (1,4,16):
                q=seeds[(seeds.dataset==ds)&(seeds.representation==rep)&(seeds.boundaries=='oracle')&(seeds.K_multiple==k)&(seeds.weighting=='segment')]
                filt=dict(dataset=ds,representation=rep,boundaries='oracle',K_multiple=k)
                rr.append([ds,rep,k,ci(val(e2,**filt,metric='transition_retention')),ci(val(e2,**filt,metric='macro_transition_retention')),f'{q.occupied.mean():.1f}',f'{q.effective_K.mean():.1f}'])
    lines += [table(['Dataset','Features','K/C','Overall retention','Macro-type retention','Occupied K','Effective K'],rr),
      '![Transition retention matrices](../../figures/vocab/exp2_transitions.png)','',
      'Transition matrices give retained/collapsed counts for every ordered action-label pair, separately by seed: [exp2_transitions.csv](exp2_transitions.csv). The figure shows the eight-bin endpoint vocabulary sizes; n is the number of unique annotated transitions, not the count multiplied by seeds. Occupancy and effective vocabulary average within-fold quantities; cluster identities are never pooled across codebooks. Higher transition retention alone is insufficient: increasing K makes distinct neighboring codes more likely even without semantic improvement.','',
      '### Label consistency, fragmentation, transfer, and same-code saturation','']
    rr=[]
    for ds in ('lara','babel1'):
        for rep in ('mean','eight'):
            for kind in ('fixed','oracle'):
                for k in (1,4,16):
                    q=seeds[(seeds.dataset==ds)&(seeds.representation==rep)&(seeds.boundaries==kind)&(seeds.K_multiple==k)&(seeds.weighting=='segment')]
                    filt=dict(dataset=ds,representation=rep,boundaries=kind,K_multiple=k)
                    rr.append([ds,rep,kind,k,f'{q.homogeneity.mean():.3f}',f'{q.completeness.mean():.3f}',f'{q.AMI.mean():.3f}',ci(val(e2,**filt,metric='code_segment_balanced_accuracy')),ci(val(e2,**filt,metric='code_segment_macro_f1')),f'{q.recalled_classes.mean():.1f} / {q.mean_fold_recalled_classes.mean():.1f}',f'{100*q.same_code_at_least_10.mean():.1f}'])
    lines += [table(['Dataset','Features','Bounds','K/C','Homogeneity','Completeness','AMI','Code BA','Code macro-F1','Recalled classes (pooled / fold mean)','≥10 same-code (%)'],rr),
      'Code-to-label transfer is supervised evaluation of an unsupervised vocabulary. Class coverage includes both predicted and correctly recalled classes in [exp2_seeds.csv](exp2_seeds.csv); the table shows recalled classes. That file also gives H(label|code), H(code|label) in nats, purity, occupied/effective K and seed-specific results; [exp2_folds.csv](exp2_folds.csv) retains each outer fold. Higher homogeneity/purity must be read alongside fragmentation (lower completeness/higher H(code|label)) and AMI. Same-code saturation is the fraction of all query anchors with at least ten eligible gallery anchors sharing their code; high saturation means retrieval largely measures within-code label composition.','']
    gap=read('oracle_readout_gap.csv')
    lines += ['### Secondary paired readout gap on identical oracle intervals','',
              table(['Dataset','Features','Continuous linear − code linear BA (K=16C)','Macro-F1 difference'],
                    [[ds,rep,ci(val(gap,dataset=ds,representation=rep,K_multiple=16,metric='balanced_accuracy')),ci(val(gap,dataset=ds,representation=rep,K_multiple=16,metric='macro_f1'))] for ds in ('lara','babel1') for rep in ('mean','eight')]),
              'These explanatory comparisons quantify the diagnostic reference after the primary analysis; they were not extra primary endpoints. Both sides predict the same oracle-segment labels, with the same class-balanced linear family and nested regularization grid. Continuous inputs receive fitting-only feature standardization; categorical inputs retain their one-hot scale. The full comparison at every K is in [oracle_readout_gap.csv](oracle_readout_gap.csv).','']
    if (OUT/'exp3_summary.csv').exists() and len(read('exp3_summary.csv'))==36:
        e3=read('exp3_summary.csv');p3=read('exp3_paired.csv');params=read('exp3_parameters.csv')
        population=json.loads((OUT/'CONTEXT_POPULATION.json').read_text(encoding='utf-8'))
        lines += ['## Experiment 3 — Fixed-window local context','',
          '**Research question:** do five-window code contexts expose useful action information beyond a centre code, and does ordered context improve over unordered context? This experiment began only after both primary experiments completed. No oracle boundaries are used.','',
          'Only the prespecified eight-bin representation and K=C,16C are used. Predict the centre window’s existing majority label. Two neighboring tokens on each side are required within the same recording; the existing final partial window is retained as a valid token. All compared inputs use identical eligible centers.','',
          '![Context diagnostics](../../figures/vocab/exp3_context.png)','',
          'Error bars and the continuous-reference bands show 95% group-bootstrap intervals. Paired differences below use the shared bootstrap draws.','']
        rr=[]
        for ds in ('lara','babel1'):
            for k in (1,16):
                for tag in ('center','histogram','unordered','ordered'):
                    q=params[(params.dataset==ds)&(params.K_multiple==k)&(params.input==tag)]
                    rank=str(int(q.solver_dimension.min())) if q.solver_dimension.min()==q.solver_dimension.max() else f'{int(q.solver_dimension.min())}–{int(q.solver_dimension.max())}'
                    count=str(int(q.solver_parameters.min())) if q.solver_parameters.min()==q.solver_parameters.max() else f'{int(q.solver_parameters.min())}–{int(q.solver_parameters.max())}'
                    rr.append([ds,k,tag,int(q.dimension.iloc[0]),int(q.parameters.iloc[0]),rank+' / '+count,ci(val(e3,dataset=ds,K_multiple=k,input=tag,metric='balanced_accuracy')),ci(val(e3,dataset=ds,K_multiple=k,input=tag,metric='macro_f1'))])
            q=params[(params.dataset==ds)&(params.input=='continuous')]
            rr.append([ds,'—','continuous',int(q.dimension.iloc[0]),int(q.parameters.iloc[0]),f'{int(q.solver_dimension.iloc[0])} / {int(q.solver_parameters.iloc[0])}',ci(val(e3,dataset=ds,K_multiple=0,input='continuous',metric='balanced_accuracy')),ci(val(e3,dataset=ds,K_multiple=0,input='continuous',metric='macro_f1'))])
        lines += [table(['Dataset','K/C','Input','Input dimensions','Nominal coefficients','Solver dims / coefficients','BA','Macro-F1'],rr),
          'Centre code and histogram use one-hot/category dimensions. Mean and ordered inputs use fixed-dimensional cluster prototypes; continuous context concatenates the original vectors for those same five windows. All use the same regularized linear family and tuning grid. Ordered and continuous inputs have five times the dimension of mean prototype pooling, so parameter counts differ: the shared family does not fully eliminate capacity as an explanation. Histogram is secondary.','',
          table(['Dataset','K/C','Contrast','BA difference','Macro-F1 difference'],[[ds,k,comp,ci(val(p3,dataset=ds,K_multiple=k,comparison=comp,metric='balanced_accuracy')),ci(val(p3,dataset=ds,K_multiple=k,comparison=comp,metric='macro_f1'))] for ds in ('lara','babel1') for k in (1,16) for comp in ('ordered_minus_unordered','ordered_minus_center','unordered_minus_center','continuous_minus_ordered')]),
          'All histogram contrasts are included in [exp3_paired.csv](exp3_paired.csv). Coverage, selected regularization and nominal parameter counts by fold/seed are in [exp3_parameters.csv](exp3_parameters.csv); seed-level performance is in [exp3_seeds.csv](exp3_seeds.csv). Confusions are in [exp3_confusions.csv](exp3_confusions.csv). Prototype inputs occupy a lower-rank subspace, so nominal parameter counts do not fully describe effective capacity. Unordered pooling retains the within-window representation; the order comparison concerns positions across windows. The linear readouts test additive context, not nonlinear motifs. These endpoints measure action-label accessibility, not hierarchical composition or discovered primitives.','']
        lines += [table(['Dataset','Eligible centre windows','All fixed windows','Retained (%)','Mean centre purity'],
                        [[ds,int(params[params.dataset==ds].eligible_windows.iloc[0]),int(params[params.dataset==ds].total_windows.iloc[0]),
                          f'{100*params[params.dataset==ds].eligible_windows.iloc[0]/params[params.dataset==ds].total_windows.iloc[0]:.1f}',
                          f'{np.average(params[(params.dataset==ds)&(params.input=="continuous")].mean_purity,weights=params[(params.dataset==ds)&(params.input=="continuous")].n_eval):.3f}'] for ds in ('lara','babel1')]),'']
        lines += [f"The eligibility restriction retains {100*population['lara']['eligible_windows']/population['lara']['total_windows']:.1f}% of LARa and {100*population['babel1']['eligible_windows']/population['babel1']['total_windows']:.1f}% of BABEL-1 windows. Context scores should therefore not be compared directly with full-population Exp2 classifier scores. "
                  f"As a descriptive label-persistence check, all five windows have the same majority label in {100*population['lara']['five_windows_same_label']:.1f}% and {100*population['babel1']['five_windows_same_label']:.1f}% of eligible cases, respectively ([CONTEXT_POPULATION.json](CONTEXT_POPULATION.json)). "
                  "Ordered concatenation identifies the center position and allows position-specific weights; gains over a mean can reflect these properties as well as temporal relationships. They do not isolate learned composition, and label persistence remains a possible source of context benefits.",'']
        if (OUT/'CONVERGENCE_SENSITIVITY_AGGREGATE.csv').exists():
            sens=read('CONVERGENCE_SENSITIVITY_AGGREGATE.csv');rr=[]
            for (ds,k,tag),q in sens.groupby(['dataset','K_multiple','input']):
                ba=val(q,metric='balanced_accuracy');f1=val(q,metric='macro_f1')
                rr.append([ds,k,tag,f'{100*ba.published_value:.3f} / {100*ba.refined_value:.3f}',
                           f'{100*f1.published_value:.3f} / {100*f1.refined_value:.3f}',f'{100*ba.value:+.4f}',f'{100*f1.value:+.4f}'])
            lines += ['### Numerical sensitivity of capped selected fits','',
                      table(['Dataset','K/C','Input','Primary / refined BA','Primary / refined macro-F1','BA change','F1 change'],rr),
                      'Only the flagged selected refits are replaced in this sensitivity; their regularization choices are fixed, and the original bounded inner selection remains. Original results above remain the primary estimates. All paired context contrasts are recomputed with the same bootstrap draws in [CONVERGENCE_SENSITIVITY_PAIRED.csv](CONVERGENCE_SENSITIVITY_PAIRED.csv). This checks the numerical stability of the research interpretation without choosing results based on outer scores.','']
    else:
        lines += ['## Experiment 3 — incomplete or not run','',
                  'The context comparison is not complete. Saved cells are retained in `exp3/`; partial summaries, if present, are in `exp3_summary.csv`. No missing condition is treated as a negative result.','']
    decision=OUT/'DECISION.md'
    # Keep all per-class recalls directly reviewable in the report as well as CSV.
    recall=read('exp1_recalls.csv');recall=recall[recall.weighting=='segment']
    lines += ['## Appendix — per-class recall (segment-weighted)','',
              table(['Dataset','Action','Mean linear','Mean MLP','Eight-bin linear','Eight-bin MLP'],
                    [[ds,action]+[f"{100*val(recall,dataset=ds,action=action,representation=rep,family=fam).recall:.1f}" for rep,fam in [('mean','linear'),('mean','mlp'),('eight','linear'),('eight','mlp')]]
                     for ds in ('lara','babel1') for action in recall[recall.dataset==ds].action.unique()]),'']
    lines += ['## Reproduction and artifact index','',
      'Run from the repository root with the existing `smq` environment. Each training script saves completed cells atomically and skips them on resume. Exp3 checks primary completion before fitting. The environment variable below sets a fresh execution deadline for a later reproduction; this run used its original absolute deadline. To refit instead of resume, archive the existing `results/vocab/exp1`, `exp2`, `exp3`, and `convergence_sensitivity` directories first.','',
      '```powershell',
      '$env:SMQ_VOCAB_DEADLINE = [DateTimeOffset]::UtcNow.AddHours(7.2).ToUnixTimeSeconds()',
      "& 'C:/Users/gzaz976/AppData/Local/anaconda3/envs/smq/python.exe' -B script/vocab/run_exp1.py",
      "& 'C:/Users/gzaz976/AppData/Local/anaconda3/envs/smq/python.exe' -B script/vocab/run_exp2.py",
      "& 'C:/Users/gzaz976/AppData/Local/anaconda3/envs/smq/python.exe' -B script/vocab/run_exp3.py",
      "& 'C:/Users/gzaz976/AppData/Local/anaconda3/envs/smq/python.exe' -B script/vocab/analyze.py",
      "& 'C:/Users/gzaz976/AppData/Local/anaconda3/envs/smq/python.exe' -B script/vocab/compare_readouts.py",
      "& 'C:/Users/gzaz976/AppData/Local/anaconda3/envs/smq/python.exe' -B script/vocab/convergence_sensitivity.py",
      "& 'C:/Users/gzaz976/AppData/Local/anaconda3/envs/smq/python.exe' -B script/vocab/verify.py --complete",
      "& 'C:/Users/gzaz976/AppData/Local/anaconda3/envs/smq/python.exe' -B script/vocab/report.py",'```','',
      'Saved fold artifacts in `exp1/`, `exp2/`, `exp3/` contain evaluation predictions/confusions and selected tuning configurations. Exp2 additionally saves codebooks, assignments, scalers and transition counts. Logs include the initial [exp1.log](exp1.log) and [exp2.log](exp2.log), corrected [exp1_refit.log](exp1_refit.log) and [code_readout_refit.log](code_readout_refit.log), per-fold `exp2_<dataset>_f<fold>.log` and `exp3_<dataset>_f<fold>.log`, and the initial [exp3.log](exp3.log). Analysis tables retain more precision than displayed. Historical root reports and model files were not changed by this run.']
    if (OUT/'RUN_SUMMARY.json').exists():
        runtime=json.loads((OUT/'RUN_SUMMARY.json').read_text(encoding='utf-8'))
        lines += ['',f"Execution: {runtime['started_utc']} to {runtime['finished_utc']}; {runtime['elapsed_hours']:.2f} hours, within the eight-hour limit. See [RUN_SUMMARY.json](RUN_SUMMARY.json)."]
    if decision.exists():lines += ['',decision.read_text(encoding='utf-8'),'']
    (OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Report and figures generated',flush=True)

if __name__=='__main__':figures();report()
