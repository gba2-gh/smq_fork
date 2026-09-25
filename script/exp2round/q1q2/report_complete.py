"""Rebuild the Q1/Q2 report from saved measurements; never run experimental fits.

Usage: python script/exp2round/q1q2/report_complete.py
"""
from __future__ import annotations

import hashlib
import itertools
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'results/exp2round/q1q2'
DETAIL = OUT / 'report_details'
KS = [10, 20, 50, 100, 500, 1000]
SEEDS = [111, 222, 1538574472]
WINDOWS = {'hugadb': [60, 30, 15], 'lara': [50, 25, 12]}
CONTEXT = ('HuGaDB: 10 observed classes, 18 subjects, wearable IMU channels. '
           'LARa: 8 classes including None, 16 subjects, optical motion-capture channels. '
           'Cross-dataset comparisons differ in class count, subject count and modality; '
           'HuGaDB cannot establish body-shape effects.')
METRICS = ['MoF', 'Edit', 'F1@10', 'F1@25', 'F1@50']
IM = ['mi_unit_action', 'mi_unit_subject', 'cmi_unit_subject_given_action', 'h_unit',
      'h_action_given_unit', 'h_subject_given_unit', 'nmi_unit_action', 'nmi_unit_subject',
      'ami_unit_action', 'ami_unit_subject']


def table(rows, columns=None):
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows, columns=columns)
    def val(v):
        if isinstance(v, (float, np.floating)):
            return '—' if np.isnan(v) else f'{v:g}'
        return str(v).replace('|', '\\|').replace('\n', ' ')
    return '\n'.join(['| ' + ' | '.join(map(str, df.columns)) + ' |',
                      '| ' + ' | '.join(['---'] * len(df.columns)) + ' |'] +
                     ['| ' + ' | '.join(map(val, row)) + ' |' for row in df.itertuples(index=False, name=None)])


def ms(values, digits=2):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return 'unavailable'
    return f'{a.mean():.{digits}f} ± {a.std(ddof=0):.{digits}f}'


def subset(df, **kwargs):
    mask = np.ones(len(df), dtype=bool)
    for k, v in kwargs.items():
        mask &= df[k].isna() if v is None else df[k].eq(v)
    return df.loc[mask]


def paired(left, right, metric='MoF'):
    a = left[['seed', metric]].merge(right[['seed', metric]], on='seed', validate='one_to_one', suffixes=('_a', '_b'))
    assert set(a.seed) == set(SEEDS)
    return a[metric + '_a'] - a[metric + '_b']


def normalized_keys(df, fields):
    return {tuple(None if pd.isna(v) else v for v in row)
            for row in df[fields].itertuples(index=False, name=None)}


def audit(c, i, s, u):
    ck = ['dataset', 'representation', 'window', 'num_units', 'seed', 'readout', 'temperature_multiplier']
    sk = ['dataset', 'grouping', 'window', 'num_units', 'seed', 'readout', 'temperature_multiplier', 'fold', 'level']
    expected_c, expected_i, expected_s, expected_u = set(), set(), set(), set()
    for ds, ws in WINDOWS.items():
        for w in ws:
            for seed in SEEDS:
                expected_c.add((ds, 'latent', w, None, seed, 'continuous', None))
                for k in KS:
                    expected_i.add((ds, 'latent', w, k, seed))
                    for method in ['hard', 'hard_length_weighted', 'majority_unit_hungarian', 'majority_unit_many_to_one', 'permuted_hard']:
                        expected_c.add((ds, 'latent', w, k, seed, method, None))
                    if w != ws[1]:
                        for t in [.5, 1., 2.]:
                            expected_c.add((ds, 'latent', w, k, seed, 'soft', t))
                        if k in [10, 100, 1000]:
                            expected_c.add((ds, 'latent', w, k, seed, 'within_subject', None))
                            for rep in ['raw', 'per_recording']:
                                expected_c.add((ds, rep, w, k, seed, 'hard', None))
                                expected_i.add((ds, rep, w, k, seed))
        for w, group in itertools.product([ws[0], ws[-1]], ['recording', 'subject']):
            levels = [(f, 'fold') for f in range(4)] + [(None, 'pooled_oof')]
            for f, level in levels:
                expected_s.add((ds, group, w, None, None, 'continuous', None, f, level))
                for k, seed in itertools.product(KS, SEEDS):
                    for method, t in [('hard', None), ('soft', .5), ('soft', 1.), ('soft', 2.)]:
                        expected_s.add((ds, group, w, k, seed, method, t, f, level))
            if group == 'subject':
                for k, seed, fold in itertools.product(KS, SEEDS, range(4)):
                    expected_u.update((ds, group, w, k, seed, fold, unit) for unit in range(k))
    checks = {}
    for name, frame, fields, expected in [
        ('pooled', c, ck, expected_c),
        ('information', i, ck[:5], expected_i),
        ('supervised', s, sk, expected_s),
        ('reuse', u, ['dataset', 'grouping', 'window', 'num_units', 'seed', 'fold', 'unit'], expected_u)]:
        actual = normalized_keys(frame, fields)
        checks[name] = {'rows': len(frame), 'expected': len(expected), 'missing': len(expected-actual),
                        'unexpected': len(actual-expected), 'duplicates': len(frame)-len(actual)}
        assert actual == expected and len(frame) == len(actual), checks[name]
    assert np.all(i.apply(lambda r: sum(json.loads(r.occupancy)) == r.n_windows, axis=1))
    assert np.all(i.apply(lambda r: np.count_nonzero(json.loads(r.occupancy)) == r.occupied_units, axis=1))
    assert u.loc[u.unused, 'subject_distribution_entropy'].isna().all()
    assert u.loc[~u.unused, 'normalized_subject_distribution_entropy'].between(-1e-9, 1+1e-9).all()
    for _, row in s.iterrows():
        cm = np.asarray(json.loads(row.confusion))
        assert cm.sum() == row.n_test_segments
        expected = np.divide(cm.diagonal(), cm.sum(axis=1), out=np.zeros(len(cm), float), where=cm.sum(axis=1)>0)
        assert np.allclose(list(json.loads(row.per_class_recall).values()), expected)
        assert np.isclose(row.balanced_accuracy, 100*expected[cm.sum(axis=1)>0].mean())
    checks['derived_checks'] = ['All semantic grid keys present exactly once', 'Occupancy totals and occupied-unit counts',
                                'Confusion totals, per-class recall and balanced accuracy', 'Undefined entropy for unused units']
    return checks


def population_tables(s):
    fold_rows, support_rows, preprocess_rows = [], [], []
    for ds in WINDOWS:
        z = np.load(ROOT / f'results/preds/{ds}_pretrained.npz', allow_pickle=True)
        names = np.asarray(z['names']).astype(str)
        gt = z['gt']
        frames = np.array([len(x) for x in gt])
        segments = np.array([1+np.count_nonzero(x[1:] != x[:-1]) for x in gt])
        mapping = {}
        for line in (ROOT / f'data/{ds}/mapping/mapping.txt').read_text().splitlines():
            k, value = line.split(' ', 1); mapping[k] = value
        for k in sorted(set(np.concatenate(gt))):
            support_rows.append({'Dataset': ds, 'Class ID': int(k), 'Class': mapping[str(k)],
                                 'Segments': sum(np.count_nonzero(np.asarray(x)[np.r_[True, x[1:] != x[:-1]]] == k) for x in gt),
                                 'Frames': sum(np.count_nonzero(x == k) for x in gt)})
        for group in ['recording', 'subject']:
            fold = np.load(OUT / f'artifacts/{ds}/folds/{group}.npz')
            assert np.array_equal(names, fold['names'])
            for f in range(4):
                test = fold['folds'] == f
                if group == 'subject':
                    assert not set(fold['subjects'][test]) & set(fold['subjects'][~test])
                for w in [WINDOWS[ds][0], WINDOWS[ds][-1]]:
                    prep = np.load(OUT / f'artifacts/{ds}/cv/{group}/w{w}_f{f}.npz')
                    ids = prep['fit_ids']; assert np.all(fold['folds'][ids[:, 0]] != f)
                    assert np.all(ids[:, 1] % w == 0) and np.all(ids[:, 1]+w <= frames[ids[:, 0]])
                    assert len(np.unique(ids, axis=0)) == len(ids)
                    preprocess_rows.append({'Dataset': ds, 'Population': group, 'Window': w, 'Fold': f,
                                            'Fitting windows': len(ids), 'PCA dimension': len(prep['pca_components']),
                                            'Explained variance %': 100*prep['pca_explained_variance_ratio'].sum()})
                frow = subset(s, dataset=ds, grouping=group, window=WINDOWS[ds][0], num_units=10, seed=111, readout='hard', level='fold', fold=f).iloc[0]
                assert frow.n_test_segments == segments[test].sum()
                fold_rows.append({'Dataset': ds, 'Grouping': group, 'Fold': f, 'Fit recordings': int((~test).sum()),
                                  'Test recordings': int(test.sum()), 'Fit segments': int(segments[~test].sum()),
                                  'Test segments': int(segments[test].sum()), 'Test frames': int(frames[test].sum()),
                                  'Held-out subjects': ', '.join(sorted(set(fold['subjects'][test]))) if group == 'subject' else 'subjects may overlap',
                                  'Missing fit classes': frow.missing_train_classes})
        for rep in ['latent', 'raw', 'per_recording']:
            for w in WINDOWS[ds] if rep == 'latent' else [WINDOWS[ds][0], WINDOWS[ds][-1]]:
                prep = np.load(OUT / f'artifacts/{ds}/{rep}/w{w}_preprocess.npz')
                ids = prep['fit_ids']; assert np.all(ids[:,1]+w <= frames[ids[:,0]])
                preprocess_rows.append({'Dataset': ds, 'Population': rep, 'Window': w, 'Fold': 'pooled',
                                        'Fitting windows': len(ids), 'PCA dimension': len(prep['pca_components']),
                                        'Explained variance %': 100*prep['pca_explained_variance_ratio'].sum()})
    return pd.DataFrame(fold_rows), pd.DataFrame(support_rows), pd.DataFrame(preprocess_rows)


def aggregate(frame, keys, metrics):
    rows = []
    for values, group in frame.groupby(keys, dropna=False, sort=True):
        if not isinstance(values, tuple): values = (values,)
        row = dict(zip(keys, values)); row['n_rows'] = len(group)
        for metric in metrics:
            row[metric+'_mean'] = group[metric].mean()
            row[metric+'_seed_sd'] = group[metric].std(ddof=0)
        rows.append(row)
    return pd.DataFrame(rows)


def figures(c, i, s, reuse):
    plt.rcParams.update({'font.size': 9})
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for d, ds in enumerate(WINDOWS):
        for j, metric in enumerate(['mi_unit_action', 'mi_unit_subject', 'cmi_unit_subject_given_action']):
            ax = axes[d, j]
            for w in WINDOWS[ds]:
                g = subset(i, dataset=ds, representation='latent', window=w).groupby('num_units')
                m, sd = g[metric].mean(), g[metric].std(ddof=0)
                line = ax.errorbar(m.index, m, yerr=sd, marker='o', label=f'W={w} observed')
                ref = g[metric+'_permutation_mean'].mean()
                ax.plot(ref.index, ref, '--', color=line[0].get_color(), label=f'W={w} reference mean')
            ax.set(title=f'{ds}: '+['I(U;A)', 'I(U;S)', 'I(U;S | A)'][j], xscale='log', xlabel='Vocabulary size', ylabel='nats')
            ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.suptitle('Pooled latent information: solid ± seed SD; dashed descriptive permutation mean')
    fig.tight_layout(); fig.savefig(DETAIL/'information_all_windows.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(4, 3, figsize=(15, 14))
    for row, (ds, w) in enumerate((ds, w) for ds, ws in WINDOWS.items() for w in [ws[0], ws[-1]]):
        for col, mode in enumerate(['pooled', 'recording', 'subject']):
            ax = axes[row, col]
            for method, temp, label in [('hard', None, 'hard'), ('soft', .5, 'soft 0.5×'), ('soft', 1., 'soft 1×'), ('soft', 2., 'soft 2×')]:
                if mode == 'pooled':
                    data = subset(c, dataset=ds, representation='latent', window=w, readout=method, temperature_multiplier=temp)
                    metric='MoF'
                else:
                    data = subset(s, dataset=ds, grouping=mode, window=w, readout=method, temperature_multiplier=temp, level='pooled_oof', status='complete')
                    metric='frame_mof'
                g=data.groupby('num_units')[metric]; m=g.mean()
                ax.errorbar(m.index, m, yerr=g.std(ddof=0), marker='o', label=label)
            ax.set(title=f'{ds}, W={w}: '+('pooled clustering' if mode=='pooled' else mode+' CV readout'), xlabel='Vocabulary size', ylabel='frame MoF (%)', xscale='log')
            ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(DETAIL/'readout_all_temperatures.png', dpi=160); plt.close(fig)
    fig, axes=plt.subplots(2,2,figsize=(12,8))
    for row,ds in enumerate(WINDOWS):
        for col,group in enumerate(['recording','subject']):
            ax=axes[row,col]
            for w in [WINDOWS[ds][0],WINDOWS[ds][-1]]:
                for method,t in [('hard',None),('soft',1.)]:
                    data=subset(s,dataset=ds,grouping=group,window=w,readout=method,temperature_multiplier=t,level='pooled_oof',status='complete')
                    g=data.groupby('num_units').balanced_accuracy; m=g.mean()
                    ax.errorbar(m.index,m,yerr=g.std(ddof=0),marker='o',label=f'W={w}, {method}')
            ax.set(title=f'{ds}: {group} CV',ylabel='segment balanced accuracy (%)',xlabel='Vocabulary size',xscale='log');ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(DETAIL/'balanced_accuracy.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for row,ds in enumerate(WINDOWS):
        for col,w in enumerate([WINDOWS[ds][0],WINDOWS[ds][-1]]):
            ax=axes[row,col]
            for fold in range(4):
                data=subset(reuse,dataset=ds,window=w,fold=fold)
                g=data.groupby('num_units').fraction_used;m=g.mean()
                ax.errorbar(m.index,m,yerr=g.std(ddof=0),marker='o',label=f'fold {fold}')
            ax.set(title=f'{ds}, W={w}: fold-specific reuse',ylabel='fraction of all units',xlabel='Vocabulary size',xscale='log',ylim=(0,1.05));ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(DETAIL/'reuse_by_fold.png',dpi=160);plt.close(fig)


def main():
    DETAIL.mkdir(exist_ok=True)
    if (OUT/'REPORT.md').exists() and not (DETAIL/'REPORT_previous.md').exists():
        shutil.copy2(OUT/'REPORT.md', DETAIL/'REPORT_previous.md')
    c,i,s,u = [pd.read_csv(OUT/(name+'.csv')) for name in ['q1q2_cells','q1q2_information','supervised_readouts','unit_reuse']]
    manifest=json.loads((OUT/'manifest.json').read_text())
    checks=audit(c,i,s,u)
    vocabulary_iterations = []
    for artifact in (OUT/'artifacts').rglob('w*_k*_s*.npz'):
        with np.load(artifact) as saved:
            vocabulary_iterations.append(int(saved['n_iter']))
    checks['saved_vocabulary_iterations'] = {
        'artifacts':len(vocabulary_iterations), 'minimum':min(vocabulary_iterations),
        'maximum':max(vocabulary_iterations), 'at_iteration_limit':sum(n>=300 for n in vocabulary_iterations)}
    folds,support,prep=population_tables(s)
    for name,data in [('folds',folds),('class_support',support),('preprocessing',prep)]: data.to_csv(DETAIL/(name+'.csv'),index=False)
    ca=aggregate(c,['dataset','representation','window','num_units','readout','temperature_multiplier'],METRICS)
    ia=aggregate(i,['dataset','representation','window','num_units'],IM+['occupied_units','n_windows','terminal_windows'])
    sa=aggregate(subset(s,level='pooled_oof',status='complete'),['dataset','grouping','window','num_units','readout','temperature_multiplier'],['frame_mof','balanced_accuracy'])
    for name,data in [('clustering_summary',ca),('information_summary',ia),('supervised_summary',sa)]: data.to_csv(DETAIL/(name+'.csv'),index=False)
    reuse=u.groupby(['dataset','window','num_units','seed','fold'],as_index=False).agg(
        fraction_used=('used_by_at_least_half','mean'), unused_units=('unused','sum'),
        entropy_used_units=('subject_distribution_entropy','mean'), normalized_entropy_used_units=('normalized_subject_distribution_entropy','mean'),
        held_subjects=('n_held_subjects','first'),threshold=('half_subject_threshold','first'))
    reuse.to_csv(DETAIL/'reuse_fold_seed_summary.csv',index=False)
    # Detailed appendices retain all grid cells, including invalid continuous readouts.
    parts=['# A. Information diagnostics — every representation and window\n\n'+CONTEXT,
           'Observed values are mean ± population SD over the three clustering seeds. Reference mean is the mean of the three 20-permutation means. Reference SD column is the mean of the three within-cell permutation SDs; it is not a seed SD or a pooled 60-permutation SD. Entropies and MI are in nats.']
    for (ds,rep,w),g in i.groupby(['dataset','representation','window']):
        parts.append(f'## {ds}, {rep}, W={w}')
        rows=[]
        for k,h in g.groupby('num_units'):
            for metric in IM:
                rows.append([int(k),metric,ms(h[metric],4),f'{h[metric+"_permutation_mean"].mean():.4f}',f'{h[metric+"_permutation_sd"].mean():.4f}'])
        parts.append(table(rows,['Ku','Quantity','Observed ± seed SD','Reference mean','Mean within-cell reference SD']))
    (DETAIL/'A_information.md').write_text('\n\n'.join(parts)+'\n',encoding='utf-8')
    parts=['# B. All pooled clustering scores\n\n'+CONTEXT,'Scores are percentages, mean ± population SD across exactly three seeds; no vocabulary-size or temperature averaging. All boundaries are oracle ground-truth boundaries.']
    for (ds,rep,w),g in c.groupby(['dataset','representation','window']):
        parts.append(f'## {ds}, {rep}, W={w}')
        rows=[]
        for (k,method,t),h in g.groupby(['num_units','readout','temperature_multiplier'],dropna=False):
            assert set(h.seed)==set(SEEDS)
            rows.append(['independent of Ku' if pd.isna(k) else int(k),method,'—' if pd.isna(t) else t]+[ms(h[m]) for m in METRICS])
        parts.append(table(rows,['Ku','Readout','Temperature']+METRICS))
    (DETAIL/'B_clustering.md').write_text('\n\n'.join(parts)+'\n',encoding='utf-8')
    parts=['# C. Supervised readouts and class recall\n\n'+CONTEXT,'Primary metrics come from pooled out-of-fold predictions within each seed. Continuous diagnostics have no clustering seed; all are invalid and are listed only in the failure table of the main report. Fold support is in folds.csv and class_support.csv; original per-fold per-class supports remain in supervised_readouts.csv.']
    recall_rows=[]
    for (ds,group,w),g in subset(s,level='pooled_oof',status='complete').groupby(['dataset','grouping','window']):
        parts.append(f'## {ds}, {group} grouping, W={w}')
        rows=[]
        for (k,method,t),h in g.groupby(['num_units','readout','temperature_multiplier'],dropna=False):
            assert set(h.seed)==set(SEEDS)
            rows.append([int(k),method,'—' if pd.isna(t) else t,ms(h.frame_mof),ms(h.balanced_accuracy)])
            keys=list(json.loads(h.iloc[0].per_class_recall))
            for label in keys:
                values=np.array([json.loads(x)[label]*100 for x in h.per_class_recall])
                recall_rows.append({'dataset':ds,'grouping':group,'window':w,'num_units':int(k),'readout':method,'temperature_multiplier':t,
                                    'class_id':int(label),'recall_mean_percent':values.mean(),'recall_seed_sd':values.std(ddof=0),
                                    'test_segment_support':json.loads(h.iloc[0].test_class_support)[label]})
        parts.append(table(rows,['Ku','Readout','Temperature','Frame MoF ± seed SD','Balanced accuracy ± seed SD']))
    recalls=pd.DataFrame(recall_rows);recalls.to_csv(DETAIL/'per_class_recall.csv',index=False)
    fold_class_rows=[]
    for r in subset(s,level='fold').itertuples():
        train=json.loads(r.train_class_support);test=json.loads(r.test_class_support)
        for label,recall in json.loads(r.per_class_recall).items():
            fold_class_rows.append({'dataset':r.dataset,'grouping':r.grouping,'window':r.window,
                                    'num_units':r.num_units,'seed':r.seed,'readout':r.readout,
                                    'temperature_multiplier':r.temperature_multiplier,'fold':r.fold,
                                    'status':r.status,'class_id':int(label),'train_segment_support':train[label],
                                    'test_segment_support':test[label],'recall_percent':100*recall})
    pd.DataFrame(fold_class_rows).to_csv(DETAIL/'fold_class_support_recall.csv',index=False)
    for ds in WINDOWS:
        parts.append(f'## {ds}: full-patch per-class recall, hard and base-soft subject-grouped readouts')
        q=subset(recalls,dataset=ds,grouping='subject',window=WINDOWS[ds][0])
        q=q[(q.readout=='hard')|q.temperature_multiplier.eq(1)]
        rr=[]
        for (label,method),h in q.groupby(['class_id','readout']):
            name=support[(support.Dataset==ds)&(support['Class ID']==label)].iloc[0]['Class']
            rr.append([int(label),name,method]+[f'{h[h.num_units==k].iloc[0].recall_mean_percent:.2f} ± {h[h.num_units==k].iloc[0].recall_seed_sd:.2f}' for k in KS])
        parts.append(table(rr,['Class ID','Class','Readout']+list(map(str,KS))))
    (DETAIL/'C_supervised.md').write_text('\n\n'.join(parts)+'\n',encoding='utf-8')
    parts=['# D. Held-out subject unit reuse, by fold\n\n'+CONTEXT,
           'Each row summarizes three clustering seeds within one fold. Folds fit different vocabularies; units are never matched across folds. Entropy means exclude units unused by every held-out subject. Unused counts are separate. Extra Ku=20,50,500 results extend the restricted D grid; they are retained explicitly.']
    rows=[]
    for (ds,w,k,f),h in reuse.groupby(['dataset','window','num_units','fold']):
        rows.append([ds,w,k,f,int(h.held_subjects.iloc[0]),int(h.threshold.iloc[0]),ms(h.fraction_used,4),ms(h.unused_units),ms(h.entropy_used_units,4),ms(h.normalized_entropy_used_units,4)])
    parts.append(table(rows,['Dataset','W','Ku','Fold','Subjects','Threshold','Reuse fraction ± seed SD','Unused units ± seed SD','Entropy (nats) ± seed SD','Normalized entropy ± seed SD']))
    (DETAIL/'D_unit_reuse.md').write_text('\n\n'.join(parts)+'\n',encoding='utf-8')
    figures(c,i,s,reuse)
    write_main(c,i,s,reuse,folds,support,prep,manifest,checks)
    checks['source_sha256']={name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in ['q1q2_cells.csv','q1q2_information.csv','supervised_readouts.csv','unit_reuse.csv','manifest.json']}
    checks['report_generated_utc']=datetime.now(timezone.utc).isoformat()
    checks['additional_checks']=['Saved subject folds disjoint', 'Fit sample IDs contain fitting recordings only and complete windows',
                                 'Fold segment counts match source ground truth', 'Three seed identities checked in each valid aggregate']
    (DETAIL/'report_audit.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks,indent=2))


def write_main(c,i,s,reuse,folds,support,prep,manifest,checks):
    # This report uses descriptive, predefined contrasts; it does not select a best cell.
    txt=['# Q1/Q2: motion-unit information and readout diagnostics',
         '## Ten-line summary',
         '\n'.join([
         '1. Both checkpoint/quantization gates passed; D7 reproduced 44.85% MoF on HuGaDB and 41.66% on LARa.',
         '2. HuGaDB has 10 observed classes / 18 subjects / IMU inputs; LARa has 8 classes / 16 subjects / optical motion-capture inputs.',
         '3. At full-patch windows, action MI rises from Ku=10 to 1000 on both datasets; subject MI and conditional subject MI rise as well.',
         '4. HuGaDB full-patch action AMI falls across those endpoints, while LARa action AMI rises; raw MI and AMI describe different quantities.',
         '5. Full-patch hard pooled MoF changes from 49.23 ± 0.36 to 34.95 ± 0.55 on HuGaDB, and from 28.79 ± 0.02 to 38.05 ± 2.55 on LARa.',
         '6. Full-patch hard subject-grouped readout MoF changes from 41.69 ± 0.45 to 79.16 ± 0.20 on HuGaDB, and from 26.62 ± 0.59 to 50.56 ± 0.62 on LARa.',
         '7. Soft assignment, duration weighting, window size and grouping change scores; all matched comparisons and temperature settings are retained below.',
         '8. Oracle many-to-one and Hungarian majority-unit mappings differ substantially; raw, recording-standardized and within-subject controls are reported separately.',
         '9. The fraction of units shared by at least half the held-out subjects declines with vocabulary size in the displayed fold-specific reuse diagnostics.',
         '10. All eight continuous supervised configurations are invalid (32 nonconverged folds); artifact and timing gaps prevent claiming full protocol compliance.'
         ]),
         '## Reading this report',
         CONTEXT+' All score comparisons below are descriptive. Percentages are used for MoF, Edit, F1 and balanced accuracy; differences are percentage points (pp). “±” denotes population SD across exactly the three clustering seeds unless explicitly labelled otherwise. Seed SD measures initialization sensitivity, not sampling uncertainty. No cell is selected as “best”.',
         'This revision reads the existing measurements and saved artifacts without fitting models or rerunning experiments. The [previous report](report_details/REPORT_previous.md) is preserved. Full tables: [A: information and permutation references](report_details/A_information.md), [B: all five clustering metrics](report_details/B_clustering.md), [C: all supervised settings](report_details/C_supervised.md), [D: reuse by fold](report_details/D_unit_reuse.md). Numerical audit: [report_audit.json](report_details/report_audit.json).',
         '## 1. Provenance, populations and scope']
    rows=[]
    for g in manifest['gates']:
        ds=g['dataset'];pop=support[support.Dataset==ds]
        rows.append([ds,g['n_recordings'],g['n_frames'],int(pop.Segments.sum()),g['same_checkpoint_quantisation_agreement'],f'{g["historical_prediction_agreement"]:.9f}',g['d7']['MoF'],'passed'])
    txt.append(table(rows,['Dataset','Recordings','Frames','GT segments','Exact quantization agreement','Historical agreement','D7 MoF','Gate']))
    for g in manifest['gates']:
        txt.append(f'**{g["dataset"]} checkpoint:** `{g["checkpoint"]}`; SHA-256 `{g["checkpoint_sha256"]}`. Cache: `{g["cache"]}`.')
    txt += ['The gates require exact same-checkpoint latent quantization, historical agreement ≥0.999, and D7 MoF within 1.5 pp of the reference. HuGaDB uses a fresh released-checkpoint cache rather than the incompatible epoch-30 dump. LARa reuses the verified cache. D7 uses the original SMQ codebook, whereas the new vocabulary majority-unit control uses Ku clusters; they are different measurements.',
            'The existing `exp2round` scores are historical references, not matched controls: both the clustering fit policy and HuGaDB checkpoint differ. CPU-only scope excludes merge curves, decoders, duration models, BABEL and encoder training.',
            'Recorded software: Python 3.10.14, NumPy 1.26.4, scikit-learn 1.2.2, PyTorch 2.1.0+cu121; numerical-library thread cap 8. A CUDA-capable package version does not imply GPU execution.',
            '### Action support', table(support),
            'HuGaDB IDs are non-contiguous: 1–8, 10 and 11. Mapping entries 9 (bicycling) and 12 (sitting_in_car) have no observations in this evaluated collection. No HuGaDB none class is present. LARa class 8 (`None`) is included throughout. No observed class was removed.']
    txt += ['## 2. Methods and interpretation of metrics',
        '**Shared grid.** Windows are HuGaDB 60/30/15 and LARa 50/25/12; Ku=10,20,50,100,500,1000; clustering seeds 111,222,1538574472. Sampling, PCA and splits use 111. Task C uses only full/quarter windows; raw and per-recording controls use Ku=10,100,1000 at those windows.',
        '**Preprocessing and vocabulary.** Non-overlapping flattened windows; incomplete tails excluded from fitting and zero-padded for assignment, with all real frames retained in evaluation. Each fitting population samples 10,000 complete windows uniformly without replacement. Coordinate standardization and randomized PCA-64 fit only this sample. Full Lloyd KMeans uses k-means++, n_init=5, max_iter=300, tol=1e-4 for every Ku and for C-class final segment clustering. Sample IDs and transformations are saved. PCA explained variance varies by representation; see [preprocessing.csv](report_details/preprocessing.csv).',
        '**Segment features.** Hard and soft histograms weight each window by its actual frame overlap with the GT segment, normalize to one, then take the elementwise square root. Soft probabilities use squared PCA-space Euclidean distances and temperature equal to 0.5/1/2 times the median strictly positive second-nearest-minus-nearest distance gap on fitting windows. Comparisons share the vocabulary and transforms. Continuous features are overlap-weighted means of PCA window features, not original frame-latent means.',
        '**Pooled versus CV.** Pooled preprocessing, vocabularies and segment clustering fit the whole collection without action supervision, but GT boundaries and evaluation mapping are label-based. CV fits preprocessing, vocabulary, temperature and readout on fitting recordings only. Four folds are built by descending group frame count, seed-111 tie breaking and greedy frame balancing. Frozen encoder exposure remains transductive in either setting.',
        '**Supervised fitting.** Each training segment is one sample; class balancing uses segment counts. Logistic regression uses L2, C=1, intercept, LBFGS, max_iter=2000, tol=1e-4, multinomial treatment for more than two classes, and no further feature standardization. Primary BA is class-average segment recall from pooled OOF predictions. Frame MoF weights each correctly predicted segment by its duration. Neither is an information-theoretic ceiling.',
        '**Scoring.** Existing helpers produce dataset frame MoF; Edit averaged over recordings; F1 from pooled TP/FP/FN at IoU 0.10/0.25/0.50. Adjacent segments with equal predictions can merge in the scored frame sequence. Hungarian matching is dataset-wide and one-to-one; unmatched unit IDs map to a sentinel. Many-to-one assigns each unit its most frequent evaluation class, explicitly an oracle mapping. All five clustering metrics are available in Appendix B. Saved individual clustering scores are rounded to 0.01 before reporting seed summaries.',
        '**Information.** One observation per assigned window, including terminal windows; window action is its majority real-frame label, subject is parsed from recording ID. MI/CMI and entropies use natural logs. NMI uses arithmetic normalization; AMI includes chance adjustment. Twenty global count-preserving permutations (seeds 111–130) supply marginal references; within-action permutations supply conditional subject references. Reference SD describes permutations, not independent temporal uncertainty; unit entropy is invariant under global permutation.',
        '### Pooled preprocessing and window counts']
    rr=[]
    for (ds,rep,w),h in i.groupby(['dataset','representation','window']):
        p=subset(prep,Dataset=ds,Population=rep,Window=w).iloc[0]
        rr.append([ds,rep,w,int(h.n_windows.iloc[0]),int(h.terminal_windows.iloc[0]),int(p['Fitting windows']),int(p['PCA dimension']),f'{p["Explained variance %"]:.2f}'])
    txt.append(table(rr,['Dataset','Representation','W','Assigned windows','Terminal windows','Fitting windows','PCA dim','Explained variance %']))
    txt += ['## 3. Task A — action and subject information',
            'The following full-patch table shows observed information and its corresponding reference mean. All three windows and restricted representations, every entropy, arithmetic NMI, AMI, and reference mean/SD are in Appendix A. Occupancy distributions are preserved in `q1q2_information.csv`.']
    rr=[]
    for ds,ws in WINDOWS.items():
        for k in KS:
            h=subset(i,dataset=ds,representation='latent',window=ws[0],num_units=k)
            rr.append([ds,k]+[ms(h[m],3) for m in IM[:3]]+[f'{h[m+"_permutation_mean"].mean():.3f}' for m in IM[:3]]+[ms(h.ami_unit_action,3),ms(h.ami_unit_subject,3)])
    txt.append(table(rr,['Dataset','Ku','I(U;A)','I(U;S)','I(U;S given A)','Ref I(U;A)','Ref I(U;S)','Ref conditional','AMI action','AMI subject']))
    txt.append('Full-patch action MI increases from 0.985 to 1.569 nats on HuGaDB and 0.293 to 0.798 on LARa. The count-preserving references also vary with Ku, so unadjusted MI increases alone do not establish improved action invariance. Conditional subject MI remains above its within-action reference in these cells. Action and subject associations can coexist; their magnitudes do not establish causal attribution.')
    txt.append('![Information across all window sizes](report_details/information_all_windows.png)')
    rr=[]
    for (ds,w),h in subset(i,representation='latent').groupby(['dataset','window']):
        for k in [10,100,1000]:
            g=subset(h,num_units=k)
            occup=[np.asarray(json.loads(x)) for x in g.occupancy]
            rr.append([ds,w,k,ms(g.occupied_units,1),ms(g.h_unit,3),ms(g.h_action_given_unit,3),ms(g.h_subject_given_unit,3),ms([100*a.max()/a.sum() for a in occup])])
    txt.append('### Occupancy and conditional entropies at the prespecified restricted Ku values')
    txt.append(table(rr,['Dataset','W','Ku','Occupied units','H(U)','H(A given U)','H(S given U)','Largest unit % of windows']))
    txt += ['## 4. Task B — matched pooled sweep and controls', '### Hard histograms: all windows and vocabulary sizes']
    rr=[]
    for ds,ws in WINDOWS.items():
        for w,k in itertools.product(ws,KS):
            h=subset(c,dataset=ds,representation='latent',window=w,num_units=k,readout='hard')
            rr.append([ds,w,k]+[ms(h[m]) for m in METRICS])
    txt.append(table(rr,['Dataset','W','Ku']+METRICS))
    txt += ['HuGaDB full-patch hard MoF decreases between Ku=10 and 1000, while LARa increases across those endpoints. The relationship is not uniformly monotonic across either full grid. Window changes also produce different results; no window is chosen from evaluation scores.',
            '### Majority-unit mapping: identical predictions, different oracle mapping constraints']
    rr=[]
    for ds,ws in WINDOWS.items():
        for w,k in itertools.product(ws,KS):
            a=subset(c,dataset=ds,representation='latent',window=w,num_units=k,readout='majority_unit_hungarian')
            b=subset(c,dataset=ds,representation='latent',window=w,num_units=k,readout='majority_unit_many_to_one')
            rr.append([ds,w,k,ms(a.MoF),ms(b.MoF),ms(paired(b,a))])
    txt.append(table(rr,['Dataset','W','Ku','Hungarian MoF','Oracle many-to-one MoF','Many-to-one minus Hungarian pp']))
    txt.append('At full-patch Ku=1000, the mapping difference is much larger than at Ku=10. The table quantifies that difference without assigning its cause. The many-to-one mapping uses evaluation labels and is not a deployable classifier or a CV result.')
    txt += ['### Count-preserving permutation and duration weighting',
            'Each contrast pairs the same dataset/window/Ku/seed. The permutation control uses one global count-preserving shuffle per clustering seed (`seed + 10000000`), then rebuilds segment histograms and fits final clusters. This differs from Task A’s 20-permutation information references. Length weighting changes only the final histogram KMeans sample weights to segment durations.']
    rr=[]
    for ds,ws in WINDOWS.items():
        for w,k in itertools.product(ws,KS):
            base=subset(c,dataset=ds,representation='latent',window=w,num_units=k,readout='hard')
            perm=subset(c,dataset=ds,representation='latent',window=w,num_units=k,readout='permuted_hard')
            weight=subset(c,dataset=ds,representation='latent',window=w,num_units=k,readout='hard_length_weighted')
            rr.append([ds,w,k,ms(base.MoF),ms(perm.MoF),ms(paired(base,perm)),ms(weight.MoF),ms(paired(weight,base))])
    txt.append(table(rr,['Dataset','W','Ku','Hard MoF','Permuted MoF','Hard minus permuted pp','Duration-weighted MoF','Weighted minus hard pp']))
    txt.append('The permutation control is not universally below the observed score: for example, LARa W=12/Ku=10 has hard MoF 30.42 versus permuted 35.44 (seed means). It therefore should not be labelled a universal chance floor. Duration weighting also changes scores in both directions across cells. These descriptive comparisons do not establish a cause.')
    txt += ['### Continuous pooled segment clustering', 'Continuous representations have no vocabulary-size parameter. Their final C-cluster KMeans still has three initialization seeds; this seed variation is applicable to pooled clustering.']
    rr=[]
    for (ds,w),h in subset(c,readout='continuous').groupby(['dataset','window']):rr.append([ds,w]+[ms(h[m]) for m in METRICS])
    txt.append(table(rr,['Dataset','W']+METRICS))
    txt += ['## 5. Task C — hard/soft comparisons and supervised readouts',
            '### Pooled clustering and temperature sensitivity',
            'Hard and all soft temperatures share preprocessing and prototypes. The continuous pooled values are in §4. All five scores for each temperature are in Appendix B.']
    rr=[]
    for ds,ws in WINDOWS.items():
        for w,k in itertools.product([ws[0],ws[-1]],KS):
            h=subset(c,dataset=ds,representation='latent',window=w,num_units=k,readout='hard')
            soft=[subset(c,dataset=ds,representation='latent',window=w,num_units=k,readout='soft',temperature_multiplier=t) for t in [.5,1.,2.]]
            rr.append([ds,w,k,ms(h.MoF)]+[ms(a.MoF) for a in soft]+[ms(paired(soft[1],h))])
    txt.append(table(rr,['Dataset','W','Ku','Hard MoF','Soft 0.5× MoF','Soft 1× MoF','Soft 2× MoF','Base-soft minus hard pp']))
    txt.append('At the prespecified full-patch Ku=100 comparison, HuGaDB hard/base-soft pooled MoF is 42.02/53.35, while LARa is 32.59/30.31. The sign of the hard-to-soft difference therefore differs in this matched comparison. These are seed means; paired SDs are in the last table column. No temperature was selected by evaluation performance.')
    txt += ['### Cross-validation fold populations', table(folds),
            'All folds retain every observed training class. Subject folds are disjoint by parsed ID, and saved fitting-window IDs fall exclusively in fitting recordings; these were checked in this report audit. Recording folds allow the same subject in fitting and evaluation recordings. [Per-class support](report_details/class_support.csv) and original fold-level training/evaluation supports remain available.',
            '### Hard and base-soft readouts: both grouping protocols, all windows and Ku',
            'Each entry aggregates three independently initialized vocabularies. Metrics are computed on pooled OOF predictions within each seed, not averages of fold percentages.']
    rr=[]
    for ds,ws in WINDOWS.items():
        for w,k in itertools.product([ws[0],ws[-1]],KS):
            for method,t in [('hard',None),('soft',1.)]:
                a=subset(s,dataset=ds,window=w,num_units=k,readout=method,temperature_multiplier=t,grouping='recording',level='pooled_oof')
                b=subset(s,dataset=ds,window=w,num_units=k,readout=method,temperature_multiplier=t,grouping='subject',level='pooled_oof')
                rr.append([ds,w,k,method,ms(a.frame_mof),ms(b.frame_mof),ms(paired(a,b,'frame_mof')),ms(a.balanced_accuracy),ms(b.balanced_accuracy)])
    txt.append(table(rr,['Dataset','W','Ku','Readout','Recording MoF','Subject MoF','Recording minus subject pp','Recording BA','Subject BA']))
    txt.append('The grouping difference is a protocol difference, not an isolated estimate of previously seeing a person. For example, full-patch hard subject-grouped MoF at Ku=1000 is 79.16 on HuGaDB and 50.56 on LARa, while their pooled hard-clustering MoF values are 34.95 and 38.05. Both the fitting protocol and readout differ across these settings; the contrast does not isolate one component.')
    txt.append('At full-patch Ku=100, hard/base-soft subject-grouped MoF is 66.08/69.24 on HuGaDB and 43.10/52.28 on LARa. Corresponding balanced accuracy is 69.88/75.86 and 49.14/55.95. Thus the LARa soft comparison at this setting differs between pooled clustering and supervised readout; this is a measured protocol-dependent difference, not a causal diagnosis.')
    txt += ['### Supervised temperature sensitivity', 'Below are paired differences from the same seed’s base-temperature soft readout. Absolute MoF and BA at all three temperatures are in Appendix C.']
    rr=[]
    for ds,ws in WINDOWS.items():
        for w,group,k in itertools.product([ws[0],ws[-1]],['recording','subject'],KS):
            hh=[subset(s,dataset=ds,window=w,grouping=group,num_units=k,readout='soft',temperature_multiplier=t,level='pooled_oof') for t in [.5,1.,2.]]
            rr.append([ds,w,group,k,ms(paired(hh[0],hh[1],'frame_mof')),ms(paired(hh[2],hh[1],'frame_mof')),ms(paired(hh[0],hh[1],'balanced_accuracy')),ms(paired(hh[2],hh[1],'balanced_accuracy'))])
    txt.append(table(rr,['Dataset','W','Grouping','Ku','ΔMoF 0.5×−1×','ΔMoF 2×−1×','ΔBA 0.5×−1×','ΔBA 2×−1×']))
    txt += ['![All temperatures and both CV grouping protocols](report_details/readout_all_temperatures.png)',
            '![Balanced accuracy in a separate figure](report_details/balanced_accuracy.png)',
            '### Continuous supervised readouts — invalid, not comparable evidence',
            'Every one of the 32 continuous fold fits emitted a convergence warning at max_iter=2000. Consequently all eight pooled OOF configurations are invalid. The table retains their computed values solely for failure audit; they must not be used as valid continuous-versus-histogram comparisons. No replacement regularization, scaling, solver, or iteration limit was used. Seed spread is not applicable.']
    invalid=subset(s,readout='continuous',level='pooled_oof')
    rr=[[r.dataset,r.window,r.grouping,'4/4',f'{r.frame_mof:.2f}',f'{r.balanced_accuracy:.2f}','INVALID'] for r in invalid.itertuples()]
    txt.append(table(rr,['Dataset','W','Grouping','Nonconverged folds','Invalid OOF MoF','Invalid OOF BA','Status']))
    txt += ['### Per-class recall and support',
            'Appendix C includes full-patch hard/base-soft subject-grouped recall at every Ku and every class, named using the dataset mapping files. [per_class_recall.csv](report_details/per_class_recall.csv) includes all valid temperatures, both windows and both groupings. Class supports are segment counts; frame imbalance is shown separately in §1. BA and frame MoF should be read together because one averages classes over segments and the other weights duration.']
    txt += ['## 6. Task D — restricted Q1 controls',
            '### Raw features and per-recording latent standardization',
            'Raw controls use the existing model-input channels without the encoder. The standardized-latent control computes each coordinate’s population mean/SD over the complete recording (SD below 1e-8 replaced with 1) before windowing. Both then use the same sampled standardization/PCA/KMeans pipeline. Recording standardization uses future frames and is not an online method. These are pooled diagnostics; raw features have no encoder exposure, but their fitting is transductive.']
    rr=[]
    for ds,ws in WINDOWS.items():
        for w,k in itertools.product([ws[0],ws[-1]],[10,100,1000]):
            baseline=subset(c,dataset=ds,representation='latent',window=w,num_units=k,readout='hard')
            for rep in ['latent','raw','per_recording']:
                a=subset(c,dataset=ds,representation=rep,window=w,num_units=k,readout='hard')
                inf=subset(i,dataset=ds,representation=rep,window=w,num_units=k)
                rr.append([ds,w,k,rep,ms(a.MoF),'reference' if rep=='latent' else ms(paired(a,baseline))]+[ms(inf[m],3) for m in IM[:3]])
    txt.append(table(rr,['Dataset','W','Ku','Representation','Hard MoF','ΔMoF vs latent pp','I(U;A)','I(U;S)','I(U;S given A)']))
    txt.append('At full-patch Ku=100, HuGaDB latent/raw/per-recording-standardized MoF is 42.02/43.99/59.15; LARa is 32.59/32.67/24.00. Per-recording standardization therefore changes the score in opposite directions in this prescribed comparison. All seeds, other restricted Ku values and window scales are retained above.')
    txt.append('These controls alter representation and the subsequent fitted geometry. The raw-feature comparison is not an isolated causal estimate of encoder benefit. The per-recording control is not evidence that subject identity has been removed; its subject-information measurements are reported directly.')
    txt += ['### Within-subject segment clustering',
            'The pooled latent vocabulary and hard histograms remain fixed; only final C-cluster segment models and Hungarian mappings are fitted separately per subject. Coverage is 100% of frames in every saved cell, and no subject had fewer than C segments. Separate mappings add oracle flexibility.']
    rr=[]
    for ds,ws in WINDOWS.items():
        for w,k in itertools.product([ws[0],ws[-1]],[10,100,1000]):
            a=subset(c,dataset=ds,representation='latent',window=w,num_units=k,readout='hard')
            b=subset(c,dataset=ds,representation='latent',window=w,num_units=k,readout='within_subject')
            rr.append([ds,w,k,ms(a.MoF),ms(b.MoF),ms(paired(b,a)),ms(b.Edit),ms(b['F1@50'])])
    txt.append(table(rr,['Dataset','W','Ku','Pooled MoF','Within-subject MoF','Within minus pooled pp','Within Edit','Within F1@50']))
    txt.append('Within-subject MoF is not uniformly higher: at LARa full-patch Ku=1000 it is 29.03 versus pooled 38.05. The extra oracle flexibility therefore does not itself guarantee a higher score after this clustering fit.')
    txt += ['### Unit reuse across held-out subjects',
            'For a fold, a unit is shared if it occurs in at least ceil(held-subject-count/2) subjects, using complete windows only. Per-unit entropy first divides each subject’s unit count by that subject’s complete-window count, normalizes these rates across subjects, then computes entropy in nats and divides by log(number of held subjects). Units unused everywhere have undefined entropy and are counted separately.',
            'The original plan restricts Task D to Ku=10,100,1000. The implementation also saved reuse at Ku=20,50,500 by reusing Task C vocabularies; these are explicitly additional descriptive outputs. The primary table below uses the restricted values. Fold-specific three-seed means/SD and all extra Ku values are in Appendix D. The figure separates folds instead of mixing seed and fold variation.',
            '![Reuse with separate fold curves](report_details/reuse_by_fold.png)']
    rr=[]
    for (ds,w,k,f),h in reuse[reuse.num_units.isin([10,100,1000])].groupby(['dataset','window','num_units','fold']):
        rr.append([ds,w,k,f,int(h.held_subjects.iloc[0]),ms(h.fraction_used,3),ms(h.unused_units,1),ms(h.entropy_used_units,3),ms(h.normalized_entropy_used_units,3)])
    txt.append(table(rr,['Dataset','W','Ku','Fold','Subjects','Reuse fraction','Unused units','Mean entropy, used units','Mean normalized entropy']))
    txt += ['## 7. Coverage, validation and reproducibility limits',
            table([[name, v['expected'],v['rows'],v['missing'],v['unexpected'],v['duplicates']] for name,v in checks.items() if isinstance(v,dict) and 'expected' in v],['Table','Expected rows','Observed','Missing','Unexpected','Duplicate keys']),
            'Coverage was checked by the full dataset/representation/window/Ku/seed/readout/temperature/fold key sets, not merely by total row counts. The reuse expected set includes the explicitly additional Ku=20,50,500 outputs. Of 2,920 supervised rows, 2,304 valid fold rows and 576 valid pooled OOF rows describe hard/soft readouts; 32 invalid fold rows and 8 invalid pooled OOF rows describe continuous readouts. The 40 invalid rows are eight failed configurations, not 40 independent experiments.',
            'The prior implementation tests cover information examples, count preservation, overlap histograms, mapping fragmentation and folds. This report additionally verifies occupancy sums, confusion totals, per-class recall/BA, real fold segment support, subject disjointness and complete fitting-window membership. Frame MoF cannot be independently reconstructed from confusion matrices alone because they omit durations; individual saved prediction sequences are needed for that audit.',
            f'The {checks["saved_vocabulary_iterations"]["artifacts"]} saved vocabulary artifacts have iteration counts from {checks["saved_vocabulary_iterations"]["minimum"]} to {checks["saved_vocabulary_iterations"]["maximum"]}; none reached the 300-iteration limit. This does not verify final segment-clustering convergence. The archived smoke outputs with incorrect non-contiguous HuGaDB class indices are excluded; source CSVs are the consolidated, corrected measurements.',
            '### Gaps in the saved evidence',
            '- Vocabulary artifacts retain centers, per-recording window assignments, inertia and iteration counts; preprocessing artifacts retain sample IDs and transformation arrays. Fold memberships are saved. The current audit checks their population membership; it does not rerun quantization or model fitting.',
            '- The runners compute final segment labels and logistic predictions but do not persist them. Final segment-clustering models and logistic coefficients are also not saved. Thus the requested incremental prediction/model archive is incomplete, even though metric rows cover the grid.',
            '- Information contingency matrices are not saved explicitly. They could be reconstructed from assignments and source labels, but the delivered table stores only the derived quantities and occupancy distributions.',
            '- Final segment-clustering convergence is not recorded. “Complete” pooled status indicates a recorded score, not verified convergence of every final KMeans. Saved vocabulary iteration counts provide a narrower diagnostic.',
            '- Runtime fields mix individual and cumulative cell timings; summing them double-counts shared work. Some CV rows omit cache identity/sample/convergence metadata requested per cell. Exact total compute time and execution-priority compliance cannot be reconstructed reliably.',
            '### Timing inconsistency',
            f'The source manifest records start `{manifest["started_utc"]}` and completion `{manifest["completed_utc"]}`. Their difference is {(datetime.fromisoformat(manifest["completed_utc"].replace("Z","+00:00"))-datetime.fromisoformat(manifest["started_utc"].replace("Z","+00:00"))).total_seconds()/3600:.2f} hours. Completion precedes its recorded deadline `{manifest["deadline_utc"]}`, yet `deadline_compliance` is false and the note claims the deadline was exceeded. Those fields contradict each other. The previous report’s overrun claim is unsupported by these timestamps. The historical manifest is preserved; the report audit records this inconsistency rather than inventing a corrected run history. Reporting time in this revision is separate from the original run.',
            '## Open questions',
            '- What accounts for the different Ku trends in pooled clustering and supervised readouts within each dataset?',
            '- How much of the observed subject association persists with encoders trained without held-out subjects?',
            '- How would these measurements change without ground-truth segment boundaries and evaluation-label mappings?',
            '- Why do the count-preserving histogram permutation controls exceed observed MoF in some LARa cells?',
            '- What explains the hard/soft and temperature differences across windows and datasets?',
            '- Would a separately specified converged continuous-readout protocol change its comparison with histogram readouts?',
            '## What these numbers cannot establish',
            'They do not establish a causal bottleneck, which component dominates, a project contribution, or what the project should do next. They do not identify body-shape effects from HuGaDB IMU inputs. Action/subject MI depends on the observed window population and finite-sample partition; raw values at different Ku are not controlled estimates of invariance. All encoder results remain transductive. Pooled raw fitting is transductive despite no encoder exposure. The CV grouping gap changes the protocol rather than isolating previous exposure to a person.',
            'Oracle boundaries, Hungarian mappings, oracle many-to-one mappings, supervised fitting/evaluation, information labels and within-action permutations all use action labels. Subject IDs are used for grouping, separate subject fits and diagnostics. These diagnostics do not measure unsupervised boundary discovery or end-to-end action segmentation without labels. Seed spreads and permutation spreads are descriptive; neither is sampling uncertainty. Invalid continuous readouts cannot support a valid comparison or an information ceiling.',
            '## Not run',
            'No planned measurement keys are missing from the saved score grid. The 8 continuous supervised configurations are invalid because every fold failed the prescribed convergence criterion; their 40 rows are retained in [not_run.csv](not_run.csv). No substitute method was fitted in this reporting revision. Required individual predictions, final fitted readout models, explicit contingency artifacts and full convergence/runtime logs are unavailable as detailed above. Experiments outside Q1/Q2 scope were not requested and are not treated as omissions.',
            '## Files and reproduction',
            '- Source measurements: [clustering](q1q2_cells.csv), [information](q1q2_information.csv), [supervised](supervised_readouts.csv), [per-unit reuse](unit_reuse.csv), [manifest](manifest.json).',
            '- Complete readable tables: [A information](report_details/A_information.md), [B clustering](report_details/B_clustering.md), [C supervised](report_details/C_supervised.md), [D reuse](report_details/D_unit_reuse.md).',
            '- Derived CSVs: [clustering summaries](report_details/clustering_summary.csv), [information summaries](report_details/information_summary.csv), [supervised summaries](report_details/supervised_summary.csv), [recall](report_details/per_class_recall.csv), [fold class support/recall](report_details/fold_class_support_recall.csv), [folds](report_details/folds.csv), [preprocessing](report_details/preprocessing.csv), [reuse per fold/seed](report_details/reuse_fold_seed_summary.csv).',
            '- Regenerate this report without refitting: `python script/exp2round/q1q2/report_complete.py` in the existing SMQ environment. [Audit and source hashes](report_details/report_audit.json) identify the exact input files.']
    (OUT/'REPORT.md').write_text('\n\n'.join(txt)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
