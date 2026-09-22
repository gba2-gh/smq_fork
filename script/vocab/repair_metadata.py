"""Restore per-item duration metadata overwritten by the duration confusion field.

No predictions or evaluation matrices change. Retained for a transparent audit trail.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from script.vocab.common import OUT,ROOT,PL,P
from script.vocab.analyze import load
from script.vocab.run_exp1 import save
for ds in ('lara','babel1'):
    meta=load(OUT/'exp1'/f'{ds}_meta.pkl')
    mapping=PL.read_mapping(ds);a2i={v:k for k,v in mapping.items()}
    durations=[np.diff(P.oracle_edges(PL.load_gt_labels(ROOT/'data'/ds,n,a2i))) for n in meta['names']]
    for path in (OUT/'exp1').glob(f'{ds}_*_f*_*.pkl'):
        cell=load(path)
        if 'item_duration' in cell:continue
        expected=np.concatenate([d for r,d in enumerate(durations) if meta['folds'][r]==cell['fold']])
        assert len(expected)==len(cell['truth'])
        assert cell['duration'][0].sum()==expected.sum()
        cell['item_duration']=expected
        save(path,cell)
    print('Duration metadata restored:',ds)
