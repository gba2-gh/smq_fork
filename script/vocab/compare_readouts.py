"""Secondary paired diagnostic: oracle continuous versus categorical linear readouts."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('MKL_NUM_THREADS','4')
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from script.vocab.common import OUT
from script.vocab.analyze import load,csvout,readout_stats,interval
from script.seg.stats import make_weights
rows=[]
for ds,nf in [('lara',4),('babel1',5)]:
    meta=load(OUT/'exp1'/f'{ds}_meta.pkl');W=make_weights(meta['groups'])
    for rep in ('mean','eight'):
        continuous=sum(load(OUT/'exp1'/f'{ds}_{rep}_f{f}_linear.pkl')['segment'] for f in range(nf))
        ref=readout_stats(continuous,W)
        for mult in (1,4,16):
            conf=np.stack([sum(load(OUT/'exp2'/f'{ds}_oracle_{rep}_f{f}_k{mult}_s{s}.pkl')['conf_segment'] for f in range(nf)) for s in (0,1,2)])
            np.testing.assert_array_equal(conf.sum(3),np.repeat(continuous.sum(3),3,axis=0))
            quantized=readout_stats(conf,W)
            for metric in ref:
                rows.append(dict(dataset=ds,representation=rep,K_multiple=mult,metric=metric,**interval(ref[metric]-quantized[metric])))
csvout('oracle_readout_gap.csv',rows)
print('Secondary paired oracle readout gaps saved',flush=True)
