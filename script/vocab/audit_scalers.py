"""Independent moment-combination audit of saved outer-fit scalers."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('MKL_NUM_THREADS','4')
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from script.vocab.common import OUT,ROOT
from script.vocab.analyze import load

rows=[]
for ds,nf in [('lara',4),('babel1',5)]:
    mp=OUT/'exp2'/f'{ds}_meta.pkl'
    if not mp.exists():continue
    meta=load(mp);means=[];variances=[];sizes=[]
    for name in meta['names']:
        z=np.load(ROOT/'results'/'seg'/'cache'/ds/f'{name}.npy').astype(np.float64)
        means.append(z.mean(0));variances.append(z.var(0));sizes.append(len(z))
    means=np.array(means);variances=np.array(variances);sizes=np.array(sizes)
    for f in range(nf):
        fit=meta['folds']!=f
        mu=np.average(means[fit],axis=0,weights=sizes[fit])
        variance=np.average(variances[fit]+(means[fit]-mu)**2,axis=0,weights=sizes[fit])
        sd=np.sqrt(variance);sd=np.maximum(sd,1e-3*np.median(sd[sd>0]))
        files=list((OUT/'exp2').glob(f'{ds}_*_f{f}_k*_s*.pkl'))
        for p in files:
            c=load(p)
            np.testing.assert_allclose(c['mu'],mu,rtol=1e-6,atol=1e-7)
            np.testing.assert_allclose(c['sd'],sd,rtol=1e-6,atol=1e-7)
        rows.append(dict(dataset=ds,fold=f,cells_checked=len(files),fit_recordings=int(fit.sum()),
                         eval_recordings=int((~fit).sum()),status='passed'))
    print('Independent fitting-only scaler audit passed:',ds,flush=True)
(OUT/'SCALER_AUDIT.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
