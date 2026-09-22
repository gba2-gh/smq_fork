"""Refresh categorical diagnostics only; codebooks/retrieval remain byte-identical."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('MKL_NUM_THREADS','4')
import sys,time,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from script.vocab.common import OUT
from script.vocab.analyze import load
from script.vocab.run_exp1 import save
from script.vocab.run_exp2 import check_time
from script.vocab import diagnostic as D,readout as M
torch.set_num_threads(4)
archive=OUT/'archive'/'fp32_150'/'code_readouts';archive.mkdir(parents=True,exist_ok=True)
for ds in ('lara','babel1'):
    meta=load(OUT/'exp2'/f'{ds}_meta.pkl');C=len(meta['class_names']);n=len(meta['names'])
    for path in sorted((OUT/'exp2').glob(f'{ds}_*_k*_s*.pkl')):
        cell=load(path)
        if cell['readout'].get('solver')==M.LINEAR_SOLVER:continue
        check_time();t=time.time();f=cell['fold'];rec=cell['rec'];fit=meta['folds'][rec]!=f;ev=~fit
        untouched=hashlib.sha256(cell['centers'].tobytes()+cell['codes'].tobytes()+cell['retrieval']['S'].tobytes()).hexdigest()
        old=archive/path.name
        if not old.exists():save(old,{k:cell[k] for k in ('readout','predictions','conf_segment','conf_duration','class_coverage')})
        eye=np.eye(cell['K'],dtype=np.float32);y=cell['truth'];code=cell['codes']
        ro=D.run(eye[code[fit]],y[fit],meta['groups'][rec[fit]],eye[code[ev]],C,scale=False)
        pred=ro.pop('predictions')[0];cell['readout']=ro;cell['predictions']=pred
        for weighting,weights in [('segment',None),('duration',cell['duration'][ev])]:
            cell['conf_'+weighting]=M.confusion_by_recording(y[ev],pred,rec[ev],n,C,weights)
        cell['class_coverage']=dict(predicted=int(len(np.unique(pred))),recalled=int(len(np.unique(y[ev][pred==y[ev]]))),total=C)
        assert untouched==hashlib.sha256(cell['centers'].tobytes()+cell['codes'].tobytes()+cell['retrieval']['S'].tobytes()).hexdigest()
        cell['readout_refit_seconds']=time.time()-t;cell['discrete_payload_sha256']=untouched
        save(path,cell)
        print('READOUT REFIT',path.name,round(cell['readout_refit_seconds'],1),'iterations',ro['optimization'][0]['iterations'],flush=True)
print('COMPLETE categorical readout correction',flush=True)
