"""Finish corrected primary readouts and analyses before restarting context."""
import sys,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results'/'vocab'
jobs=[('script/vocab/run_exp1.py','exp1_refit.log'),('script/vocab/refit_code_readouts.py','code_readout_refit.log')]
processes=[]
for script,logname in jobs:
    log=open(OUT/logname,'w');p=subprocess.Popen([sys.executable,'-B',script],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    processes.append((p,log,script))
for p,log,script in processes:
    code=p.wait();log.close();print('PRIMARY REFIT EXIT',script,code,flush=True)
    if code:raise SystemExit('Primary correction failed; preserve outputs and inspect log.')
for exp in ('1','2'):
    with open(OUT/f'analysis_exp{exp}.log','w') as log:
        p=subprocess.run([sys.executable,'-B','script/vocab/analyze.py','--experiment',exp],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    if p.returncode:raise SystemExit('Primary corrected analysis failed.')
print('CORRECTED PRIMARY COMPARISONS COMPLETE',flush=True)
with open(OUT/'exp3.log','w') as log:
    p=subprocess.run([sys.executable,'-B','script/vocab/run_exp3.py'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
print('CONTEXT EXIT',p.returncode,flush=True)
if p.returncode:raise SystemExit(p.returncode)
