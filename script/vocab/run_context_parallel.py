"""Run two independent context folds at a time, preserving all diagnostic settings."""
import sys,subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results'/'vocab'

def run(job):
    ds,fold=job
    with open(OUT/f'exp3_{ds}_f{fold}.log','w') as log:
        result=subprocess.run([sys.executable,'-B','script/vocab/run_exp3.py',
                               '--dataset',ds,'--fold',str(fold)],cwd=ROOT,
                              stdout=log,stderr=subprocess.STDOUT)
    print('CONTEXT FOLD EXIT',ds,fold,result.returncode,flush=True)
    return result.returncode

if __name__=='__main__':
    jobs=[(ds,f) for ds,nf in [('lara',4),('babel1',5)] for f in range(nf)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(run,jobs))
    if any(results):raise SystemExit('A context fold failed; inspect logs and resume saved cells.')
    print('COMPLETE all context folds',flush=True)
