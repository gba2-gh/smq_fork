"""Three independent outer-fold processes; no change to fitting or evaluation."""
import os,sys,subprocess,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results'/'vocab'

def run(job):
    ds,fold=job;path=OUT/f'exp2_{ds}_f{fold}.log'
    with open(path,'w') as log:
        p=subprocess.run([sys.executable,'-B','script/vocab/run_exp2.py','--dataset',ds,'--fold',str(fold)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    print('FOLD EXIT',ds,fold,p.returncode,flush=True)
    return p.returncode

if __name__=='__main__':
    jobs=[(ds,f) for ds,nf in [('lara',4),('babel1',5)] for f in range(nf)]
    with ThreadPoolExecutor(max_workers=3) as pool:
        results=list(pool.map(run,jobs))
    if any(results):raise SystemExit('A fold failed; inspect fold logs and resume completed cells.')
    print('COMPLETE all primary clustering folds',flush=True)
