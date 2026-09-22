"""Wait for primary cells, then run optional context within the same time budget."""
import os,sys,time,subprocess
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results'/'vocab'
deadline=datetime(2026,9,21,20,50,tzinfo=timezone.utc).timestamp()
last=-1
while time.time()<deadline:
    n=len(list((OUT/'exp2').glob('*_k*_s*.pkl')))
    if n!=last:
        print('Primary clustering cells',n,'/324',flush=True);last=n
    if n==324:
        # Exp3 itself asserts that both primary experiments have every expected file.
        with open(OUT/'exp3.log','w') as log:
            result=subprocess.run([sys.executable,'-B','script/vocab/run_exp3.py'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        print('Context process exit',result.returncode,flush=True)
        break
    time.sleep(10)
else:
    print('Time budget: context not launched; primary artifacts retained.',flush=True)
