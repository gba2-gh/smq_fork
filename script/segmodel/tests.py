import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from script.segmodel.core import dp
from script.seg.partition import brute_force_dp

def test_dp_matches_reference():
    c=np.full((7,3),np.inf); rng=np.random.default_rng(1)
    for end in range(1,7): c[end,:min(3,end)]=rng.random(min(3,end))
    e,v=dp(c,.3); ref,edges=brute_force_dp(c,1,3,.3)
    assert np.isclose(v,ref) and np.array_equal(e,np.asarray(edges))
if __name__=='__main__': test_dp_matches_reference();print('PASS')
