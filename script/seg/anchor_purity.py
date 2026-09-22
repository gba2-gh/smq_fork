import sys, pickle, json
sys.path.insert(0, '.')
import numpy as np
from script.seg import pipeline as PL, partition as P
from script.seg.exp1 import build_partitions
out = {}
for ds in ['lara', 'babel1', 'babel2', 'babel3']:
    d = PL.Data(ds); anchors = d.make_anchors()
    fixed, oracle, rand = build_partitions(d)
    e2 = pickle.load(open(f'results/seg/raw/exp2_{ds}.pkl', 'rb'))['meta']['dp_edges']
    parts = dict(fixed=fixed, oracle=oracle, random0=rand[0], dp=e2)
    out[ds] = {}
    for name, ed in parts.items():
        agree = tot = 0; pur = []
        for r in range(d.n):
            cnt = PL.seg_label_counts(d.labels[r], ed[r], d.C)
            sid = np.searchsorted(ed[r], anchors[r], side='right') - 1
            maj = cnt.argmax(1)[sid]
            agree += int((maj == d.labels[r][anchors[r]]).sum()); tot += len(sid)
            pur.extend((cnt.max(1) / cnt.sum(1))[sid].tolist())
        out[ds][name] = dict(anchor_label_equals_segment_majority=agree / tot, mean_segment_purity_at_anchor=float(np.mean(pur)))
json.dump(out, open('results/seg/anchor_purity.json', 'w'), indent=1)
for ds in out:
    print(ds, {k: (round(v['anchor_label_equals_segment_majority'], 3), round(v['mean_segment_purity_at_anchor'], 3)) for k, v in out[ds].items()})
