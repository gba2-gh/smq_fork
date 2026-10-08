"""Historical pooled B0 parity (separate from the held-out experiment; its plans never enter any cell).

Re-runs the saved ts5_v1/phase5 LARa `centre` unit (all 439 recordings, preparation grouping, 50 outer iterations, ladder
500/1000/2000/4000) through the original helpers and requires identical stored metrics."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from script.compositional_dynamics import config as C


def run_parity(out_dir: Path, seed: int = 111) -> int:
    from script.action_transport import continuous
    from script.week5.raw_kmeans_cross_dataset import datasets as D
    from script.week5.timescale_inspection import radius as RAD
    from script.week5.timescale_inspection import run_phase1 as R1
    from script.quantized_input_comparison.artifacts import Loader

    t0 = time.perf_counter()
    loader = Loader(C.prep_dir(seed))
    spec = D.SPECS[C.DATASET]
    names = sorted(loader.recording_names())
    lookup, mean = loader.lookup(C.FAMILY), loader.mean(C.FAMILY).astype(np.float64)
    recs, starts = [], {}
    for n in names:
        r = loader.recording(C.FAMILY, n)
        starts[n] = r["starts"]
        z, zero = continuous.l2_normalize(lookup[r["codes"]].astype(np.float64) - mean[None, :])
        recs.append(continuous.ContinuousRecordingInput(n, r["codes"].astype(np.int32), r["lengths"].astype(np.int32),
                                                        spec.fps, spec.window, z, zero))
    mu0, _ = continuous.build_mu0(loader.grouping(C.FAMILY), recs, spec.num_states)
    gts = D.load_gts(spec, names)
    fcs = {n: loader.frame_count(n) for n in names}
    res = R1.fit_and_score(recs, mu0, spec.num_states, R1.coeffs_for(0.3), RAD.CURRENT, 50, R1.RUNGS, gts, starts, fcs)
    saved = json.loads(C.historical_unit(seed).read_text())
    rows, exact, worst = [], True, 0.0
    for new, old in zip(res["ladder"], saved["ladder"]):
        assert new["rung_steps"] == old["rung_steps"]
        for m in ("MoF", "Edit", "F1_10", "F1_25", "F1_50"):
            diff = float(abs(new[m] - old[m]))
            worst, exact = max(worst, diff), bool(exact and diff == 0.0)
            rows.append(dict(rung=new["rung_steps"], metric=m, new=new[m], saved=old[m], abs_diff=diff))
    out = dict(seed=seed, exact_match=exact, max_abs_diff=worst, ot_status=res["ot_status"], saved_ot_status=saved["ot_status"],
               outer_iterations=res["outer_iterations"], saved_outer_iterations=saved["outer_iterations"],
               seconds=time.perf_counter() - t0, rows=rows, code_hash=C.code_hash())
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "parity.json").write_text(json.dumps(out, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(f"parity seed {seed}: exact={exact} max |diff|={worst} ({out['seconds']:.0f}s)")
    if not exact:
        print("NOT EXACT: document the numerical/environment cause; do not widen a tolerance.")
    return 0 if exact else 1
