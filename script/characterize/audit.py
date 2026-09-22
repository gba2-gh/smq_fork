"""Validate frozen manifests, matched pairs, coverage and characterization artifacts."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/characterization"


def rows(name):
    return list(csv.DictReader((OUT/name).open(encoding="utf-8")))


def main():
    manifest = json.loads((OUT/"manifest.json").read_text())
    assert hashlib.sha256((OUT/"PROTOCOL.md").read_bytes()).hexdigest() == manifest["protocol_sha256"]
    for cfg in manifest["datasets"].values():
        assert not set(cfg["calibration_subjects"]) & set(cfg["evaluation_subjects"])
        for cp in cfg["checkpoints"]:
            assert hashlib.sha256((ROOT/cp["checkpoint"]).read_bytes()).hexdigest() == cp["checkpoint_sha256"]
    with np.load(OUT/"lara_cache.npz") as f:
        cache = {k:f[k] for k in f.files}
    groups = defaultdict(list)
    for r in rows("primary_pairs.csv"):
        groups[(r["seed"],r["grouping"],r["subset"])].append(r)
    posture = (cache["trajectory"].mean(1)-cache["posture_center"])/cache["posture_scale"]
    pairing_audit = []
    for (seed, grouping, subset), group in groups.items():
        si = manifest["seeds"].index(int(seed))
        if grouping == "smq":
            codes = cache["codes"][:,si]
        else:
            with np.load(OUT/f"posture_kmeans_s{seed}.npz") as f:
                codes = f["codes"]
        used = []
        for r in group:
            a,b,c = [int(r[k]) for k in ("anchor","same","control")]
            used.extend([a,b,c])
            assert cache["evaluation"][[a,b,c]].all()
            assert codes[a] == codes[b] and codes[a] != codes[c]
            assert cache["subjects"][cache["rec"][a]] != cache["subjects"][cache["rec"][b]]
            assert cache["rec"][b] == cache["rec"][c]
            # Dependency intervals of every selected patch in one recording do not overlap.
            assert abs(cache["start"][b]-cache["start"][c]) >= 78
            ab, ac, bc = [float(np.sqrt(np.mean((posture[x]-posture[y])**2))) for x,y in ((a,b),(a,c),(b,c))]
            assert ab <= .500001 and ac <= .500001 and bc <= .250001 and abs(ab-ac) <= .050001
            assert abs(ab-float(r["posture_same"])) < 1e-6
            assert abs(ac-float(r["posture_control"])) < 1e-6
            for metric in ("displacement","velocity","composite","dtw"):
                assert abs(float(r[f"{metric}_control"])-float(r[f"{metric}_same"])-float(r[f"{metric}_delta"]))<1e-10
            if subset == "different_actions":
                assert cache["action"][a] != cache["action"][b] and cache["action"][a] != cache["action"][c]
        assert len(set(used)) == len(used)
        meaningful = [r for r in group if all(int(r[k]) not in (7,8) for k in ("anchor_action","same_action","control_action"))]
        pure = [r for r in meaningful if float(r["minimum_action_purity"]) >= .8]
        pairing_audit.append(dict(seed=seed,grouping=grouping,subset=subset,n=len(group),
            triples_excluding_none_sync=len(meaningful),triples_excluding_none_sync_purity80=len(pure),
            action_triples=dict(Counter(f"{r['anchor_action']}/{r['same_action']}/{r['control_action']}" for r in group))))
    for r in rows("primary_summary.csv"):
        group = groups[(r["seed"],r["grouping"],r["subset"])]
        assert int(r["n"]) == len(group)
        for field in ("same","control","delta"):
            key = f"{r['metric']}_{field}"
            if r["weighting"] == "frequency":
                estimate = np.mean([float(g[key]) for g in group])
            else:
                estimate = np.mean([np.mean([float(g[key]) for g in group if g["code"] == code])
                                    for code in set(g["code"] for g in group)])
            assert abs(estimate-float(r[field])) < 1e-10
    coverage = rows("primary_coverage.csv")
    for r in coverage:
        assert int(r["anchor_candidates"]) == sum(int(r[k]) for k in ("anchor_already_used","no_same_code","same_outside_caliper","no_control","matched"))
    exemplars = rows("exemplars.csv")
    montage_groups = defaultdict(list)
    for r in exemplars:
        montage_groups[(r["dataset"],r["seed"],r["code"])].append(r)
        assert (ROOT/r["montage"]).exists()
    assert len(montage_groups) == 54
    for rr in montage_groups.values():
        assert len(rr) == 6 and len({r["participant"] for r in rr}) == 6
    validations = {ds:json.loads((OUT/f"stability_{ds}_validation.json").read_text()) for ds in ("lara","hugadb")}
    for v in validations.values():
        assert v["clean_crop_mismatches"] == 0 and v["raw_overlap_max_abs"] == 0
    sources = list((ROOT/"script/characterize").glob("*.py")) + list((ROOT/"src/model").glob("*.py"))
    result = dict(protocol_hash_matches=True,checkpoint_hashes_match=True,
        analysis_source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        primary_analyses_validated=len(groups),primary_triples_validated=sum(map(len,groups.values())),
        primary_summary_rows_validated=len(rows("primary_summary.csv")),
        montage_codes=54,exemplar_occurrences=len(exemplars),
        clean_context_code_checks=sum(v["clean_crop_checks"] for v in validations.values()),
        clean_context_mismatches=0,pairing_coverage=pairing_audit)
    (OUT/"audit.json").write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ("pairing_coverage","analysis_source_sha256")},indent=2))


if __name__ == "__main__":
    main()
