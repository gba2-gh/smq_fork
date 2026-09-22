"""Posture-controlled, cross-person movement matching under the frozen protocol."""
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
import numpy as np
from numba import njit
from sklearn.cluster import MiniBatchKMeans

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from script.characterize.prepare import descriptors, SEEDS

OUT = ROOT / "results/characterization"


@njit(cache=False)
def dtw(ad, av, bd, bv, band=5):
    n, dim = ad.shape
    cost = np.full((n + 1, n + 1), np.inf)
    length = np.zeros((n + 1, n + 1), np.int64)
    cost[0, 0] = 0.
    for i in range(1, n + 1):
        for j in range(max(1, i-band), min(n, i+band)+1):
            local = 0.
            for k in range(dim):
                local += .5*((ad[i-1,k]-bd[j-1,k])**2+(av[i-1,k]-bv[j-1,k])**2)/dim
            best = cost[i-1,j-1]
            count = length[i-1,j-1]
            if cost[i-1,j] < best:
                best, count = cost[i-1,j], length[i-1,j]
            if cost[i,j-1] < best:
                best, count = cost[i,j-1], length[i,j-1]
            cost[i,j] = best + local
            length[i,j] = count + 1
    return np.sqrt(cost[n,n]/length[n,n])


def fixed(a, b):
    return float(np.sqrt(np.mean((a-b)**2)))


def metric_checks():
    t = np.linspace(0., 1., 50)
    forward = np.stack((t, .4*t, -.3*t), -1)
    reverse = forward[::-1].copy()
    warp = t + .04*np.sin(2*np.pi*t)
    slower = np.stack((warp, .4*warp, -.3*warp), -1)
    # All have equal mean posture; center explicitly to remove roundoff.
    xx = np.stack([x-x.mean(0) for x in (forward, reverse, slower)])
    d, v = descriptors(xx, np.ones(3), 50)
    vals = {}
    for label, i in (("identical", 0), ("opposite_direction", 1), ("modest_speed_variation", 2)):
        vals[label] = dict(posture_distance=fixed(xx[0].mean(0), xx[i].mean(0)),
            displacement=fixed(d[0], d[i]), velocity=fixed(v[0], v[i]),
            composite=.5*(fixed(d[0],d[i])+fixed(v[0],v[i])),
            dtw=float(dtw(d[0],v[0],d[i],v[i])))
    for metric in ("displacement", "velocity", "composite", "dtw"):
        assert vals["identical"][metric] == 0
        assert vals["modest_speed_variation"][metric] < vals["opposite_direction"][metric]
    vals["passed"] = True
    (OUT / "metric_checks.json").write_text(json.dumps(vals, indent=2))
    print("Controlled equal-posture movement checks passed", flush=True)


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def point_and_ci(rows, metric, weighting, rng_seed=4321):
    # Resample whole disjoint participant dyads, retaining both endpoints and all recordings.
    sums = np.zeros((5,8,3))
    counts = np.zeros((5,8))
    for r in rows:
        dy, code = r["dyad"], r["code"]
        sums[dy,code] += [r[f"{metric}_same"], r[f"{metric}_control"], r[f"{metric}_delta"]]
        counts[dy,code] += 1

    def estimate(mult):
        ss = np.einsum("bd,dkm->bkm", mult, sums)
        cc = mult @ counts
        if weighting == "frequency":
            total = cc.sum(1)
            val = np.divide(ss.sum(1), total[:,None], out=np.full((len(mult),3),np.nan), where=total[:,None]>0)
        else:
            by_code = np.divide(ss, cc[:,:,None], out=np.full_like(ss,np.nan), where=cc[:,:,None]>0)
            valid = (cc>0).sum(1)
            val = np.divide(np.nansum(by_code,1),valid[:,None],out=np.full((len(mult),3),np.nan),where=valid[:,None]>0)
        return val
    point = estimate(np.ones((1,5)))[0]
    rng = np.random.default_rng(rng_seed)
    multiplicities = rng.multinomial(5, np.full(5,.2), size=2000)
    bs = estimate(multiplicities)[:,2]
    valid = bs[np.isfinite(bs)]
    interval = np.percentile(valid,[2.5,97.5]) if len(valid) else [np.nan,np.nan]
    return dict(same=float(point[0]), control=float(point[1]), delta=float(point[2]),
                ci_low=float(interval[0]), ci_high=float(interval[1]), bootstrap_valid=len(valid))


def main():
    metric_checks()  # Must pass before grouping outcomes are computed.
    with np.load(OUT / "lara_cache.npz") as archive:
        d = {k: archive[k] for k in archive.files}
    manifest = json.loads((OUT/"manifest.json").read_text())
    dyads = manifest["datasets"]["lara"]["primary_dyads"]
    assert len(dyads) == 5
    rec, starts = d["rec"], d["start"]
    subjects = d["subjects"][rec]
    posture = ((d["trajectory"].mean(1)-d["posture_center"])/d["posture_scale"]).astype(np.float32)
    disp, vel = descriptors(d["trajectory"], d["posture_scale"], 50)
    ds, vs = d["distance_scales"]
    dispn, veln = disp/ds, vel/vs
    partner, dyad_of = {}, {}
    for i, (a,b) in enumerate(dyads):
        partner[a], partner[b] = b,a
        dyad_of[a] = dyad_of[b] = i
    eligible = np.flatnonzero(d["evaluation"] & np.isin(subjects, list(partner)))
    anchors = []
    for recording in np.unique(rec[eligible]):
        ids = eligible[rec[eligible] == recording]
        anchors.extend(ids[np.linspace(0,len(ids)-1,min(12,len(ids))).astype(int)])
    anchors = sorted(anchors, key=lambda a: hashlib.sha256(
        f"anchor/{d['names'][rec[a]]}/{starts[a]}".encode()).hexdigest())
    donor_sets = {s: np.flatnonzero(subjects == s) for s in partner}
    record_sets = {r: np.flatnonzero(rec == r) for r in np.unique(rec[eligible])}
    summaries, coverage, per_code, all_pairs = [], [], [], []
    for si, seed in enumerate(SEEDS):
        km = MiniBatchKMeans(n_clusters=8, random_state=seed,n_init=10,max_iter=200,batch_size=2048)
        km.fit(posture[~d["evaluation"]])
        kmcodes = km.predict(posture)
        np.savez_compressed(OUT/f"posture_kmeans_s{seed}.npz",centers=km.cluster_centers_,codes=kmcodes)
        for grouping, labels in (("smq",d["codes"][:,si]),("posture_kmeans",kmcodes)):
            for subset in ("all_actions","different_actions"):
                used = np.zeros(len(rec),bool)
                rows = []
                reasons = dict(anchor_already_used=0,no_same_code=0,same_outside_caliper=0,
                               no_control=0,matched=0)
                attempts_per_code = np.bincount(labels[anchors],minlength=8)
                for a in anchors:
                    code = int(labels[a])
                    if used[a]:
                        reasons["anchor_already_used"] += 1; continue
                    candidates = donor_sets[partner[subjects[a]]]
                    candidates = candidates[(labels[candidates] == code) & ~used[candidates]]
                    if subset == "different_actions":
                        candidates = candidates[d["action"][candidates] != d["action"][a]]
                    if not len(candidates):
                        reasons["no_same_code"] += 1; continue
                    pdist = np.sqrt(np.mean((posture[candidates]-posture[a])**2,1))
                    j = int(np.argmin(pdist)); b = int(candidates[j]); pab = float(pdist[j])
                    if pab > .5:
                        reasons["same_outside_caliper"] += 1; continue
                    controls = record_sets[rec[b]]
                    controls = controls[(labels[controls] != code) & ~used[controls]]
                    if subset == "different_actions":
                        controls = controls[d["action"][controls] != d["action"][a]]
                    pac = np.sqrt(np.mean((posture[controls]-posture[a])**2,1))
                    pbc = np.sqrt(np.mean((posture[controls]-posture[b])**2,1))
                    keep = (pbc <= .25) & (pac <= .5) & (np.abs(pac-pab) <= .05)
                    if not keep.any():
                        reasons["no_control"] += 1; continue
                    controls, pac, pbc = controls[keep], pac[keep], pbc[keep]
                    j = int(np.argmin(pbc)); c = int(controls[j])
                    used[[a,b,c]] = True
                    reasons["matched"] += 1
                    row = dict(dataset="lara",seed=seed,grouping=grouping,subset=subset,
                        dyad=dyad_of[subjects[a]],code=code,anchor=int(a),same=b,control=c,
                        anchor_record=str(d["names"][rec[a]]), donor_record=str(d["names"][rec[b]]),
                        anchor_subject=str(subjects[a]),donor_subject=str(subjects[b]),
                        anchor_start=int(starts[a]),same_start=int(starts[b]),control_start=int(starts[c]),
                        anchor_action=int(d["action"][a]),same_action=int(d["action"][b]),control_action=int(d["action"][c]),
                        minimum_action_purity=float(d["purity"][[a,b,c]].min()),
                        posture_same=pab,posture_control=float(pac[j]),posture_balance=float(pac[j]-pab),
                        posture_partner_control=float(pbc[j]))
                    for metric, feature in (("displacement",disp),("velocity",vel)):
                        x,y = fixed(feature[a],feature[b]),fixed(feature[a],feature[c])
                        row[f"{metric}_same"],row[f"{metric}_control"],row[f"{metric}_delta"] = x,y,y-x
                    for side in ("same","control"):
                        row[f"composite_{side}"] = .5*(row[f"displacement_{side}"]/ds + row[f"velocity_{side}"]/vs)
                    row["composite_delta"] = row["composite_control"]-row["composite_same"]
                    x = float(dtw(dispn[a],veln[a],dispn[b],veln[b]))
                    y = float(dtw(dispn[a],veln[a],dispn[c],veln[c]))
                    row["dtw_same"],row["dtw_control"],row["dtw_delta"] = x,y,y-x
                    # Guard confounds/independence and fixed matching rules.
                    assert subjects[a] != subjects[b] and rec[b] == rec[c]
                    assert labels[a] == labels[b] and labels[a] != labels[c]
                    assert abs(starts[b]-starts[c]) >= 2*50
                    rows.append(row)
                prefix = dict(dataset="lara",seed=seed,grouping=grouping,subset=subset)
                coverage.append(dict(**prefix,eligible_pool=len(eligible),anchor_candidates=len(anchors),
                    **reasons,matched_fraction=len(rows)/len(anchors),
                    participant_dyads=len(set(r["dyad"] for r in rows)),
                    recordings=len(set(r[k] for r in rows for k in ("anchor_record","donor_record"))),
                    codes_represented=len(set(r["code"] for r in rows)),
                    posture_same_mean=float(np.mean([r["posture_same"] for r in rows])) if rows else np.nan,
                    posture_control_mean=float(np.mean([r["posture_control"] for r in rows])) if rows else np.nan,
                    posture_balance_mean=float(np.mean([r["posture_balance"] for r in rows])) if rows else np.nan,
                    posture_balance_abs_max=max([abs(r["posture_balance"]) for r in rows],default=np.nan),
                    action_purity_mean=float(np.mean([r["minimum_action_purity"] for r in rows])) if rows else np.nan))
                for weighting in ("frequency","code_balanced"):
                    for metric in ("displacement","velocity","composite","dtw"):
                        summaries.append(dict(**prefix,weighting=weighting,metric=metric,n=len(rows),
                                              **point_and_ci(rows,metric,weighting)))
                for code in range(8):
                    cr = [r for r in rows if r["code"] == code]
                    for metric in ("displacement","velocity","composite","dtw"):
                        per_code.append(dict(**prefix,code=code,metric=metric,
                            evaluation_occurrences=int(np.sum((labels==code)&d["evaluation"])),
                            anchor_candidates=int(attempts_per_code[code]),matched=len(cr),
                            participant_dyads=len(set(r["dyad"] for r in cr)),
                            **point_and_ci(cr,metric,"frequency")))
                all_pairs.extend(rows)
                print(f"{seed} {grouping} {subset}: {len(rows)}/{len(anchors)} matched",flush=True)
                write_csv(OUT/"primary_summary.csv",summaries)
                write_csv(OUT/"primary_coverage.csv",coverage)
                write_csv(OUT/"primary_per_code.csv",per_code)
                write_csv(OUT/"primary_pairs.csv",all_pairs)
    print("Primary analysis complete",flush=True)


if __name__ == "__main__":
    main()
