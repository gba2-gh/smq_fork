"""Build the LARa-only inspection workspace from saved predictions and input arrays.

Run: python script/visual_audit/lara.py
No model fitting, transport solve, or checkpoint selection occurs here.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from script.visual_audit import build as old
from script.week5.raw_kmeans_cross_dataset import datasets as D
from script.repro.d_error_decomposition import score
from script.exp2round.q1q2 import core as Q

OUT = ROOT / "results/visual_audit/lara_001"
BASE = old.OUT
SEEDS = old.SEEDS
FPS = 50
WINDOW = 12
SOURCE = ROOT / "data/lara"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict], columns=None) -> None:
    cols = columns or list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def windows_to_frames(values: np.ndarray, lengths: np.ndarray, count: int) -> np.ndarray:
    assert len(values) == len(lengths) and int(lengths.sum()) == count
    assert np.all(lengths > 0) and np.all(lengths[:-1] == WINDOW)
    assert int(lengths[-1]) <= WINDOW
    return np.repeat(values, lengths)


def boundaries(x: np.ndarray) -> np.ndarray:
    return np.flatnonzero(x[1:] != x[:-1]) + 1


def match_boundaries(gt: np.ndarray, pred: np.ndarray, tolerance: int) -> dict[int, int]:
    """Maximum one-to-one matches, then minimum total absolute offset.

    Sorted boundary positions permit order-preserving dynamic programming.
    Remaining ties prefer a match, then skipping a predicted boundary.
    """
    n, m = len(gt), len(pred)
    counts = np.zeros((n + 1, m + 1), dtype=np.int16)
    costs = np.zeros((n + 1, m + 1), dtype=np.int32)
    choice = np.zeros((n, m), dtype=np.uint8)  # 0 skip GT, 1 skip pred, 2 match
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            best = (int(counts[i+1, j]), -int(costs[i+1, j]), 0)
            skip_pred = (int(counts[i, j+1]), -int(costs[i, j+1]), 1)
            if skip_pred > best:
                best = skip_pred
            delta = abs(int(gt[i]) - int(pred[j]))
            if delta <= tolerance:
                match = (1 + int(counts[i+1, j+1]), -delta - int(costs[i+1, j+1]), 2)
                if match > best:
                    best = match
            counts[i, j], costs[i, j], choice[i, j] = best[0], -best[1], best[2]
    out = {}
    i = j = 0
    while i < n and j < m:
        c = int(choice[i, j])
        if c == 2:
            out[i] = j
            i += 1
            j += 1
        elif c == 1:
            j += 1
        else:
            i += 1
    return out


def verify_boundaries() -> None:
    assert match_boundaries(np.array([10, 20]), np.array([9, 11, 20]), 2) == {0: 0, 1: 2}
    assert match_boundaries(np.array([10, 20]), np.array([9, 11, 20]), 0) == {1: 2}
    assert match_boundaries(np.array([10]), np.array([], dtype=int), 12) == {}
    assert len(match_boundaries(np.array([10, 20]), np.array([11, 19]), 1)) == 2


def select_recordings(payload) -> list[dict]:
    """Freeze example selection using only seed-111 T2 B0 vs ordered GRU."""
    cells = {(c["arm"], c["seed"]): c for c in payload["cells"]}
    base = cells[("T2 B0", 111)]["records"]
    ordered = cells[("Ordered GRU", 111)]["records"]
    names = sorted(base)
    ranking = sorted(names, key=lambda n: (base[n]["metrics"]["F1@50"], n))
    best = sorted(names, key=lambda n: (-base[n]["metrics"]["F1@50"], n))
    delta = lambda n: ordered[n]["metrics"]["F1@50"]-base[n]["metrics"]["F1@50"]
    improvement = sorted(names, key=lambda n: (-delta(n), n))
    decline = sorted(names, key=lambda n: (delta(n), n))
    roles = [("worst_reference_F1", ranking),
             ("median_reference_F1", sorted(names, key=lambda n: (
                 abs(base[n]["metrics"]["F1@50"] - np.median([base[x]["metrics"]["F1@50"] for x in names])), n))),
             ("best_reference_F1", best),
             ("largest_GRU_gain_F1", improvement),
             ("largest_GRU_decline_F1", decline)]
    used, rows = set(), []
    for role, candidates in roles:
        name = next(n for n in candidates if n not in used)
        used.add(name)
        rows.append(dict(kind="recording", reason=role, recording=name, start_frame=0,
                         end_frame=payload["recordings"][name]["frames"], action=""))
    rng = np.random.default_rng(111)
    for name in rng.choice(sorted(set(names)-used), min(3, len(names)-len(used)), replace=False):
        name = str(name)
        rows.append(dict(kind="recording", reason="uniform_random_seed111", recording=name,
                         start_frame=0, end_frame=payload["recordings"][name]["frames"], action=""))
    for action_id, action_name in payload["labels"].items():
        candidates = []
        for name in names:
            for start, end, c in payload["recordings"][name]["gt"]:
                if c == int(action_id):
                    candidates.append((name, start, end))
        rng = np.random.default_rng(111)
        rng.shuffle(candidates)
        seen = set()
        for name, start, end in candidates:
            if name in seen:
                continue
            seen.add(name)
            rows.append(dict(kind="segment", reason="action_stratified_seed111", recording=name,
                             start_frame=start, end_frame=end, action=action_name))
            if len(seen) == 3:
                break
    return rows


def atlas(payload) -> tuple[list[dict], list[dict], list[dict]]:
    records, actions, transitions = [], [], []
    for cell in payload["cells"]:
        arm, seed = cell["arm"], cell["seed"]
        gt_all = {n: np.repeat([v for _, _, v in r["gt"]],
                    [e-s for s,e,_ in r["gt"]]).astype(np.int32)
                  for n,r in payload["recordings"].items()}
        mapped = {n: np.repeat([v for _,_,v in cell["records"][n]["pred"]],
                   [e-s for s,e,_ in cell["records"][n]["pred"]]).astype(np.int32)
                  for n in gt_all}
        for n in sorted(gt_all):
            g,p = gt_all[n],mapped[n]
            assert len(g)==len(p)==payload["recordings"][n]["frames"]
            m=cell["records"][n]["metrics"]
            for tol in (0, 5, 12, 25):
                bc=old.boundary_counts(g,p,tol)
                records.append(dict(arm=arm,seed=seed,recording=n,subject=payload["recordings"][n]["subject"],
                     tolerance_frames=tol,tolerance_seconds=tol/FPS,frames=len(g),
                     MoF=m["MoF"],F1_50=m["F1@50"],gt_runs=m["gt_segments"],pred_runs=m["predicted_segments"],
                     matched=bc["matched"],extra=bc["extra"],missed=bc["missed"]))
            for a,name in payload["labels"].items():
                a=int(a);truth=g==a;pred=p==a
                actions.append(dict(arm=arm,seed=seed,recording=n,subject=payload["recordings"][n]["subject"],
                    action=name,gt_frames=int(truth.sum()),pred_frames=int(pred.sum()),
                    correct_frames=int((truth&pred).sum()),gt_segments=sum(x[2]==a for x in payload["recordings"][n]["gt"])))
            gr=payload["recordings"][n]["gt"]
            pb=boundaries(p)
            gt_bound=np.asarray([x[1] for x in gr[:-1]],dtype=np.int32)
            paired=match_boundaries(gt_bound,pb,12)
            assert len(paired)==old.boundary_counts(g,p,12)["matched"]
            for gi,(left,right) in enumerate(zip(gr[:-1],gr[1:])):
                t=left[1]; assert t==right[0]
                offset=int(pb[paired[gi]]-t) if gi in paired else None
                transitions.append(dict(arm=arm,seed=seed,recording=n,subject=payload["recordings"][n]["subject"],
                    from_action=payload["labels"][str(left[2])],to_action=payload["labels"][str(right[2])],
                    gt_frame=t,matched_024s=gi in paired,matched_offset_frames=offset,
                    predicted_left=payload["labels"][str(int(p[t-1]))],
                    predicted_right=payload["labels"][str(int(p[t]))]))
    return records, actions, transitions


def load_cell_costs(cell, names):
    from script.compositional_dynamics import config as CD
    from script.input_representation import config as IC
    arm,seed=cell["arm"],int(cell["seed"])
    if arm=="T2 B0" or arm in ("Ordered GRU","Bag","Occupancy"):
        variant="t2"
    else:
        variant="t2_bodyscale" if arm=="Body-scale B0" else "t2_centered"
    prep=IC.prep_dir(variant,seed)
    fam=IC.family(variant)
    folder=ROOT/cell["source"]
    info=json.loads((folder/"info.json").read_text())
    assert sorted(info["eval_names"])==names
    assert info.get("status")=="complete" and not info.get("smoke")
    with np.load(folder/"costs_eval.npz",allow_pickle=False) as z:
        costkey="cost_cosine" if "cost_cosine" in z else "cost_scaled"
        cost=z[costkey].copy()
        codes=z["codes"].copy()
        lengths=z["lengths"].copy()
        recidx=z["rec_idx"].copy() if "rec_idx" in z else None
    assert len(cost)==len(codes)==len(lengths)
    if recidx is not None:
        assert np.array_equal(recidx,np.repeat(np.arange(len(names)),[int((recidx==i).sum()) for i in range(len(names))]))
    return dict(arm=arm,seed=seed,prep=prep,family=fam,folder=folder,source=cell["source"],
                cost=cost,codes=codes,lengths=lengths,eval_names=info["eval_names"],costkey=costkey,
                offsets=np.r_[0,np.cumsum([int(np.ceil(D.frame_counts(D.SPECS['lara'],[n])[n]/WINDOW)) for n in info["eval_names"]])])


def motion(name, expected_frames):
    path=SOURCE/"features"/name
    arr=np.load(path,mmap_mode="r")
    if arr.shape!=(6,expected_frames,22,1):
        raise ValueError(f"Unexpected LARa source layout: {name}: {arr.shape}")
    pos=np.asarray(arr[3:6,:,:,0].transpose(1,2,0),dtype=np.float32)
    if not np.isfinite(pos).all() or not np.allclose(pos[:,21,:],0,atol=1e-3):
        raise ValueError(f"Invalid hip-relative positions: {name}")
    # Every second source frame is displayed. Predictions and GT retain all 50 Hz frames.
    view=np.around(pos[::2],1)
    radius=float(np.max(np.abs(pos)))
    axis_radius=np.max(np.abs(pos),axis=(0,1)).tolist()
    return dict(frames=view.tolist(), stride=2, channels=[3,4,5], joints=22,
                hip=21, radius=round(radius,2), axis_radius=[round(float(x),2) for x in axis_radius],
                source=path.relative_to(ROOT).as_posix(),sha256=digest(path))


def historical_lara(names: list[str], gts: dict[str,np.ndarray]):
    """Pooled historical predictions; maps are fitted over their original 439 recordings."""
    predroot=ROOT/"results/preds/lara_pretrained.npz"
    with np.load(predroot,allow_pickle=True) as z:
        full_names=[str(x) for x in z["names"]]
        full_gt={n:np.asarray(x,dtype=np.int32) for n,x in zip(full_names,z["gt"])}
        smq={n:np.asarray(x,dtype=np.int32) for n,x in zip(full_names,z["pred"])}
    assert len(full_names)==439 and set(names).issubset(full_names)
    assert all(np.array_equal(gts[n],full_gt[n]) for n in names)
    loaded_gt=D.load_gts(D.SPECS["lara"],full_names)
    assert all(np.array_equal(full_gt[n],loaded_gt[n]) for n in full_names)
    specs=[dict(name="Released SMQ",seed="released",kind="frames",path=predroot)]
    stage=ROOT/"results/action_transport_stage_c/stage_c_pooled_001/predictions"
    phase7=ROOT/"results/week5/timescale_inspection/ts5_v1/phase7/s0"
    phase9=ROOT/"results/week5/timescale_inspection/ts5_v2/phase9/units"
    for seed in SEEDS:
        for label,arm in (("Stage C context only","context_only"),("Stage C contextual ASOT","contextual_asot")):
            specs.append(dict(name=label,seed=seed,kind="stage_c",path=stage/f"lara_raw_pooled_pooled_seed{seed}_{arm}__E.npz"))
        specs.append(dict(name="Short mask T2",seed=seed,kind="frames",path=phase7/
            f"phase6__lara__t2_relational__seed{seed}__span2__context/predictions.npz"))
        specs.append(dict(name="Phase 9 continuous I1",seed=seed,kind="frames",path=phase9/
            f"lara__i1__seed{seed}__ig/predictions.npz"))
    historical={n:[] for n in names}; registry=[]
    for item in specs:
        path=item["path"]
        if item["name"]=="Released SMQ":
            raw=smq
        elif item["kind"]=="stage_c":
            with np.load(path,allow_pickle=True) as z:
                z_names=[str(x) for x in z["names"]]
                assert z_names==full_names and str(z["order"])=="original_timeline"
                states=z["states"]
                raw={n:np.repeat(np.asarray(s,dtype=np.int32),WINDOW)[:len(full_gt[n])]
                     for n,s in zip(z_names,states)}
                assert all(len(raw[n])==len(full_gt[n]) for n in full_names)
                saved_mapping={int(k):int(v) for k,v in json.loads(str(z["mapping"]))["all"].items()}
        else:
            with np.load(path,allow_pickle=False) as z:
                assert sorted(z.files)==sorted(full_names)
                raw={n:z[n].astype(np.int32) for n in full_names}
        full_g=[full_gt[n] for n in full_names]
        full_r=[raw[n] for n in full_names]
        if item["kind"]=="stage_c":
            mapping=saved_mapping
            mapped=[np.asarray([mapping[int(v)] for v in r],dtype=np.int32) for r in full_r]
        else:
            mapped=Q.map_predictions(full_g,full_r,"hungarian")
            raw_all=np.concatenate(full_r);mapped_all=np.concatenate(mapped)
            pairs=np.unique(np.stack((raw_all,mapped_all),axis=1),axis=0)
            mapping={int(a):int(b) for a,b in pairs}
        measured=score(full_g,mapped)
        if item["name"]=="Phase 9 continuous I1":
            info=json.loads((path.parent/"info.json").read_text())
            ref=info["ladder"][-1]
            assert abs(measured["MoF"]-ref["MoF"])<.011
            assert abs(measured["F1@50"]-ref["F1_50"])<.011
        if item["name"]=="Short mask T2":
            info=json.loads((path.parent/"info.json").read_text())
            ref=info["ladder"][-1]
            assert abs(measured["MoF"]-ref["MoF"])<.011
            assert abs(measured["F1@50"]-ref["F1_50"])<.011
        if item["name"]=="Released SMQ":
            assert abs(measured["F1@50"]-16.39)<.011
        for n,r,m in zip(full_names,full_r,mapped):
            if n not in historical:continue
            historical[n].append(dict(name=item["name"],seed=item["seed"],raw=old.runs(r),
                                      pred=old.runs(m),metrics=score([full_gt[n]],[m])))
        registry.append(dict(arm=item["name"],seed=item["seed"],source=path.relative_to(ROOT).as_posix(),
            source_sha256=digest(path),population="439 pooled recordings",mapping_scope="global 439-recording evaluation",
            MoF=measured["MoF"],F1_50=measured["F1@50"],status="verified" if item["name"]!="Stage C context only" and item["name"]!="Stage C contextual ASOT" else "saved mapping applied"))
        print(f"Historical {item['name']} seed {item['seed']}: MoF {measured['MoF']}, F1@50 {measured['F1@50']}",flush=True)
    return historical,registry


def build():
    old.verify_helpers()
    verify_boundaries()
    if not (BASE/"audit.json").exists():
        old.main()
    payload=json.loads((BASE/"audit.json").read_text(encoding="utf-8"))
    assert payload["dataset"]=="LARa" and len(payload["cells"])==18
    names=sorted(payload["recordings"])
    gts=D.load_gts(D.SPECS["lara"],names)
    historical,history_registry=historical_lara(names,gts)
    OUT.mkdir(parents=True,exist_ok=True)
    selection=select_recordings(payload)
    write_csv(OUT/"inspection_selection.csv",selection)
    observations=[dict(**r,seed=111,arm="",motion_available=True,direct_observation="",
         candidate_explanation="",alternative_explanation="",other_seed_consistency="",
         reviewer="",status="pending") for r in selection]
    write_csv(OUT/"observations.csv",observations)
    record_rows,action_rows,transition_rows=atlas(payload)
    write_csv(OUT/"recordings.csv",record_rows)
    write_csv(OUT/"actions.csv",action_rows)
    write_csv(OUT/"transitions.csv",transition_rows)
    # Preserve the original exact per-GT-segment diagnostics and source hashes.
    for name in ("segments.csv","classes.csv"):
        (OUT/name).write_bytes((BASE/name).read_bytes())
    sources=[]
    cells={}
    for c in payload["cells"]:
        x=load_cell_costs(c,names)
        cells[(c["arm"],int(c["seed"]))]=x
        sources.append(dict(arm=c["arm"],seed=c["seed"],path=c["source"],
            score_sha256=digest(x["folder"]/"scores.json"),
            prediction_sha256=digest(x["folder"]/"frame_predictions_eval.npz"),
            costs_sha256=digest(x["folder"]/"costs_eval.npz"),
            preparation=x["prep"].relative_to(ROOT).as_posix()))
    template=Path(__file__).with_name("lara_recording.html").read_text(encoding="utf-8")
    recdir=OUT/"recordings";recdir.mkdir(exist_ok=True)
    records_meta={}
    for idx,name in enumerate(names):
        gt=payload["recordings"][name]
        frame_count=gt["frames"]
        source=motion(name,frame_count)
        layers=[]
        for c in payload["cells"]:
            x=cells[(c["arm"],int(c["seed"]))]
            i=x["eval_names"].index(name)
            start,end=map(int,x["offsets"][i:i+2])
            lengths=x["lengths"][start:end]
            codes=x["codes"][start:end]
            cc=x["cost"][start:end]
            assert len(cc)==len(codes)==len(lengths)==int(np.ceil(frame_count/WINDOW))
            prepfile=x["prep"]/"families"/x["family"]/"recordings"/(name+".npz")
            with np.load(prepfile,allow_pickle=False) as p:
                assert np.array_equal(codes,p["codes"]) and np.array_equal(lengths,p["lengths"])
                assert np.array_equal(p["starts"],np.r_[0,np.cumsum(lengths[:-1])])
            local=np.argmin(cc,axis=1).astype(np.int32)
            local_frames=windows_to_frames(local,lengths,frame_count)
            # The stored final mapping is fixed while viewing local costs.
            mapping={int(k):int(v) for k,v in c["mapping"].items()}
            local_label=np.asarray([mapping.get(int(v),-1) for v in local_frames],dtype=np.int32)
            saved=c["records"][name]
            raw=np.repeat([v for _,_,v in saved["raw"]],[e-s for s,e,_ in saved["raw"]])
            assert len(raw)==frame_count
            assert np.array_equal(raw,windows_to_frames(raw[::WINDOW],lengths,frame_count))
            layers.append(dict(arm=c["arm"],seed=c["seed"],raw=saved["raw"],
                pred=saved["pred"],oracle=saved["oracle"],
                local=old.runs(local_label),local_raw=old.runs(local_frames),
                token=old.runs(windows_to_frames(codes,lengths,frame_count)),
                local_cost_key=x["costkey"],mapping=c["mapping"],metrics=saved["metrics"]))
        record=dict(name=name,subject=gt["subject"],fps=FPS,window=WINDOW,gt=gt["gt"],
                    frames=frame_count,motion=source,layers=layers,historical=historical[name],labels=payload["labels"],
                    available_recordings=names,selected=[r for r in selection if r["recording"]==name])
        safe=json.dumps(record,separators=(",",":"),allow_nan=False).replace("<","\\u003c")
        path=recdir/(Path(name).stem+".html")
        path.write_text(template.replace("__AUDIT_DATA__",safe),encoding="utf-8")
        records_meta[name]=dict(page=path.relative_to(OUT).as_posix(),subject=gt["subject"],
                                frames=frame_count,motion_sha256=source["sha256"],
                                page_sha256=digest(path))
        print(f"[{idx+1}/{len(names)}] {name}: aligned motion, tokens and 18 predictions",flush=True)
    # Aggregate index: lean, sortable table; recording pages hold motion lazily.
    compact=dict(dataset="LARa",fps=FPS,names=names,records=records_meta,
                 selected=selection,historical=history_registry,cells=[dict(arm=c["arm"],seed=c["seed"],
                    metrics=c["metrics"],record_metrics={n:c["records"][n]["metrics"] for n in names},
                    recall=c["recall"],confusion=c["confusion"]) for c in payload["cells"]],
                 labels=payload["labels"])
    index_template=Path(__file__).with_name("lara_index.html").read_text(encoding="utf-8")
    (OUT/"index.html").write_text(index_template.replace("__AUDIT_DATA__",json.dumps(compact,separators=(",",":"))),encoding="utf-8")
    manifest=dict(dataset="lara",population="same 44 primary evaluation recordings across six arms and three seeds",
        protocol="held-out decoder evaluation; transductive tokenizer; subject overlap; global evaluation Hungarian mapping",
        fps=FPS,motion=dict(format="(6, T, 22, 1)",channels=[3,4,5],root_joint=21,
              display_stride_frames=2,source="src/data/lara.py; source features are hip-relative"),
        sources=sources,recordings=records_meta,verified=dict(cells=18,recordings=len(names),
              original_scores="verified in source audit",cost_codes="equal saved preparation",terminal_windows="checked"),
        implementation={p.name:digest(p) for p in [Path(__file__),Path(__file__).with_name("lara_index.html"),
                                                  Path(__file__).with_name("lara_recording.html"),
                                                  Path(__file__).with_name("report_lara.py")]})
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    # Grouping completed, but retained evaluation states/plans are absent.
    missing=[dict(run="unit_grouping/ug_lara_001",artifact="evaluation frame/window predictions",
                  status="unavailable",reason="only aggregate scores, fitting plans and prototypes saved; no evaluation states or costs")]
    write_csv(OUT/"missing_artifacts.csv",missing)
    write_csv(OUT/"run_registry.csv",sources)
    write_csv(OUT/"historical_registry.csv",history_registry)
    write_csv(OUT/"compatibility.csv",[
        dict(pair="Six matched arms, three seeds",status="matched comparison",reason="Same 44 recordings and global mapping protocol; inputs/objectives differ"),
        dict(pair="Stage C context only versus contextual ASOT",status="matched comparison",reason="Same 439 pooled recordings within Stage C and saved global mapping policy"),
        dict(pair="Released SMQ / Stage C / short-mask / Phase 9 versus 44-recording study",status="descriptive side-by-side only",reason="Pooled 439-recording fitting and mapping versus 395/44 held-out decoder split"),
        dict(pair="Unit grouping versus matched B0",status="unavailable",reason="Evaluation predictions and costs not saved in grouping run")])
    report=["# LARa visual inspection workspace","",f"[Open the audit](index.html) · {len(names)} recordings, 18 cells.","",
            "Saved metrics reproduce the original evaluation scores in the source audit. Source positions, codes, window lengths and final mapping are checked on every displayed recording.","",
            "The viewer shows hip-relative joint positions as three 2-D joint clouds. No unverified skeleton edges are drawn. Motion display samples every second source frame (25 fps), while labels, token boundaries and metrics retain the original 50 fps grid.","",
            "The local row is the argmin of each cell's saved window-cost matrix, mapped with that cell's saved final global Hungarian mapping. It is a diagnostic for that fixed cost/prototype set; no new mapping is fitted. This is not a model trained without OT.","",
            "The grouping run is listed in missing_artifacts.csv: evaluation predictions were not saved, so no grouping timeline is fabricated.","",
            "Historical pooled LARa views cover released SMQ, Stage C context-only/contextual ASOT, short-mask T2 and Phase 9 continuous I1. Their saved maps or reproduced full-population Hungarian maps apply to all 439 recordings. These views are protocol-separated from the 44-recording study.","",
            "The selection list and observation ledger are generated. Entries remain pending until a person reviews the motion; no causal observations are asserted here.","",
            "Recorded source data were hip centered during dataset preparation. Original absolute motion is unavailable in these arrays; the viewer cannot assess global translation.",""]
    (OUT/"REPORT.md").write_text("\n".join(report),encoding="utf-8")
    from script.visual_audit import report_lara
    report_lara.main()
    print(f"Built {OUT / 'index.html'}")


if __name__=="__main__":
    build()
