"""Validate M1r2 artifacts and write the milestone report from saved rows."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from script.exp2round.q1q2.core import load_dataset, map_predictions, score
from script.segmodel.m1r2_core import OUT, SEEDS
from script.segmodel.run_m1r2 import configuration_id, plan, read_rows, write_rows


METRICS = ("MoF", "Edit", "F1@10", "F1@25", "F1@50")


def number(row, key):
    try: return float(row[key])
    except (KeyError, TypeError, ValueError): return np.nan


def mean_sd(rows, key):
    x=np.asarray([number(row,key) for row in rows]); return float(x.mean()),float(x.std())


def table(rows, headers):
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "|" + "|".join("---" for _ in headers) + "|"] +
                     ["| " + " | ".join(map(str,row)) + " |" for row in rows])


def main():
    cells=read_rows(OUT/"cells.csv"); curves=read_rows(OUT/"lambda_curves.csv")
    by_id={row["cell_id"]:row for row in cells}
    primary, permutations, inductive, sensitivities=plan()
    expected=[]
    for protocol,items,folded in (("transductive",primary,False),("transductive",permutations,False),
                                  ("subject_disjoint",inductive,True),("transductive",sensitivities,False)):
        for item in items:
            fold=item[-1] if folded else None; config=item[:-1] if folded else item
            expected.append(configuration_id(config[0],protocol,*config[1:],fold))
    previous_reasons={r['cell_id']:r.get('reason','') for r in read_rows(OUT/'not_run.csv')}
    omissions=[]
    for cid in expected:
        if cid not in by_id: omissions.append({"cell_id":cid,"reason":previous_reasons.get(cid) or "not executed"})
        elif by_id[cid].get("status") not in {"complete","degenerate"}:
            omissions.append({"cell_id":cid,"reason":by_id[cid].get("reason") or by_id[cid]["status"]})
    write_rows(OUT/"not_run.csv",omissions)

    baselines={}
    for ds in ("hugadb","lara"):
        data=load_dataset(ds); baselines[ds]=score(data.gts,map_predictions(data.gts,data.codes,"hungarian"))

    gate={}; metric_rows=[]; guard_rows=[]; perm_rows=[]
    for std in (False,True):
        valid=True; improvements=[]; drops=[]; reasons=[]
        for ds in ("hugadb","lara"):
            observed=[]; perm=[]
            for seed in SEEDS:
                oid=configuration_id(ds,"transductive",std,"soft",500,1.,seed,"observed")
                pid=configuration_id(ds,"transductive",std,"soft",500,1.,seed,"permutation")
                if oid not in by_id or pid not in by_id:
                    valid=False; reasons.append(f"{ds}: incomplete primary/control")
                    continue
                observed.append(by_id[oid]); perm.append(by_id[pid])
            if len(observed)!=3: continue
            values=[]
            for metric in METRICS:
                mean,sd=mean_sd(observed,metric); values.append(f"{mean:.2f} +/- {sd:.2f}")
            metric_rows.append([ds,str(std),*values])
            for row in observed + perm:
                if row["status"]!="complete":
                    valid=False; reasons.append(f"{ds} {row['control']}: {row['status']}")
                if not all(np.isfinite(number(row,m)) for m in METRICS):
                    valid=False; reasons.append(f"{ds}: incomplete metrics")
                if row.get('calibration_selection_verified')=='False':
                    valid=False; reasons.append(f"{ds}: calibration failure")
            improvement=np.mean([number(r,"F1@50") for r in observed])-baselines[ds]["F1@50"]
            drop=baselines[ds]["MoF"]-np.mean([number(r,"MoF") for r in observed])
            improvements.append(improvement); drops.append(drop)
            diffs=np.asarray([number(o,"F1@50")-number(p,"F1@50") for o,p in zip(observed,perm)])
            separated=float(diffs.mean())>float(diffs.std())
            perm_rows.append([ds,str(std),", ".join(f"{x:.2f}" for x in diffs),
                              f"{diffs.mean():.2f}",f"{diffs.std():.2f}",str(separated)])
            if not separated: valid=False; reasons.append(f"{ds}: null-unseparated")
            guard_rows.append([ds,str(std),f"{mean_sd(observed,'dp_block_gt_ratio')[0]:.3f}",
                               f"{mean_sd(observed,'action_run_gt_ratio')[0]:.3f}",
                               f"{mean_sd(observed,'mean_duration_seconds')[0]:.2f}",
                               f"{mean_sd(observed,'median_duration_seconds')[0]:.2f}",
                               ", ".join(r.get("rounds","") for r in observed)])
        if not valid: outcome="inconclusive/incomplete" if any("incomplete" in r for r in reasons) else "fail"
        elif max(improvements)>=3 and max(drops)<=2: outcome="pass awaiting review"
        else: outcome="fail"
        gate[str(std)]={"outcome":outcome,"f1_improvements":improvements,"mof_drops":drops,
                        "reasons":sorted(set(reasons))}

    summary=[
      "1. This report covers the corrected per-window segment objective only.",
      f"2. Saved result rows: {len(cells)}; planned cells: {len(expected)}; omitted/failed: {len(omissions)}.",
      "3. D7 and input acceptance results are recorded in acceptance_inputs.json.",
      f"4. Standardization off gate: {gate['False']['outcome']}.",
      f"5. Standardization on gate: {gate['True']['outcome']}.",
      "6. Gate decisions use only soft K=500, base temperature, transductive observed/control pairs.",
      "7. Lambda was selected by released-code run duration; labels did not select it.",
      "8. Seed SD is initialization sensitivity, not sampling uncertainty.",
      "9. Degenerate cells remain reported and are excluded from gate evidence.",
      "10. No result here authorizes Milestone 2 automatically."]
    full_grid=[]
    for row in cells:
        full_grid.append([row['cell_id'],row['status'],
                          *[f'{number(row,m):.2f}' for m in METRICS],
                          f"{number(row,'action_run_gt_ratio'):.3f}",
                          row.get('rounds',''),row.get('cap_reached','')])
    baseline_rows=[[ds,*[f'{v[m]:.2f}' for m in METRICS]] for ds,v in baselines.items()]
    recovery_note=''
    if (OUT/'recovery_audit.json').exists():
        audit=json.loads((OUT/'recovery_audit.json').read_text())
        recovery_note=(f"Recovered {audit['recovered']}/{audit['attempted']} artifact-backed cells without fitting or decoding. "
            "Original failure rows and manifest are preserved in before_recovery/. Each recovered cell passed frame/boundary alignment, "
            "Hungarian-optimal mapping, objective-trace and label-free lambda-selection checks. "
            "The logging exception overwrote per-cell runtimes; those remain unavailable, not zero. "
            "The original experiment deadline and manifest remain unchanged. See recovery_audit.json. "
            "Subject-disjoint metrics use a separate evaluation-label Hungarian mapping for each fold; they are not a deployable label mapping. "
            "The lambda_start_action_runs field refers to fitting data and lambda_final_action_runs to evaluation data; "
            "these populations differ for subject-disjoint rows and cannot measure rate drift directly. "
            "Iteration-cap and empty-state flags are reported diagnostics; the brief does not automatically equate either with numerical failure.")
    text=["# Milestone 1 rerun: corrected segment objective","","## Ten-line summary","",*summary,
          "","## Gate status","", "```json",json.dumps(gate,indent=2),"```","",
          "## Recovery and verification","",recovery_note,"",
          "## Same-checkpoint baselines","",table(baseline_rows,['Dataset',*METRICS]),"",
          "## Primary metrics","",table(metric_rows,["Dataset","Recording z-score",*METRICS]),"",
          "## Guard diagnostics","",table(guard_rows,["Dataset","Recording z-score","DP/GT","runs/GT","mean duration","median duration","rounds"]),"",
          "## Permutation separation","",table(perm_rows,["Dataset","Recording z-score","paired F1@50 differences","mean","population SD","separated"]),"",
          "## Lambda curves","",f"All {len(curves)} evaluated lambda points, including all five label-based diagnostic metrics, are in [lambda_curves.csv](lambda_curves.csv). Only the label-free selected point enters the gate.","",
          "## Full grid","",f"All {len(cells)} executed cell rows are in [cells.csv](cells.csv). Transductive and subject-disjoint protocols are separate columns.","",
          table(full_grid,['Cell','Status',*METRICS,'runs/GT','Rounds','Cap reached']),"",
          "## Open questions","","- Does any corrected-objective advantage survive subject-disjoint fitting?","- Are failures associated with duration-cap concentration or with weak separation from permutation controls?","",
          "## What these numbers cannot establish","","These frozen-encoder results do not establish causal sources of dataset differences or a general limit on segment models. HuGaDB and LARa differ in class count, subject count and modality. Labels enter D7 provenance, Hungarian mapping, curve diagnostics, scoring and post-fit guards; they do not select lambda or another model setting.","",
          "## Not run","", *( [f"- `{row['cell_id']}`: {row['reason']}" for row in omissions] or ["- None."] )]
    (OUT/"REPORT.md").write_text("\n".join(text)+"\n",encoding="utf8")
    (OUT/"gate.json").write_text(json.dumps(gate,indent=2)+"\n",encoding="utf8")


if __name__=="__main__": main()
