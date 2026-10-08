"""Summarize machine-verified LARa audit measurements without inventing human observations."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"results/visual_audit/lara_001"
BASE=ROOT/"results/visual_audit/lara_matched_001"


def read_csv(name):
    with (OUT/name).open(encoding="utf-8",newline="") as f:
        return list(csv.DictReader(f))


def main():
    d=json.loads((BASE/"audit.json").read_text(encoding="utf-8"))
    history=read_csv("historical_registry.csv")
    records=read_csv("recordings.csv")
    selected=read_csv("inspection_selection.csv")
    matches=defaultdict(list)
    for row in records:
        if row["tolerance_frames"]=="12":
            matches[row["arm"]].append(row)
    lines=["# LARa visual inspection: generated measurements","",
        "[Open the atlas](index.html). All 44 matched evaluation recordings have motion pages. Historical pooled recordings are displayed with their own global mapping and protocol labels.","",
        "## Matched held-out decoder cells", "",
        "Means over the three seeds. Boundary columns average over 44 recordings and three seeds at ±0.24 s. These are descriptive counts; a displaced transition can contribute to both columns.","",
        "| Arm | MoF | F1@50 | Pooled pred./GT runs | Extra / rec. | Missed / rec. |", "|---|---:|---:|---:|---:|---:|"]
    for arm in d["arms"]:
        c=[x for x in d["cells"] if x["arm"]==arm]
        r=matches[arm]
        lines.append(f"| {arm} | {np.mean([x['metrics']['MoF'] for x in c]):.1f} | "
            f"{np.mean([x['metrics']['F1@50'] for x in c]):.1f} | "
            f"{sum(float(x['pred_runs']) for x in r)/sum(float(x['gt_runs']) for x in r):.2f} | "
            f"{np.mean([float(x['extra']) for x in r]):.1f} | "
            f"{np.mean([float(x['missed']) for x in r]):.1f} |")
    lines += ["", "## Earlier pooled LARa runs: separate protocol", "",
              "These runs were fitted and mapped over 439 recordings. Their scores are not paired differences from the 44-recording decoder cells.", "",
              "| Run | Seeds | Mean MoF | Mean F1@50 |", "|---|---|---:|---:|"]
    for arm in dict.fromkeys(x["arm"] for x in history):
        rows=[x for x in history if x["arm"]==arm]
        lines.append(f"| {arm} | {', '.join(x['seed'] for x in rows)} | "
                     f"{np.mean([float(x['MoF']) for x in rows]):.1f} | "
                     f"{np.mean([float(x['F1_50']) for x in rows]):.1f} |")
    cell=next(x for x in d["cells"] if x["arm"]=="T2 B0" and x["seed"]==111)
    classes=read_csv("classes.csv")
    support={x["action"]:int(x["frames"]) for x in classes if x["arm"]=="T2 B0" and x["seed"]=="111"}
    lines += ["", "## T2 B0 action recall, seed 111", "",
              "The large Handling(centred) class makes aggregate MoF sensitive to how its states are mapped. Recall remains a separate per-class readout.","",
              "| Action | GT frames | Recall |", "|---|---:|---:|"]
    for action_id,action in d["labels"].items():
        lines.append(f"| {action} | {support[action]} | {cell['recall'][action_id]:.1f}% |")
    lines += ["", "## Inspection coverage", "",
       f"The frozen list has {sum(x['kind']=='recording' for x in selected)} whole recordings and "
       f"{sum(x['kind']=='segment' for x in selected)} action-stratified segments. "
       "All observations are marked pending until motion is reviewed by a person.","",
       "The grouping run reports a failed gate, but its evaluation predictions and costs were not saved. Its recording-level failures are therefore absent from this workspace.","",
       "Source features were hip centered during preparation, so absolute translation cannot be inspected. The viewer shows verified position channels 3–5 at 25 displayed frames/s; action/timing arrays remain at 50 frames/s. No unverified joint connections are drawn.","",
       "The local decision is argmin of each saved window-cost row, mapped with the same final global Hungarian mapping. It does not represent a separate trained model without OT.","",
       "Original matched-cell metrics and window-code alignment were checked for every cell. Browser screenshots verified the main layout; the automatic control checks cover seed, method, display, frame navigation and atlas ordering. Human motion interpretation remains pending."]
    (OUT/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    claims=["# LARa claim audit · before human motion review","",
       "| Claim | Current measurement | Status | Next visual check |", "|---|---|---|---|",
       "| Excess predicted boundaries | All matched arms have excess boundaries; counts and tolerances are in recordings.csv. | Observed count pattern | Check whether switches recur inside actions away from GT edges. |",
       "| Missing boundaries | The same runs also miss GT boundaries under one-to-one matching. Displaced transitions can appear in both counts. | Observed count pattern | Inspect GT transitions with local cost and final OT rows. |",
       "| Multiple states within one action | Saved confusion matrices and oracle relabeling expose mapping granularity. | Observed association; semantic meaning unresolved | Inspect repeatability of movement phases across recordings and seeds. |",
       "| Subject/recording cues drive errors | Current viewer can sort by subject and recording. Association does not identify a physical cause. | Unresolved | Compare action mix, motion and acquisition across affected recordings. |",
       "| Context corrects unit evidence | Historical contextual runs and the matched GRU run have different protocols/objectives; model-specific before/after costs exist only for matched arms. | Unresolved | Inspect paired local/final rows and compare matched contextual settings separately. |",
       "| Grouping repairs action identity | Grouping gate failed; evaluation predictions were not saved. | Contradicted for tested spectral grouping at aggregate level; recording mechanism unavailable | Save evaluation states in any future grouping test before making a visual claim. |",
       "", "These statuses describe saved measurements. Pending observation rows are not evidence of visible movement motifs."]
    (OUT/"claims_audit.md").write_text("\n".join(claims)+"\n",encoding="utf-8")
    print(f"Wrote {OUT/'REPORT.md'} and claims_audit.md")


if __name__=="__main__":main()
