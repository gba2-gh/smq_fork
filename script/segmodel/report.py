"""Turn incremental M1 JSONL output into the review files for this milestone."""
from __future__ import annotations

import csv
import json
from collections import defaultdict

import numpy as np

from script.segmodel.core import OUT


def main():
    source = OUT / "cells.jsonl"
    rows = [json.loads(line) for line in source.read_text(encoding="utf8").splitlines() if line]
    fields = sorted({key for row in rows for key in row})
    with (OUT / "m1_cells.csv").open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)

    completed = [r for r in rows if r.get("status") == "complete" and r.get("method") != "d7"]
    grouped = defaultdict(list)
    for row in completed:
        key = (row["dataset"], row["standardize_per_recording"], row["num_units"], row["temperature_multiplier"])
        grouped[key].append(row)
    lines = ["# Milestone 1: segmental clustering", "", "## Status", "", "This report contains only completed transductive cells. Ground-truth labels are used only for the D7 check and evaluation mapping/scoring.", "", "## D7 provenance gate", ""]
    for row in rows:
        if row.get("method") == "d7":
            lines.append(f"- {row['dataset']}: MoF {row['MoF']:.2f}; status {row['status']}.")
    lines += ["", "## Completed primary cells", "", "| Dataset | recording z-score | K | temperature | seeds | MoF mean (population SD) | F1@50 mean (population SD) |", "|---|---:|---:|---:|---:|---:|---:|"]
    for key, values in sorted(grouped.items()):
        mof=np.asarray([x["MoF"] for x in values]); f50=np.asarray([x["F1@50"] for x in values])
        ds, std, k, temp = key
        lines.append(f"| {ds} | {std} | {k} | {temp:g} | {len(values)} | {mof.mean():.2f} ({mof.std():.2f}) | {f50.mean():.2f} ({f50.std():.2f}) |")
    not_done=[r for r in rows if r.get("status") not in {"complete", "degenerate"}]
    lines += ["", "## Not run or invalid", ""]
    lines += [f"- {r.get('dataset')}, K={r.get('num_units')}, seed={r.get('seed')}: {r.get('status')} — {r.get('reason', '')}." for r in not_done] or ["- None."]
    lines += ["", "## Open questions", "", "- Does the primary gate hold after the planned hard-assignment and subject-disjoint evaluations?", "- How sensitive are the findings to vocabulary size and temperature?", "", "## What these numbers cannot establish", "", "- These transductive, oracle-mapped measurements do not establish deployment performance or a causal source of any dataset difference."]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf8")


if __name__ == "__main__":
    main()
