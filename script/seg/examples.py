"""
Retrieval examples (continuous 8-bin representation, common anchors), Fixed vs Oracle.

Query anchors are NOT selected by outcome: for each class, the first two anchors of fold 0's
evaluation set in a deterministic SHA-256 ordering of (recording, timestamp) are shown, so
successes and failures are both included. Gallery = fold-0 evaluation anchors of other groups.

  python script/seg/examples.py --dataset lara
"""
import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from script.seg import partition as P  # noqa: E402
from script.seg import pipeline as PL  # noqa: E402
from script.seg import represent as R  # noqa: E402

OUT = ROOT / "results" / "seg"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--per_class", type=int, default=2)
    a = ap.parse_args()
    data = PL.Data(a.dataset)
    fold_of_rec, _ = data.make_folds()
    anchors = data.make_anchors()
    ev = np.flatnonzero(fold_of_rec == 0)
    fit = np.flatnonzero(fold_of_rec != 0)
    mu, sd = R.fit_scaler([data.z[r] for r in fit])
    parts = {"fixed": [P.fixed_edges(int(t), data.W) for t in data.T],
             "oracle": [P.oracle_edges(l) for l in data.labels]}
    lines = [f"# Retrieval examples — {a.dataset}, fold 0 evaluation set, continuous 8-bin representation\n",
             "Query anchors: first "
             f"{a.per_class} per class in SHA-256 order of (recording, timestamp) — not selected by outcome. "
             "Top-3 nearest gallery anchors' segments (Euclidean), gallery excludes the query's "
             f"{data.group_kind}. ✔ = same action label as the query anchor.\n"]
    picks = {}
    for c in range(data.C):
        cand = [(hashlib.sha256(f"{data.names[r]}:{t}".encode()).hexdigest(), r, int(t))
                for r in ev for t in anchors[r] if data.labels[r][t] == c]
        picks[c] = sorted(cand)[:a.per_class]
    for cond, edges in parts.items():
        V, meta = [], []
        for r in ev:
            v = R.segment_vectors(data.z[r], edges[r], mu, sd)
            for t in anchors[r]:
                s = int(np.searchsorted(edges[r], t, side="right") - 1)
                V.append(v[s]); meta.append((r, int(t), int(edges[r][s]), int(edges[r][s + 1]), int(data.labels[r][t])))
        V = torch.as_tensor(np.stack(V), device=PL.DEV)
        grp = np.array([data.group[m[0]] for m in meta])
        lines.append(f"\n## {cond} segmentation\n")
        lines.append("| query (recording, t) | label | query segment (s) | top-1 | top-2 | top-3 | hits |")
        lines.append("|---|---|---|---|---|---|---|")
        for c in range(data.C):
            for (_, r, t) in picks[c]:
                qi = next(i for i, m in enumerate(meta) if m[0] == r and m[1] == t)
                d = torch.cdist(V[qi:qi + 1], V)[0].cpu().numpy()
                d[grp == grp[qi]] = np.inf
                order = np.argsort(d)[:3]
                cells, hits = [], 0
                for j in order:
                    rr, tt, s0, s1, ll = meta[j]
                    ok = ll == c
                    hits += ok
                    cells.append(f"{data.names[rr]} [{s0 / data.fps:.1f}–{s1 / data.fps:.1f}s] {data.class_names[ll]} {'✔' if ok else '✘'}")
                q = meta[qi]
                lines.append(f"| {data.names[r]}, {t / data.fps:.1f}s | {data.class_names[c]} | "
                             f"{q[2] / data.fps:.1f}–{q[3] / data.fps:.1f} | " + " | ".join(cells) + f" | {hits}/3 |")
    (OUT / f"retrieval_examples_{a.dataset}.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote", OUT / f"retrieval_examples_{a.dataset}.md")


if __name__ == "__main__":
    main()
