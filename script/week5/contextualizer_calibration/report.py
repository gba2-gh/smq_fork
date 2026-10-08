"""Markdown report for one calibration run, written only from the saved CSV /
JSON artifacts. It reports measurements and flags; it does not declare a
winner among checkpoint policies or tokenizers, and it never edits Task 2's
report."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

FLAG = 1.0  # practical F1@50 sensitivity flag in points; not a significance or convergence threshold
METRICS = ("F1@50", "MoF", "Edit", "F1@10", "F1@25")


def read(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def fmt(x, digits: int = 2) -> str:
    try:
        return f"{float(x):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def table(headers: list[str], rows: list[list]) -> str:
    if not rows:
        return "_no rows_\n"
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out) + "\n"


class Data:
    def __init__(self, run_dir: Path):
        self.dir = run_dir
        self.seg = read(run_dir / "segmentation.csv")
        self.train = {r["job_id"]: r for r in read(run_dir / "training.csv")}
        self.policy = read(run_dir / "checkpoint_policies.csv")
        self.not_run = read(run_dir / "not_run.csv")
        self.manifest = json.loads((run_dir / "manifest.json").read_text()) if (run_dir / "manifest.json").exists() else {}
        repro = run_dir / "reproduction.json"
        self.repro = json.loads(repro.read_text()) if repro.exists() else {}
        self.jobs = read(run_dir / "planned_training.csv")
        scale = self.manifest.get("scale", {})
        self.cap, self.cap_sens = scale.get("outer_cap", 50), scale.get("outer_cap_sensitivity", 150)

    def metric(self, job_id: str, role: str, cap: int, rung: int | str, name: str = "F1@50"):
        rows = [r for r in self.seg if r["job_id"] == job_id and r["ckpt_role"] == role and int(r["outer_cap"]) == cap]
        if not rows:
            return None
        target = max(int(r["rung_steps"]) for r in rows) if rung == "last" else int(rung)
        for r in rows:
            if int(r["rung_steps"]) == target:
                return float(r[name])
        return None

    def at_cap(self, job_id: str, cap: int, first: bool = False):
        """Final-5k-checkpoint F1@50 at one outer cap (decoded as ckpt_5k at the base cap, final at the sensitivity cap)."""
        for role in ("ckpt_5k", "final"):
            rungs = self.rungs(job_id, role, cap)
            if rungs:
                return self.metric(job_id, role, cap, rungs[0] if first else "last")
        return None

    def final_metric(self, job_id: str):
        """Fixed-final checkpoint (5k schedules decode it as ckpt_5k, the others as final)."""
        for role in ("final", "ckpt_5k"):
            value = self.metric(job_id, role, self.cap, "last")
            if value is not None:
                return value
        return None

    def rungs(self, job_id: str, role: str, cap: int) -> list[int]:
        return sorted({int(r["rung_steps"]) for r in self.seg if r["job_id"] == job_id and r["ckpt_role"] == role
                       and int(r["outer_cap"]) == cap})

    def job(self, group: str, pipeline: str, config: str | None = None, seed: int = 111):
        for j in self.jobs:
            if (j["group"] == group and j["pipeline"] == pipeline and int(j["seed"]) == seed and
                    (config is None or j["config_name"] == config)):
                return j
        return None

    def exposure(self, job_id: str) -> dict:
        path = self.dir / "jobs" / job_id / "exposure_steps.npz"
        return dict(np.load(path)) if path.exists() else {}


def endpoints(d: Data, job_id: str, role: str, cap: int | None = None) -> tuple:
    cap = d.cap if cap is None else cap
    rungs = d.rungs(job_id, role, cap)
    if not rungs:
        return (None, None)
    return d.metric(job_id, role, cap, rungs[0]), d.metric(job_id, role, cap, rungs[-1])


def write_report(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    d = Data(run_dir)
    m = d.manifest
    smoke = bool(m.get("scale", {}).get("smoke"))
    lines = [f"# Contextualizer search calibration — {m.get('run_id', run_dir.name)}", ""]
    if smoke:
        lines += ["> **SMOKE RUN — shrunken budgets on a recording subset. Not a reproduction, not scientific "
                  "evidence.**", ""]
    planned = m.get("planned", {})
    done_jobs, done_evals = len(d.train), len({r["eval_id"] for r in d.seg})
    lines += [f"Manifest: `manifest.json` (status `{m.get('status')}`, implementation `{m.get('implementation_hash')}`). "
              f"Sources: `{m.get('qic_run')}`, `{m.get('task2_run')}`.", "",
              f"Completed {done_jobs}/{planned.get('training_jobs')} training jobs and {done_evals}/"
              f"{planned.get('planned_ot_fits')} planned OT fits (before alias de-duplication). "
              f"Unfinished units: {len(d.not_run)} (see `not_run.csv`).", "",
              f"Reproduction gate against Task 2: **{d.repro.get('status', 'pending')}** (`reproduction.json`).", ""]
    if d.repro.get("status") == "failed":
        lines += ["**The reproduction gate failed.** Cells that depend on the reproduced path are not interpreted "
                  "here; see `reproduction.json` and `reproduction_mismatch_environment.json`.", ""]
    lines += ["Pooling: tokenizer/OT fitting and scoring pool all HuGaDB recordings, including the masked-validation "
              "recordings, so decoding is transductive and development-only. Development F1@50 selects procedure "
              "only; nothing here is a test result or a tokenizer comparison.", ""]

    lines += ["## 1. Main configurations", "",
              f"Validation-selected checkpoint, outer cap {d.cap}; endpoints are the first and last final-solve rungs "
              "(500 / 4000 in a full run). Raw is the primary guard-radius-0 pipeline; native VQ uses guard radius 14.", ""]
    for pipeline in ("native_vq", "raw"):
        rows = []
        for name in ("reference", "short_context", "long_mask", "combined"):
            j = d.job("main", pipeline, name)
            if not j or j["job_id"] not in d.train:
                rows.append([name, "not run"] + [""] * 8)
                continue
            t = d.train[j["job_id"]]
            lo, hi = endpoints(d, j["job_id"], "selected")
            rows.append([name, f"{j['context']}/{j['mask_span']}/{j['stride']}/{j['batch_chunks']}", t["selected_step"],
                         fmt(t["best_val_loss"], 3), fmt(lo), fmt(hi)] +
                        [fmt(d.metric(j["job_id"], "selected", d.cap, "last", k)) for k in ("MoF", "Edit", "F1@10", "F1@25")])
        lines += [f"**{pipeline}**", "", table(["config", "ctx/span/stride/batch", "selected step", "val loss",
                  "F1@50 first rung", "F1@50 last rung", "MoF", "Edit", "F1@10", "F1@25"], rows)]
    lines += ["Validation loss is not comparable across configurations that change the supervised targets "
              "(context, span, guard).", ""]

    lines += ["## 2. Target-count sensitivity (native VQ, guard 14)", ""]
    ref = d.job("main", "native_vq", "reference")
    long_ = d.job("main", "native_vq", "long_mask")
    matched = d.job("target_match", "native_vq", "long_mask")
    rows = []
    for label, j in (("span 8 reference", ref), ("span 16 long mask", long_), ("span 16, targets matched to span 8", matched)):
        if not j or j["job_id"] not in d.train:
            rows.append([label, "not run", "", "", "", ""])
            continue
        x = d.exposure(j["job_id"])
        lo, hi = endpoints(d, j["job_id"], "selected")
        rows.append([label, int(x["eligible"].sum()) if x else "n/a", fmt(x["weight"].sum(), 0) if x else "n/a",
                     fmt(lo), fmt(hi), fmt(d.final_metric(j["job_id"]))])
    lines += [table(["job", "eligible targets (total)", "real-frame weight (total)", "F1@50 first rung (selected)",
                     "F1@50 last rung (selected)", "F1@50 last rung (final)"], rows)]
    if matched and matched["job_id"] in d.train and ref and ref["job_id"] in d.train:
        xm, xr = d.exposure(matched["job_id"]), d.exposure(ref["job_id"])
        paired = bool(xm and xr and len(xm["n_ref"]) == len(xr["eligible"]) and (xm["n_ref"] == xr["eligible"]).all())
        lines += [f"Pairing check (matched job's reference counts equal the span-8 job's eligible counts at every "
                  f"update): **{'verified' if paired else 'NOT verified'}**. Matched job: "
                  f"{int(xm['retries'].sum())} mask retries; achieved targets equal reference at "
                  f"{int((xm['eligible'] == xm['n_ref']).sum())}/{len(xm['eligible'])} updates; real-frame weight "
                  f"{fmt(xm['weight'].sum(), 0)} versus the span-8 job's {fmt(xr['weight'].sum(), 0)}. Retry-based "
                  f"resampling conditions the mask distribution on feasibility. Count equality does not equalize "
                  f"terminal-window frame weights.", ""]
    lines += ["This is a supervision-count sensitivity (long gaps versus more supervised targets); it does not "
              "isolate the guard and does not change the tokenizer.", ""]

    lines += ["## 3. Genuine 3,000-update schedule versus 5,000 (reference configuration)", ""]
    rows = []
    for pipeline in ("native_vq", "raw"):
        j3, j5 = d.job("horizon3k", pipeline), d.job("main", pipeline, "reference")
        for label, j in (("3k horizon", j3), ("5k horizon", j5)):
            if not j or j["job_id"] not in d.train:
                rows.append([pipeline, label, "not run", "", "", ""])
                continue
            rows.append([pipeline, label, d.train[j["job_id"]]["selected_step"],
                         fmt(endpoints(d, j["job_id"], "selected")[1]), fmt(d.final_metric(j["job_id"])),
                         fmt(d.train[j["job_id"]]["best_val_loss"], 3)])
    lines += [table(["pipeline", "schedule", "selected step", "F1@50 last rung (selected)", "F1@50 last rung (final)",
                     "val loss"], rows),
              "Intermediate 1k/3k checkpoints of a 5k schedule measure progress under that schedule; only the 3k "
              "jobs have a cosine horizon of 3,000.", ""]

    lines += ["## 4. Checkpoint-policy replay", "",
              "Fixed final, full-history validation-selected and patience-1000 replay (strict improvement, earliest "
              f"ties). F1@50 at the last rung (4000 in a full run), outer cap {d.cap}.", ""]
    rows = []
    for j in d.jobs:
        jid = j["job_id"]
        if jid not in d.train:
            continue
        t = d.train[jid]
        rows.append([jid, t["selected_step"], t["replay_stop_step"] or "no stop", t["replay_selected_step"],
                     fmt(d.final_metric(jid)),
                     fmt(d.metric(jid, "selected", d.cap, "last")), fmt(d.metric(jid, "replay", d.cap, "last")),
                     fmt(d.metric(jid, "ckpt_1k", d.cap, "last")), fmt(d.metric(jid, "ckpt_3k", d.cap, "last"))])
    lines += [table(["job", "selected step", "replay stop", "replay step", "final", "selected", "replay", "1k", "3k"], rows),
              "Policies are compared on development segmentation only; an arbitrary checkpoint is never chosen by "
              "its highest evaluation score. Small or inconsistent differences are unresolved, not a winner.", ""]

    lines += ["## 5. Reference-seed replication (seed 111 versus 222)", ""]
    rows = []
    for pipeline in ("native_vq", "raw"):
        a, b = d.job("main", pipeline, "reference", 111), d.job("seed222", pipeline, "reference", 222)
        for role in ("selected", "final", "replay"):
            fa = d.metric(a["job_id"], role if role != "final" else "ckpt_5k", d.cap, "last") if a else None
            fb = d.metric(b["job_id"], role, d.cap, "last") if b else None
            rows.append([pipeline, role, fmt(fa), fmt(fb), fmt(None if fa is None or fb is None else fb - fa)])
    lines += [table(["pipeline", "policy", "seed 111", "seed 222", "difference"], rows),
              "Unequal seed coverage: one extra seed per primary pipeline, reference configuration only.", ""]

    lines += ["## 6. Solver-budget effects", ""]
    rows, flags = [], 0
    by_eval: dict[str, list[dict]] = {}
    for r in d.seg:
        by_eval.setdefault(r["eval_id"], []).append(r)
    for eid, rs in sorted(by_eval.items()):
        rs = sorted(rs, key=lambda r: int(r["rung_steps"]))
        delta = float(rs[-1]["F1@50"]) - float(rs[0]["F1@50"])
        flag = abs(delta) >= FLAG
        flags += flag
        rows.append([eid, rs[0]["rung_steps"], rs[-1]["rung_steps"], fmt(delta), "FLAG" if flag else "",
                     rs[-1]["status_counts"], rs[-1]["ot_status"], rs[-1]["outer_iterations"]])
    lines += [f"Final-rung continuation: F1@50 change between the first and last rung of each evaluation "
              f"(flag at {FLAG:.0f} point, a practical flag only). {flags} of {len(by_eval)} evaluations flagged.", "",
              table(["evaluation", "first rung", "last rung", "dF1@50", "flag", "recording statuses (last rung)",
                     "outer status", "outer iterations"], rows)]
    rows = []
    for j in d.jobs:
        if j["group"] != "main" or j["config_name"] not in ("reference", "combined"):
            continue
        a, b = d.at_cap(j["job_id"], d.cap), d.at_cap(j["job_id"], d.cap_sens)
        lo_a, lo_b = d.at_cap(j["job_id"], d.cap, first=True), d.at_cap(j["job_id"], d.cap_sens, first=True)
        rows.append([j["job_id"], fmt(a), fmt(b), fmt(None if a is None or b is None else b - a),
                     fmt(lo_a), fmt(lo_b)])
    lines += [f"Outer-iteration cap {d.cap} versus {d.cap_sens} (final 5k checkpoint, same initialization, separate fits):", "",
              table(["job", "F1@50 last rung, base cap", "sensitivity cap", "difference", "first rung, base cap", "first rung, sensitivity cap"], rows)]
    for pipeline in ("native_vq", "raw"):
        ref_j, comb_j = d.job("main", pipeline, "reference"), d.job("main", pipeline, "combined")
        if not ref_j or not comb_j:
            continue
        values = {cap: (d.at_cap(ref_j["job_id"], cap), d.at_cap(comb_j["job_id"], cap))
                  for cap in (d.cap, d.cap_sens)}
        if all(v is not None for pair in values.values() for v in pair):
            order = {cap: ("reference > combined" if p[0] > p[1] else "combined > reference") for cap, p in values.items()}
            lines += [f"Ordering ({pipeline}): cap {d.cap} {order[d.cap]}; cap {d.cap_sens} {order[d.cap_sens]}"
                      f"{' — **reversal**' if order[d.cap] != order[d.cap_sens] else ''}.", ""]
    for pipeline in ("native_vq", "raw"):
        order = {}
        for edge, first in (("first rung", True), ("last rung", False)):
            vals = {}
            for name in ("reference", "short_context", "long_mask", "combined"):
                j = d.job("main", pipeline, name)
                rungs = d.rungs(j["job_id"], "selected", d.cap) if j else []
                if rungs:
                    vals[name] = d.metric(j["job_id"], "selected", d.cap, rungs[0] if first else "last")
            order[edge] = sorted(vals, key=lambda k: -vals[k])
        if len(order["first rung"]) == 4 and len(order["last rung"]) == 4:
            same = order["first rung"] == order["last rung"]
            lines += [f"Configuration ordering by F1@50 ({pipeline}, selected checkpoint): first rung "
                      f"{' > '.join(order['first rung'])}; last rung {' > '.join(order['last rung'])}"
                      f"{'' if same else ' — **ordering reversal**'}.", ""]
    capped = sum("capped" in rs[-1]["status_counts"] for rs in
                 (sorted(v, key=lambda r: int(r["rung_steps"])) for v in by_eval.values()))
    lines += [f"Outer-fit status: see `ot_status` in `segmentation.csv`; outer-capped evaluations are budgeted "
              f"outcomes, not converged fits. Evaluations with any capped recording at the last rung: {capped}.", ""]

    lines += ["## 7. Missing cells and limits", ""]
    lines += [table(["unit", "status", "reason"], [[r["unit"], r["status"], r["reason"]] for r in d.not_run]) if d.not_run
              else "All planned units completed in this invocation.\n"]
    lines += ["Limits: development data only; one tokenizer checkpoint family; seed 111 (seed 222 only for the "
              "reference configuration); GPU training is not guaranteed bit-reproducible across devices; 1 point is a "
              "flag, not a significance level. This calibration does not rank tokenizers.", ""]
    path = Path(run_dir) / "REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
