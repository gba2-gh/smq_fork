"""Historical compatibility jobs (spec: 'Add two separate seed-111
compatibility jobs: native-VQ IDs and raw-K-means IDs through historical
helpers, using original masking, initialization and patience_steps=1000').

Deliberately NOT built on Task 1's qic_v3 artifacts or Task 2's
`contextual.py`/`ot_arms.py` -- these two jobs replay
`script/fsq_reconstruction/encoder_sweep.py`'s OWN extraction
(`extract_representation`) and downstream pipeline
(`script.fsq_reconstruction.experiment.run_stream_downstream`, which is what
"original masking, initialization and patience_steps=1000" refers to: the
historical `stageb_data.build_mask`, `CodeContextModel`'s pre-Task-2 id-only
init, and `PilotConfig.contextualizer_patience_steps=1000` default)
directly, at K=512, seed 111.

Two DIFFERENT streams, matching the spec's "native-VQ IDs and raw-K-means
IDs" literally (post-review correction: the original implementation ran
BOTH jobs through PCA+K-means reclustering of "vq_e30"'s pre-quantization
encoder features, which reproduces VQ-encoder K-means, not native VQ code
IDs -- even though `extract_representation`'s underlying `extract_recording`
(script/fsq_reconstruction/experiment.py) already computes the checkpoint's
own native VQ ids via `model.indices`, the exact ids Task 1's `qic_v3` uses
for its `vq_native` family and `fsq_pilot_002` uses for its own historical
`vq_native` stream. That native id array was being computed and then
silently discarded in favor of a K-means re-derivation):

    vq_native   native VQ code ids read directly off the epoch-30 VQ
                checkpoint's quantizer output (`extracted[name]["ids"]`,
                already computed by `extract_representation`/
                `extract_recording`) -- no K-means anywhere in this path.
                Its `pca` field is still needed downstream (initial vector
                prototypes for the contextual-ASOT arm), so PCA (fit on the
                SAME sample as before, `fit_pca_kmeans`) is still fit and
                used to project native features into PCA space; only the
                K-means CLUSTER ASSIGNMENT step is skipped for ids. Diffed
                against `results/fsq_reconstruction/fsq_pilot_002/
                segmentation.csv`'s own `vq_native` rows at seed 111 (the
                pilot run that produced these ids' historical reference;
                confirmed present: 96 rows, streams
                {fsq_kmeans, fsq_native, vq_kmeans, vq_native}, seeds
                {111, 222, 1538574472}).
    raw         raw K-means (PCA+K-means-512 on raw un-encoded patches,
                unchanged from before) -- kept as-is per the review, since
                raw patches have no native quantizer ids to begin with.
                Diffed against the historical
                `results/fsq_reconstruction/encoder_sweep/encoder_sweep_001/
                segmentation.csv` (its own "raw" rows), also unchanged.
"""

from __future__ import annotations

import csv
import time
from dataclasses import replace
from pathlib import Path

import torch

from script.fsq_reconstruction import encoder_sweep as sweep
from script.fsq_reconstruction.experiment import (
    OUT_ROOT, check_identity_against_released_meta, fit_pca_kmeans, project_to_pca, run_stream_downstream,
    transform_and_assign,
)

RAW_HISTORICAL_CSV = OUT_ROOT / "encoder_sweep" / "encoder_sweep_001" / "segmentation.csv"
VQ_NATIVE_HISTORICAL_CSV = OUT_ROOT / "fsq_pilot_002" / "segmentation.csv"
STREAMS = ("vq_native", "raw")
SEED = 111


def historical_csv_for(stream_name: str) -> Path:
    return VQ_NATIVE_HISTORICAL_CSV if stream_name == "vq_native" else RAW_HISTORICAL_CSV


def read_historical_rows(stream_name: str | None = None) -> list[dict]:
    """With `stream_name`, reads only that stream's own historical CSV (the
    two jobs use different historical references, see module docstring).
    Without it, reads and concatenates both."""
    paths = [historical_csv_for(stream_name)] if stream_name is not None else [RAW_HISTORICAL_CSV,
                                                                               VQ_NATIVE_HISTORICAL_CSV]
    rows: list[dict] = []
    for path in paths:
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as fh:
            rows.extend(csv.DictReader(fh))
    return rows


def run_one_stream(stream_name: str, source_run: str, device: torch.device,
                   artifacts_dir: Path, deadline: float | None = None,
                   cfg_overrides: dict | None = None) -> dict:
    if stream_name not in STREAMS:
        raise ValueError(f"unknown historical stream {stream_name!r}; expected one of {STREAMS}")
    rep = "vq_e30" if stream_name == "vq_native" else "raw"
    source_dir = OUT_ROOT / source_run
    source_cfg = sweep.load_source_config(source_dir)
    cfg = replace(source_cfg, downstream_seeds=(SEED,))
    if cfg_overrides:
        cfg = replace(cfg, **cfg_overrides)
    extracted, ckpt_hash = sweep.extract_representation(rep, source_cfg, source_dir, device)
    identity = check_identity_against_released_meta(cfg, extracted)
    lengths = {name: rec["lengths"] for name, rec in extracted.items()}
    vocab = fit_pca_kmeans(extracted, replace(cfg, preprocessing_seed=SEED))
    pca = {}
    if stream_name == "vq_native":
        # Native VQ code ids straight off the checkpoint's own quantizer
        # output (already computed by `extract_representation` via
        # `extract_recording`'s `model.indices` read) -- no K-means
        # assignment. PCA projection is still needed for the contextual arm's
        # initial vector prototypes, so project (but never re-cluster) into
        # the fitted PCA space.
        ids = {name: rec["ids"] for name, rec in extracted.items()}
        for name, rec in extracted.items():
            pca[name] = project_to_pca((vocab.scaler, vocab.pca), rec["flat"])
        kind = "native"
    else:
        ids = {}
        for name, rec in extracted.items():
            pca[name], ids[name] = transform_and_assign(vocab, rec["flat"])
        kind = "kmeans"
    stream = dict(arm=stream_name, kind=kind, ids=ids, pca=pca, lengths=lengths)
    started = time.perf_counter()
    result = run_stream_downstream(stream_name, stream, cfg, SEED, identity, deadline, artifacts_dir,
                                   contextualizer_device=device)
    result["checkpoint_sha256"] = ckpt_hash
    result["seconds"] = time.perf_counter() - started
    return result


def diff_against_historical(rows: list[dict], historical_rows: list[dict]) -> list[dict]:
    """For every (stream, method, rung_steps) pair present in both the fresh
    rows and the historical CSV at seed 111, records the MoF/Edit/F1@10/25/50
    difference. Never invents a tolerance -- this only reports the measured
    diffs; pass/fail judgment is left to the reader (spec: 'do not invent
    reproduction tolerances'). `historical_rows` should already be the
    RIGHT historical CSV's rows for the stream(s) present in `rows` (see
    `read_historical_rows(stream_name=...)` / `historical_csv_for`)."""
    hist_by_key = {(r["stream"], r["method"], r["rung_steps"]): r for r in historical_rows
                  if r.get("downstream_seed") == str(SEED)}
    out = []
    for row in rows:
        key = (row.get("stream"), row.get("method"), str(row.get("rung_steps")))
        hist = hist_by_key.get(key)
        if hist is None:
            out.append(dict(**key_dict(key), status="no_historical_row"))
            continue
        entry = key_dict(key)
        for metric in ("MoF", "Edit", "F1@10", "F1@25", "F1@50"):
            try:
                entry[f"{metric}_diff"] = float(row[metric]) - float(hist[metric])
            except (KeyError, TypeError, ValueError):
                entry[f"{metric}_diff"] = None
        out.append(entry)
    return out


def key_dict(key) -> dict:
    return dict(stream=key[0], method=key[1], rung_steps=key[2])
