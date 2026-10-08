"""Fold layout and the matched B0 decode on continuous per-window inputs (frozen PCA features for F1, learned embeddings for L_*).

The Milestone 1 machinery (script/compositional_dynamics: decode ladder, scoring, association) is reused with run-time overrides to
BABEL-2 (window 8, 30 fps, C = 5, no subject ids). Labels are read only by the scoring and diagnostic functions at the end."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from script.action_transport import continuous
from script.self_labelled import config as C

CPU = torch.device("cpu")


def apply_overrides() -> None:
    from script.compositional_dynamics import config as CD
    from script.compositional_dynamics import data as Dm
    CD.DATASET, CD.WINDOW, CD.FPS, CD.NUM_STATES = C.DATASET, C.WINDOW, C.FPS, C.NUM_STATES
    Dm.subject_of = lambda name: name              # BABEL has no subject ids


def all_names() -> list:
    from script.week5.raw_kmeans_cross_dataset import datasets as D
    return sorted(D.recording_names(D.SPECS[C.DATASET], None))


def fold_split(names: list, fold: str):
    from script.pretrained_encoder.run import make_folds
    held = set(make_folds(names)[int(fold[4:])])
    return [n for n in names if n not in held], [n for n in names if n in held]


def layout(names: list) -> dict:
    """name -> dict(lengths, starts, frame_count): the window grid of the whole pipeline (8-frame windows, short last window)."""
    from script.week5.raw_kmeans_cross_dataset import datasets as D
    spec = D.SPECS[C.DATASET]
    counts = D.frame_counts(spec, names)
    out = {}
    for n in names:
        T = int(counts[n])
        P = int(np.ceil(T / C.WINDOW))
        lens = np.full(P, C.WINDOW, dtype=np.int64)
        lens[-1] = T - (P - 1) * C.WINDOW
        out[n] = dict(lengths=lens, starts=np.arange(P, dtype=np.int64) * C.WINDOW, frame_count=T)
    return out


def population(names: list, lay: dict):
    from script.compositional_dynamics import data as Dm
    return Dm.make_population(names, [np.zeros(len(lay[n]["lengths"]), dtype=np.int64) for n in names],
                              [lay[n]["lengths"] for n in names], 1)


def kmeans_init(z_fit: dict, lay: dict, fit_names: list, seed: int) -> np.ndarray:
    """Label-free initial prototypes: length-weighted k-means++ (n_init 5) on the fitting windows' unit vectors."""
    from sklearn.cluster import KMeans
    Z = np.concatenate([z_fit[n] for n in fit_names])
    w = np.concatenate([lay[n]["lengths"] for n in fit_names]).astype(np.float64)
    km = KMeans(n_clusters=C.NUM_STATES, init="k-means++", n_init=5, random_state=seed).fit(Z, sample_weight=w)
    return continuous.l2_normalize(km.cluster_centers_)[0]


def decode(z: dict, fit_names: list, eval_names: list, lay: dict, seed: int, smoke: bool, mu_fixed=None, deadline=None) -> dict:
    """Matched B0 on per-window vectors z (name -> [P, D]). mu_fixed: decode with these prototypes and no refit (secondary readout).

    Returns the scores plus `artifacts` (fitted prototypes, eval costs, raw window states, mapped frame predictions, solver records)
    so the scored predictions can be inspected without re-running anything."""
    from script.compositional_dynamics import decode as Dc
    from script.compositional_dynamics import evaluate as Ev
    from script.compositional_dynamics import train as Tr
    apply_overrides()
    t0 = time.perf_counter()
    outer, rungs = (C.SMOKE_OUTER_CAP, C.SMOKE_RUNGS) if smoke else (C.OUTER_CAP, C.RUNGS)

    def recs(names):
        out = []
        for n in names:
            u, zero = continuous.l2_normalize(z[n].astype(np.float64))
            out.append(continuous.ContinuousRecordingInput(n, np.zeros(len(u), np.int32), lay[n]["lengths"].astype(np.int32),
                                                           C.FPS, C.WINDOW, u, zero))
        return out

    fit_recs, eval_recs = recs(fit_names), recs(eval_names)
    if mu_fixed is None:
        mu0 = kmeans_init({n: r.z for n, r in zip(fit_names, fit_recs)}, lay, fit_names, seed)
        fit = continuous.fit_continuous(fit_recs, mu0, C.NUM_STATES, Tr.coeffs(), 25, outer, 3, 1e-6, CPU, deadline)
        if fit.outer_status == "invalid":
            raise Tr.InvalidSolve("continuous fit invalid")
        mu, outer_status, outer_it, trace = fit.params, fit.outer_status, fit.outer_iterations, list(fit.objective_trace)
    else:
        mu0, mu, outer_status, outer_it, trace = None, np.asarray(mu_fixed, dtype=np.float64), "fixed", 0, []
    pop = population(eval_names, lay)
    cost = np.concatenate([continuous.continuous_cost_fn(mu, r) for r in eval_recs])
    dec = Dc.decode_ladder(cost, pop, rungs, deadline)
    final = rungs[-1]
    states = dec["states"][final]
    gts = Ev.load_labels(eval_names)
    starts = {n: lay[n]["starts"] for n in eval_names}
    fcs = {n: lay[n]["frame_count"] for n in eval_names}
    sc = Ev.score_states(states, pop, gts, starts, fcs)
    sc["balanced_accuracy"] = 100.0 * float(np.mean(list(sc["per_class_recall"].values())))
    sc["association"] = Ev.association(states, pop, gts, starts, fcs)
    sc["solver"] = dict(status_counts={str(k): v for k, v in dec["status_counts"].items()}, outer_status=outer_status, outer_iterations=outer_it,
                        final_status=dec["status"], final_steps=dec["steps"], final_residual=dec["residual"])
    sc["seconds"] = time.perf_counter() - t0
    frames = Ev.frame_predictions(states, pop, starts, fcs)
    mapping = {int(k): int(v) for k, v in sc["mapping"].items()}
    sc["artifacts"] = dict(mu=mu, mu0=mu0, cost_eval=cost.astype(np.float32), outer_trace=trace,
                           states={n: np.asarray(v, dtype=np.int16) for n, v in states.items()},
                           first_rung_states={n: np.asarray(v, dtype=np.int16) for n, v in dec["states"][rungs[0]].items()},
                           frame_predictions_raw={n: v.astype(np.int16) for n, v in frames.items()},
                           frame_predictions_mapped={n: np.array([mapping.get(int(x), -1) for x in v], dtype=np.int16) for n, v in frames.items()})
    return sc


def save_decode(path: Path, sc: dict) -> dict:
    """Write a decode's artifacts to one .npz and return the JSON-safe remainder of `sc`."""
    art = sc.pop("artifacts")
    arrays = {"mu": np.asarray(art["mu"]), "cost_eval": art["cost_eval"], "outer_trace": np.asarray(art["outer_trace"], dtype=np.float64)}
    if art["mu0"] is not None:
        arrays["mu0"] = np.asarray(art["mu0"])
    for key in ("states", "first_rung_states", "frame_predictions_raw", "frame_predictions_mapped"):
        for n, v in art[key].items():
            arrays[f"{key}/{n}"] = v
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)
    return sc


# --- shortcut diagnostic --------------------------------------------------------------------------------------------

def support_radius_windows() -> int:
    """Windows whose source supports may overlap: MotionBERT averages 243-frame clips at 60 fps (121.5 source frames at 30 fps), so a
    frame's feature can draw on frames up to ~121.5 away; a window adds its own 8 frames and the learned encoder +-4 windows. Two windows
    farther apart than 2 x (ceil(121.5 / 8) + 1 + 4) = 42 windows share no source frame. The same radius is used for every arm."""
    from script.pretrained_encoder import encoder as E
    clip_src_frames = E.CLIP / (E.MB_FPS / E.SRC_FPS)          # 121.5 frames at 30 fps; a frame can sit at either end of a clip
    return int(2 * (int(np.ceil(clip_src_frames / C.WINDOW)) + 1 + 4))


def nn_share(z_eval: dict, seed: int = 0, n_query: int = 3000) -> dict:
    """Same-recording nearest-neighbour share (cosine) on held-out windows at two exclusion radii:
      shared_context - excludes +-NN_SHARED_CONTEXT_RADIUS windows (neighbours may still share MotionBERT source frames);
      support_disjoint - excludes every same-recording window whose source support may overlap the query's.
    BABEL recordings are short, so at the support-disjoint radius many queries have no eligible same-recording candidate; the share
    is reported over all queries and over the queries that have at least one eligible same-recording candidate, with that count."""
    names = sorted(z_eval)
    feats = np.concatenate([z_eval[n] for n in names]).astype(np.float64)
    rec = np.concatenate([np.full(len(z_eval[n]), i) for i, n in enumerate(names)])
    pos = np.concatenate([np.arange(len(z_eval[n])) for n in names])
    norms = np.linalg.norm(feats, axis=1)
    ok = np.flatnonzero(norms > 1e-9)
    U = feats / np.where(norms > 1e-9, norms, 1.0)[:, None]
    rng = np.random.default_rng(seed)
    query = np.sort(rng.choice(ok, size=min(n_query, len(ok)), replace=False))
    out = {}
    for label, radius in (("shared_context", C.NN_SHARED_CONTEXT_RADIUS), ("support_disjoint", support_radius_windows())):
        same_all = eligible = same_eligible = 0
        for s in range(0, len(query), 500):
            q = query[s:s + 500]
            sim = U[q] @ U[ok].T
            same_rec = rec[q][:, None] == rec[ok][None, :]
            near = same_rec & (np.abs(pos[q][:, None] - pos[ok][None, :]) <= radius)
            sim[near] = -np.inf
            best = ok[np.argmax(sim, axis=1)]
            hit = rec[best] == rec[q]
            has = (same_rec & ~near).any(axis=1)
            same_all += int(hit.sum())
            eligible += int(has.sum())
            same_eligible += int((hit & has).sum())
        out[label] = dict(radius=radius, share_all=same_all / len(query), queries=len(query), queries_with_eligible=eligible,
                          share_eligible=(same_eligible / eligible) if eligible else None)
    return out


# --- inputs ----------------------------------------------------------------------------------------------------------

def f1_inputs(seed: int, fit_names: list, eval_names: list) -> dict:
    """F1: frozen MotionBERT canonical window statistics -> scaler and PCA-64 fitted on THIS FOLD's fitting windows only (a seeded
    sample of at most F1_SAMPLE_CAP complete windows) -> projections of every window. No held-out window enters any fitted step."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from script.self_labelled import augment
    canon = augment.canonical_windows(fit_names + eval_names)
    fit_all = np.concatenate([canon[n] for n in fit_names])
    rng = np.random.default_rng([seed, 23])
    sample = fit_all[np.sort(rng.choice(len(fit_all), min(C.F1_SAMPLE_CAP, len(fit_all)), replace=False))]
    scaler = StandardScaler().fit(sample)
    pca = PCA(n_components=C.F1_PCA_DIM, svd_solver="randomized", random_state=seed).fit(scaler.transform(sample))
    return {n: pca.transform(scaler.transform(canon[n])).astype(np.float64) for n in fit_names + eval_names}


# --- F0 (historical, declared transductive) -----------------------------------------------------------------------

def f0_cell_dir(seed: int, fold: str, run_dir: Path) -> Path | None:
    for base in (C.PE_RUN / "b0__mb", run_dir / "b0__mb"):
        d = base / "cells" / f"b0_centre__seed{seed}__{fold}"
        if (d / "scores.json").exists():
            return d
    return None


def f0_verify(seed: int, fold: str, run_dir: Path) -> dict:
    """Check the F0 cell against the identity the current pretrained-encoder / Milestone 1 code would assign it (source preparation hash,
    code hash, fold, seed, scale). Must run in a process where nothing else has patched the Milestone 1 configuration."""
    from script.compositional_dynamics import config as CD
    from script.compositional_dynamics import data as Dm
    from script.compositional_dynamics import run as CR
    from script.pretrained_encoder import run as PR
    d = f0_cell_dir(seed, fold, run_dir)
    if d is None:
        return dict(ok=False, reason="missing")
    PR.apply_overrides("mb")
    expected = CR.cell_identity(seed, "b0_centre", CD.FULL, Dm.source_identity(seed)["recordings_and_all"], None, fold)
    info = json.loads((d / "info.json").read_text())
    ok = info.get("status") == "complete" and info.get("identity") == expected
    return dict(ok=bool(ok), reason=None if ok else f"identity {info.get('identity')} != expected {expected} or status {info.get('status')}",
                cell=str(d), identity=info.get("identity"), expected=expected)


def f0_scores(seed: int, fold: str, run_dir: Path):
    d = f0_cell_dir(seed, fold, run_dir)
    if d is None:
        return None, None
    sc = json.loads((d / "scores.json").read_text())
    return sc["eval"][str(sc["primary_endpoint_steps"])], json.loads((d / "diagnostics.json").read_text())


def f0_vectors(seed: int, names: list) -> dict:
    """F0's decoder inputs (lookup[code] - mean, as in the matched B0), for the shortcut diagnostic."""
    from script.pretrained_encoder import config as PC
    from script.quantized_input_comparison.artifacts import Loader
    loader = Loader(PC.prep_dir("mb", seed))
    fam = PC.family("mb")
    lookup, mean = loader.lookup(fam).astype(np.float64), loader.mean(fam).astype(np.float64)
    return {n: lookup[loader.recording(fam, n)["codes"]] - mean[None, :] for n in names}
