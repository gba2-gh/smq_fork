"""Preparation access, source verification and flat window populations.

Everything here is label-free. Action labels are read only by evaluate.py."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from script.compositional_dynamics import config as C
from script.quantized_input_comparison.artifacts import Loader


class PreflightError(RuntimeError):
    pass


def _sha(path) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


@dataclass
class Population:
    """Windows of a set of recordings, concatenated, with finite causal histories."""
    names: list
    codes: np.ndarray        # int64 [N]
    lengths: np.ndarray      # int64 [N] real frames per window
    rec_idx: np.ndarray      # int64 [N] index into names
    pos: np.ndarray          # int64 [N] window index inside its recording
    offsets: np.ndarray      # int64 [R+1]
    hist: np.ndarray         # int64 [N, H] preceding codes, right-aligned
    hmask: np.ndarray        # bool  [N, H] True where hist is a real preceding code
    hist_len: int

    @property
    def n(self) -> int:
        return len(self.codes)

    def slice_of(self, r: int) -> slice:
        return slice(int(self.offsets[r]), int(self.offsets[r + 1]))


def make_population(names, codes_list, lengths_list, hist_len: int) -> Population:
    names = list(names)
    codes = np.concatenate([np.asarray(c, dtype=np.int64) for c in codes_list])
    lengths = np.concatenate([np.asarray(x, dtype=np.int64) for x in lengths_list])
    sizes = np.array([len(c) for c in codes_list], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    rec_idx = np.repeat(np.arange(len(names)), sizes)
    pos = np.arange(len(codes)) - offsets[rec_idx]
    hist = np.zeros((len(codes), hist_len), dtype=np.int64)
    hmask = np.zeros((len(codes), hist_len), dtype=bool)
    for k in range(1, hist_len + 1):
        valid = pos >= k
        src = np.where(valid, np.arange(len(codes)) - k, 0)
        hist[:, hist_len - k] = np.where(valid, codes[src], 0)
        hmask[:, hist_len - k] = valid
    return Population(names, codes, lengths, rec_idx, pos, offsets, hist, hmask, hist_len)


def check_coverage(starts: np.ndarray, lengths: np.ndarray, frame_count: int, window: int, name: str = "") -> None:
    """Every real frame is covered exactly once, in order; only the terminal window may be short."""
    starts, lengths = np.asarray(starts, dtype=np.int64), np.asarray(lengths, dtype=np.int64)
    ends = starts + lengths
    ok = (len(starts) > 0 and starts[0] == 0 and bool(np.all(starts[1:] == ends[:-1])) and int(ends[-1]) == frame_count
          and bool(np.all(lengths > 0)) and bool(np.all(lengths[:-1] == window)) and int(lengths[-1]) <= window)
    if not ok:
        raise PreflightError(f"{name}: windows do not cover [0, {frame_count}) exactly once")


def subject_of(name: str) -> str:
    from script.week5.raw_kmeans_cross_dataset import datasets as D
    return D.subject_of(D.SPECS[C.DATASET], name)


def subject_folds(names, n_folds: int = C.SUBJECT_FOLDS) -> list:
    """Label-free subject-disjoint folds: subjects (parsed from file names) shuffled with the fixed seed, dealt round-robin."""
    subjects = sorted({subject_of(n) for n in names})
    order = np.random.default_rng(C.CALIB_SEED).permutation(len(subjects))
    return [sorted(subjects[i] for i in order[k::n_folds]) for k in range(n_folds)]


def split_names(loader: Loader, split: str):
    """(fitting names, evaluation names). 'primary' is the preparation's existing split; 'foldK' holds out fold K's subjects
    and fits on every other recording (including the primary evaluation recordings of the remaining subjects)."""
    if split == "primary":
        return sorted(loader.split["train"]), sorted(loader.split["val"])
    k = int(split[4:])
    names = sorted(loader.recording_names())
    held = set(subject_folds(names)[k])
    return ([n for n in names if subject_of(n) not in held], [n for n in names if subject_of(n) in held])


@dataclass
class CellData:
    seed: int
    loader: Loader
    fit: Population
    eval: Population
    fit_names: list
    eval_names: list
    starts: dict             # name -> starts
    frame_counts: dict
    lookup_raw: np.ndarray   # [K, 64] saved lookup
    mean: np.ndarray
    rms: float
    lookup_fixed: np.ndarray # [K, 64] float32: (lookup - mean) / rms, the model's fixed input vectors
    source_hash: str
    smoke: bool
    split: str = "primary"


def source_identity(seed: int) -> dict:
    d = C.prep_dir(seed)
    fam = d / "families" / C.FAMILY
    files = {"manifest": d / "manifest.json", "split": d / "split.json", "lookup": fam / "lookup.npy",
             "mean": fam / "mean.npy", "rms": fam / "rms.npy", "grouping": fam / "grouping.npy"}
    for k, p in files.items():
        if not p.exists():
            raise PreflightError(f"missing source {k}: {p}")
    out = {k: _sha(p) for k, p in files.items()}
    h = hashlib.sha256(json.dumps(out, sort_keys=True).encode())
    for p in sorted((fam / "recordings").glob("*.npz")):
        h.update(p.name.encode() + _sha(p).encode())
    out["recordings_and_all"] = h.hexdigest()[:16]
    return out


def verify_source(seed: int) -> dict:
    """Preflight facts for one seed's preparation. Raises PreflightError on any disagreement."""
    d = C.prep_dir(seed)
    ident = source_identity(seed)
    loader = Loader(d)
    man = loader.manifest
    key = man["prep_key"]
    want = dict(dataset=C.DATASET, tokenizer=C.TOKENIZER, window=C.WINDOW, num_states=C.NUM_STATES, num_codes=C.NUM_CODES,
                seed=seed, split_seed=111)
    bad = {k: (key.get(k), v) for k, v in want.items() if key.get(k) != v}
    if bad:
        raise PreflightError(f"manifest disagrees with declared settings (got, want): {bad}")
    if C.FAMILY not in man["families"]:
        raise PreflightError(f"family {C.FAMILY} not in manifest")
    if ident["split"] != C.SPLIT_SHA256:
        raise PreflightError(f"split.json hash {ident['split']} != declared {C.SPLIT_SHA256}")
    fit, ev = list(loader.split["train"]), list(loader.split["val"])
    names = loader.recording_names()
    if len(fit) != C.N_FIT or len(ev) != C.N_EVAL or set(fit) & set(ev) or set(fit) | set(ev) != set(names):
        raise PreflightError("split is not a disjoint 395/44 partition of the manifest recordings")
    subj = {n: subject_of(n) for n in names}
    lookup = loader.lookup(C.FAMILY)
    if lookup.shape != (C.NUM_CODES, C.LOOKUP_DIM):
        raise PreflightError(f"lookup shape {lookup.shape}")
    n_windows = 0
    for n in names:
        r = loader.recording(C.FAMILY, n)
        check_coverage(r["starts"], r["lengths"], r["frame_count"], C.WINDOW, n)
        if len(r["codes"]) != len(r["starts"]) or r["codes"].min() < 0 or r["codes"].max() >= C.NUM_CODES:
            raise PreflightError(f"{n}: codes out of range or misaligned with starts")
        n_windows += len(r["codes"])
    folds = subject_folds(names)
    held = [set(f) for f in folds]
    if sum(len(f) for f in folds) != len(set(subj.values())) or any(a & b for i, a in enumerate(held) for b in held[i + 1:]):
        raise PreflightError("subject folds are not a disjoint cover of the subjects")
    fold_sizes = []
    for k in range(C.SUBJECT_FOLDS):
        fit_k, eval_k = split_names(loader, f"fold{k}")
        if set(fit_k) & set(eval_k) or len(fit_k) + len(eval_k) != len(names) or not eval_k:
            raise PreflightError(f"fold{k} is not a disjoint partition of the recordings")
        if {subj[n] for n in fit_k} & {subj[n] for n in eval_k}:
            raise PreflightError(f"fold{k} shares subjects between fitting and evaluation")
        fold_sizes.append([len(fit_k), len(eval_k)])
    return dict(seed=seed, identity=ident, n_recordings=len(names), subject_folds=folds, fold_sizes_fit_eval=fold_sizes, n_fit=len(fit), n_eval=len(ev), n_windows=n_windows,
                subjects=len(set(subj.values())), subjects_fit=len({subj[n] for n in fit}), subjects_eval=len({subj[n] for n in ev}),
                hist_len=C.history_length(), lookup_shape=list(lookup.shape))


def load_cell_data(seed: int, scale: C.Scale, split: str = "primary") -> CellData:
    d = C.prep_dir(seed)
    ident = source_identity(seed)
    loader = Loader(d)
    fit_names, eval_names = split_names(loader, split)
    if scale.smoke_fit:
        rng = np.random.default_rng(C.CALIB_SEED)
        fit_names = sorted(rng.choice(fit_names, scale.smoke_fit, replace=False).tolist())
        eval_names = sorted(rng.choice(eval_names, scale.smoke_eval, replace=False).tolist())
    H = C.history_length()
    pops, starts, fcs = {}, {}, {}
    for key, names in (("fit", fit_names), ("eval", eval_names)):
        recs = [loader.recording(C.FAMILY, n) for n in names]
        for n, r in zip(names, recs):
            starts[n], fcs[n] = np.asarray(r["starts"], dtype=np.int64), int(r["frame_count"])
        pops[key] = make_population(names, [r["codes"] for r in recs], [r["lengths"] for r in recs], H)
    lookup, mean, rms = loader.lookup(C.FAMILY), loader.mean(C.FAMILY).astype(np.float64), float(loader.rms(C.FAMILY))
    fixed = ((lookup.astype(np.float64) - mean[None, :]) / rms).astype(np.float32)
    return CellData(seed, loader, pops["fit"], pops["eval"], fit_names, eval_names, starts, fcs, lookup, mean, rms, fixed,
                    ident["recordings_and_all"], bool(scale.smoke_fit), split)


def recording_inputs(pop: Population):
    """categorical.RecordingInput list for a population (codes, real lengths, fps, window)."""
    from script.action_transport.categorical import RecordingInput
    return [RecordingInput(n, pop.codes[pop.slice_of(r)].astype(np.int32), pop.lengths[pop.slice_of(r)].astype(np.int32),
                           C.FPS, C.WINDOW) for r, n in enumerate(pop.names)]
