"""Freeze manifests and compute outcome-independent descriptor calibration."""
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/characterization"
SEEDS = [1538574472, 111, 222]
SALT = "smq-characterization-20260921"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def subject(ds, name):
    pattern = r"HuGaDB_v2_various_(\d+)_" if ds == "hugadb" else r"L\d+_S(\d+)_"
    return re.match(pattern, name).group(1)


def descriptors(x, scale, fps):
    z = x / scale
    displacement = z - z[:, :1]
    velocity = np.diff(z, axis=1) * fps
    velocity = np.concatenate((velocity[:, :1], velocity), axis=1)
    return displacement, velocity


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"seeds": SEEDS, "training_exposure": "All native recordings seen by frozen SMQ",
                "protocol_sha256": digest(OUT / "PROTOCOL.md"), "datasets": {}}
    for ds, W, K in (("lara", 50, 8), ("hugadb", 60, 10)):
        names = sorted(p.name for p in (ROOT / "data" / ds / "features").glob("*.npy"))
        subjects = [subject(ds, n) for n in names]
        ordered = sorted(set(subjects), key=lambda s: hashlib.sha256(f"{SALT}/{ds}/{s}".encode()).hexdigest())
        nfit = len(ordered) // 3
        fit, evaluation = ordered[:nfit], ordered[nfit:]
        dyads = [evaluation[i:i+2] for i in range(0, len(evaluation)-1, 2)]
        checkpoints = []
        for seed in SEEDS:
            ckpt = ROOT / f"models/exp/T1/{ds}/K{K}_s{seed}/{ds}/epoch-30.model"
            dump = ROOT / f"results/exp/dumps/T1/{ds}/K{K}_s{seed}.npz"
            config = ckpt.parent / "config.json"
            checkpoints.append(dict(seed=seed, checkpoint=str(ckpt.relative_to(ROOT)),
                checkpoint_sha256=digest(ckpt), dump=str(dump.relative_to(ROOT)), dump_sha256=digest(dump),
                config=json.loads(config.read_text())))
        manifest["datasets"][ds] = dict(W=W, K=K, fps=W, calibration_subjects=fit,
            evaluation_subjects=evaluation, primary_dyads=dyads,
            unpaired_subjects=evaluation[2*len(dyads):], checkpoints=checkpoints,
            recordings=[dict(name=n, subject=s, split="calibration" if s in fit else "evaluation")
                        for n, s in zip(names, subjects)])
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Manifest frozen before descriptors/grouping outcomes", flush=True)

    for ds in ("lara", "hugadb"):
        if len(sys.argv) > 1 and ds not in sys.argv[1:]:
            continue
        cfg = manifest["datasets"][ds]
        W = cfg["W"]
        names = [r["name"] for r in cfg["recordings"]]
        subjects = np.array([r["subject"] for r in cfg["recordings"]])
        mapping = {}
        for line in (ROOT / f"data/{ds}/mapping/mapping.txt").read_text().splitlines():
            i, name = line.split(" ", 1)
            mapping[name] = int(i)
        exports = [np.load(ROOT / c["dump"], allow_pickle=True) for c in cfg["checkpoints"]]
        export_maps = [{str(n): i for i, n in enumerate(e["names"])} for e in exports]
        export_codes = [e["codes"] for e in exports]
        traj, recs, starts, actions, purities, codes = [], [], [], [], [], []
        hashes = []
        for ri, name in enumerate(names):
            path = ROOT / f"data/{ds}/features/{name}"
            raw = np.load(path)
            hashes.append(dict(name=name, features_sha256=digest(path)))
            labels = np.array([mapping[v] for v in (ROOT / f"data/{ds}/groundTruth/{Path(name).stem}.txt").read_text().splitlines()])
            assert len(labels) == raw.shape[1]
            if ds == "lara":
                assert np.all(raw[3:6, :, 21, 0] == 0)
                x = raw[3:6, :, :21, 0].transpose(1, 2, 0).reshape(raw.shape[1], -1)
            else:
                x = raw[:, :, :, 0].transpose(1, 0, 2).reshape(raw.shape[1], -1)
            for p in range(1, len(x)//W, 2):
                start = p * W
                if start + W + 14 > len(x):
                    continue
                v, counts = np.unique(labels[start:start+W], return_counts=True)
                traj.append(x[start:start+W].astype(np.float32))
                recs.append(ri); starts.append(start)
                actions.append(v[counts.argmax()]); purities.append(counts.max()/W)
                codes.append([int(c[m[name]][p]) for c, m in zip(export_codes, export_maps)])
        trajectory = np.stack(traj)
        del traj
        recs = np.array(recs, dtype=np.int32)
        is_eval = np.isin(subjects[recs], cfg["evaluation_subjects"])
        means = trajectory.mean(axis=1)
        center = means[~is_eval].mean(0)
        sd = means[~is_eval].std(0)
        floor = .05 * np.median(sd[sd > 0])
        sd = np.maximum(sd, floor).astype(np.float32)
        if ds == "lara":
            noise_scale = np.sqrt(np.mean(trajectory[~is_eval].astype(np.float64)**2))
        else:
            noise_scale = trajectory[~is_eval].reshape(-1, 36).std(0).reshape(6, 6)
        rng = np.random.default_rng(20260921)
        fitidx = np.flatnonzero(~is_eval)
        aa, bb = [], []
        while len(aa) < 5000:
            a, b = rng.choice(fitidx, 2, replace=False)
            if subjects[recs[a]] != subjects[recs[b]]:
                aa.append(a); bb.append(b)
        da, va = descriptors(trajectory[aa], sd, W)
        db, vb = descriptors(trajectory[bb], sd, W)
        distance_scales = np.array([np.median(np.sqrt(np.mean((da-db)**2, axis=(1,2)))),
                                    np.median(np.sqrt(np.mean((va-vb)**2, axis=(1,2))))])
        assert np.all(distance_scales > 0)
        np.savez_compressed(OUT / f"{ds}_cache.npz", names=np.array(names), subjects=subjects,
            rec=recs, start=np.array(starts), evaluation=is_eval, trajectory=trajectory,
            codes=np.array(codes), action=np.array(actions), purity=np.array(purities),
            posture_center=center, posture_scale=sd, noise_scale=noise_scale,
            distance_scales=distance_scales, fps=W, W=W, K=cfg["K"])
        descriptor_info = dict(dataset=ds, patches=len(recs), calibration_patches=int((~is_eval).sum()),
            evaluation_patches=int(is_eval.sum()), distance_scales=distance_scales.tolist(),
            posture_sd_floor=float(floor), noise_scale=np.asarray(noise_scale).tolist(),
            data_hashes=hashes, cache_sha256=digest(OUT/f"{ds}_cache.npz"))
        (OUT/f"{ds}_calibration.json").write_text(json.dumps(descriptor_info, indent=2))
        print(f"{ds}: cache ready ({len(recs)} patches)", flush=True)
        for e in exports:
            e.close()


if __name__ == "__main__":
    main()
