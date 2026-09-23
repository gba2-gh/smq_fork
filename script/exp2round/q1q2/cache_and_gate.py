"""Create/verify released-checkpoint latent caches and apply the provenance gates."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import sklearn
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from script.exp2round.dump_latents import encoder_latent_tvd, quantise_dumped_latent
from script.exp2round.q1q2.core import (
    CHECKPOINTS, D7_REFERENCE, EXPECTED_HASH_PREFIX, OUT, ROOT, atomic_json,
    majority_code_predictions, map_predictions, score, sha256_file,
)
from script.repro.dump_predictions import build_model
from src.model.eval_utils import read_mapping_file


def extract(dataset: str, output: Path) -> None:
    checkpoint = CHECKPOINTS[dataset]
    root = ROOT / "data" / dataset
    model, num_actions, patch_size = build_model(dataset, root / "groundTruth")
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.cpu().eval()
    mapping = read_mapping_file(root / "mapping" / "mapping.txt")
    actions = {value: key for key, value in mapping.items()}
    names = sorted(path.name for path in (root / "features").glob("*.npy"))
    codes = []
    output.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for index, name in enumerate(names):
            features = np.load(root / "features" / name)
            gt = np.asarray([actions[value] for value in
                             (root / "groundTruth" / name.replace("npy", "txt"))
                             .read_text().splitlines()], dtype=np.int16)
            x = torch.as_tensor(features, dtype=torch.float32).unsqueeze(0)
            model(x, torch.ones_like(x))
            latent_tvd = encoder_latent_tvd(model, len(gt)).numpy().astype(np.float32)
            latent = latent_tvd.reshape(len(gt), -1)
            direct = model.indices.squeeze(0).cpu().numpy().astype(np.int16)
            reconstructed = quantise_dumped_latent(
                latent_tvd, model.vq._embedding.detach().cpu(), patch_size
            )
            if not np.array_equal(direct, reconstructed):
                raise RuntimeError(f"same-checkpoint quantisation mismatch: {name}")
            np.save(output / f"{Path(name).stem}.npy", latent)
            codes.append(direct)
            if (index + 1) % 25 == 0:
                print(f"[{dataset}] cached {index + 1}/{len(names)}", flush=True)
    object_codes = np.empty(len(codes), dtype=object)
    object_codes[:] = codes
    np.savez_compressed(
        output / "_meta.npz", names=np.asarray([Path(name).stem for name in names]),
        codes=object_codes, codebook=model.vq._embedding.detach().cpu().numpy(),
        ckpt_sha256=sha256_file(checkpoint), mismatch=0,
        total=sum(map(len, codes)), patch_size=patch_size, num_actions=num_actions,
    )


def cache_path(dataset: str) -> Path:
    preferred = OUT / "cache" / dataset
    existing = ROOT / "results" / "seg" / "cache" / dataset
    expected = sha256_file(CHECKPOINTS[dataset])
    for candidate in (preferred, existing):
        meta_path = candidate / "_meta.npz"
        if meta_path.exists():
            meta = np.load(meta_path, allow_pickle=True)
            if str(meta["ckpt_sha256"]) == expected:
                return candidate
    extract(dataset, preferred)
    return preferred


def verify(dataset: str) -> dict:
    started = time.perf_counter()
    checkpoint_hash = sha256_file(CHECKPOINTS[dataset])
    if not checkpoint_hash.startswith(EXPECTED_HASH_PREFIX[dataset]):
        raise RuntimeError(f"unexpected {dataset} checkpoint hash {checkpoint_hash}")
    cache = cache_path(dataset)
    meta = np.load(cache / "_meta.npz", allow_pickle=True)
    names = [str(value) for value in meta["names"]]
    codes = [np.asarray(value, dtype=np.int32) for value in meta["codes"]]
    codebook = torch.as_tensor(meta["codebook"], dtype=torch.float32)
    pred_path = ROOT / "results" / "preds" / f"{dataset}_pretrained.npz"
    historical = np.load(pred_path, allow_pickle=True)
    historical_names = [Path(str(value)).stem for value in historical["names"]]
    if names != historical_names:
        raise RuntimeError(f"{dataset} recording order differs from historical dump")
    gts = [np.asarray(value, dtype=np.int32) for value in historical["gt"]]
    same_matches = same_total = 0
    for name, code in zip(names, codes):
        latent = np.load(cache / f"{name}.npy", mmap_mode="r")
        reconstructed = quantise_dumped_latent(
            np.asarray(latent).reshape(len(latent), -1, 16), codebook,
            int(historical["patch_size"]),
        )
        same_matches += int(np.count_nonzero(reconstructed == code))
        same_total += len(code)
    same_agreement = same_matches / same_total
    historic_matches = sum(int(np.count_nonzero(code == np.asarray(old)))
                           for code, old in zip(codes, historical["pred"]))
    historical_agreement = historic_matches / same_total
    majority_predictions = majority_code_predictions(gts, codes)
    d7 = score(gts, map_predictions(gts, majority_predictions, "hungarian"))
    discrepancy = abs(d7["MoF"] - D7_REFERENCE[dataset])
    result = {
        "dataset": dataset,
        "checkpoint": str(CHECKPOINTS[dataset].relative_to(ROOT)),
        "checkpoint_sha256": checkpoint_hash,
        "cache": str(cache.relative_to(ROOT)),
        "cache_identity": f"{cache}:{checkpoint_hash}",
        "n_recordings": len(names),
        "n_frames": same_total,
        "same_checkpoint_quantisation_agreement": same_agreement,
        "historical_prediction_agreement": historical_agreement,
        "historical_agreement_threshold": 0.999,
        "d7": d7,
        "d7_reference_mof": D7_REFERENCE[dataset],
        "d7_absolute_difference": discrepancy,
        "passed": same_agreement == 1.0 and historical_agreement >= 0.999 and discrepancy <= 1.5,
        "runtime_seconds": time.perf_counter() - started,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["hugadb", "lara"])
    args = parser.parse_args()
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    OUT.mkdir(parents=True, exist_ok=True)
    gates = []
    for dataset in args.datasets:
        result = verify(dataset)
        gates.append(result)
        atomic_json(OUT / f"gate_{dataset}.json", result)
        print(json.dumps(result, indent=2), flush=True)
    manifest = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "root": str(ROOT),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {"numpy": np.__version__, "sklearn": sklearn.__version__,
                     "torch": torch.__version__},
        "cpu_thread_cap": min(8, os.cpu_count() or 1),
        "seeds": [111, 222, 1538574472],
        "permutation_seeds": list(range(111, 131)),
        "gates": gates,
    }
    atomic_json(OUT / "manifest.json", manifest)
    if not all(gate["passed"] for gate in gates):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
