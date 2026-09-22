"""
Build job files for the sequential runner, exactly as pre-registered in
results/PREREG_T1-T4.md.

    python script/exp/make_jobs.py T1          -> models/exp/jobs_T1.jsonl
    python script/exp/make_jobs.py T2          -> models/exp/jobs_T2.jsonl (needs T1 K=C seed-1538574472 losses)
    python script/exp/make_jobs.py T4          -> models/exp/jobs_T4.jsonl
"""
import json
import math
import sys
from pathlib import Path

SEEDS = ["1538574472", "111", "222"]
DATASETS = ["hugadb", "lara", "babel1"]
C = {"hugadb": 10, "lara": 8, "babel1": 5}
FPS = {"hugadb": 60, "lara": 50, "babel1": 30}
GPU_CAP = "0.5"


def train_step(ds, root, extra, data_root="data"):
    return {"cmd": ["main.py", "--action", "train", "--dataset", ds, "--models_root", root,
                    "--data_root", data_root, "--epoch", "30", "--save_every", "30",
                    "--grad_checkpoint", "--gpu_mem_fraction", GPU_CAP, *extra],
            "creates": f"{root}/{ds}/epoch-30.model", "log": f"{root}/{ds}/train.log"}


def dump_step(ds, root, out, data_root="data"):
    return {"cmd": ["script/exp/dump_run.py", "--dataset", ds, "--ckpt", f"{root}/{ds}/epoch-30.model",
                    "--out", out, "--data_root", data_root, "--gpu_mem_fraction", GPU_CAP],
            "creates": out, "log": f"{root}/{ds}/dump.log"}


def job(name, ds, root, extra, dump_out, data_root="data"):
    return {"name": name, "steps": [train_step(ds, root, extra, data_root),
                                    dump_step(ds, root, dump_out, data_root)]}


def t1():
    jobs = []
    # released checkpoints, dumped with patch distances for T3
    for ds in ["lara", "babel1", "hugadb"]:
        out = f"results/exp/dumps/pretrained/{ds}.npz"
        jobs.append({"name": f"dump/pretrained/{ds}", "steps": [
            {"cmd": ["script/exp/dump_run.py", "--dataset", ds, "--ckpt", f"models/pretrained/{ds}.model",
                     "--out", out, "--gpu_mem_fraction", GPU_CAP],
             "creates": out, "log": f"results/exp/dumps/pretrained/{ds}.log"}]})

    def k_job(ds, mult, seed, thr=None):
        K = C[ds] * mult
        thr = thr if thr is not None else max(1, round(10 * C[ds] / K))
        tag = f"K{K}_s{seed}" + ("" if thr == max(1, round(10 * C[ds] / K)) else f"_thr{thr}")
        root = f"models/exp/T1/{ds}/{tag}"
        return job(f"T1/{ds}/{tag}", ds, root,
                   ["--seed", seed, "--num_actions", str(K), "--dead_code_threshold", str(thr)],
                   f"results/exp/dumps/T1/{ds}/{tag}.npz")

    # K=C baselines for the first seed come first: T2's weight is derived from them
    for ds in DATASETS:
        jobs.append(k_job(ds, 1, SEEDS[0]))
    for seed in SEEDS:
        for mult in [1, 2, 4, 8, 16]:
            for ds in DATASETS:
                j = k_job(ds, mult, seed)
                if j["name"] not in {x["name"] for x in jobs}:
                    jobs.append(j)
    # dead-code churn check: largest K with the published, unscaled threshold
    for ds in DATASETS:
        jobs.append(k_job(ds, 16, SEEDS[0], thr=10))
    return jobs


def t2():
    jobs = []
    for ds in DATASETS:
        base = Path(f"models/exp/T1/{ds}/K{C[ds]}_s{SEEDS[0]}/{ds}/losses.jsonl")
        last = json.loads(base.read_text().splitlines()[-1])
        lam = last["commit"] / math.log(C[ds])
        conds = {"tc1x": (lam, []), "tc10x": (10 * lam, []),
                 "tconly": (lam, ["--mse_loss_weight", "0"])}
        print(f"[T2] {ds}: final commit loss {last['commit']:.6g} -> lambda_1 = {lam:.6g}")
        for seed in SEEDS:
            for cond, (w, extra) in conds.items():
                tag = f"{cond}_s{seed}"
                root = f"models/exp/T2/{ds}/{tag}"
                jobs.append(job(f"T2/{ds}/{tag}", ds, root,
                                ["--seed", seed, "--tc_weight", f"{w:.8g}", *extra],
                                f"results/exp/dumps/T2/{ds}/{tag}.npz"))
    return jobs


RATES = {"hugadb": [2, 4], "lara": [2, 5], "babel1": [2, 3]}


def t4():
    jobs = []
    for seed in SEEDS:
        for ds in DATASETS:
            for dur, ps in (("0.5s", FPS[ds] // 2), ("2s", FPS[ds] * 2)):
                tag = f"P{dur}_s{seed}"
                root = f"models/exp/T4/{ds}/{tag}"
                jobs.append(job(f"T4/{ds}/{tag}", ds, root,
                                ["--seed", seed, "--patch_size", str(ps)],
                                f"results/exp/dumps/T4/{ds}/{tag}.npz"))
            for r in RATES[ds]:
                tag = f"R{r}_s{seed}"
                root = f"models/exp/T4/{ds}/{tag}"
                data_root = f"data/exp_rates/{ds}_r{r}"
                jobs.append(job(f"T4/{ds}/{tag}", ds, root,
                                ["--seed", seed, "--patch_size", str(FPS[ds] // r)],
                                f"results/exp/dumps/T4/{ds}/{tag}.npz", data_root))
    return jobs


if __name__ == "__main__":
    which = sys.argv[1]
    jobs = {"T1": t1, "T2": t2, "T4": t4}[which]()
    out = Path(f"models/exp/jobs_{which}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(j) + "\n" for j in jobs))
    print(f"[make_jobs] {which}: {len(jobs)} jobs -> {out}")
