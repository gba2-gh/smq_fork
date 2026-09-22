"""
D8 -- if the boundaries are fine and the labels are not, what is in the codes?

Three measurements, all at patch level (one row per patch, labelled with the
majority GT class inside that patch), so they speak directly to the unit SMQ
actually emits:

D8a  What the 1:1 Hungarian constraint costs.  Score the same code sequence
     under (i) the published bijective matching and (ii) a many-to-one map that
     sends each code to its own majority GT class.  The gap is the part of the
     error created by forcing one code per class rather than by the codes.

D8b  Linear probes, 5-fold grouped by sequence (so no patch is tested against
     patches from its own recording):
       raw    : mean-pooled raw features of the patch  -> GT class
       latent : mean-pooled pre-quantisation latent    -> GT class
       code   : the one-hot code id                    -> GT class
     latent >> code means quantisation at K = #classes is throwing away action
     information the encoder had.  latent ~= code means the encoder never had
     it, and a bigger or hierarchical codebook cannot help.

D8c  NMI and purity of the code partition w.r.t. GT classes, plus code usage
     concentration -- a collapse check.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import normalized_mutual_info_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from script.repro.dump_predictions import build_model, DEFAULTS


def patch_rows(arr, patch, reduce="mean"):
    """Split (T, D) into patches along T and pool each one."""
    T = arr.shape[0]
    n = T // patch
    if n == 0:
        return np.empty((0, arr.shape[1]))
    a = arr[:n * patch].reshape(n, patch, -1)
    return a.mean(axis=1) if reduce == "mean" else a[:, 0, :]


def patch_labels(gt, patch):
    T = len(gt)
    n = T // patch
    out = np.empty(n, int)
    for i in range(n):
        v, c = np.unique(gt[i * patch:(i + 1) * patch], return_counts=True)
        out[i] = v[np.argmax(c)]
    return out


def probe(X, y, groups, seed=0):
    """5-fold grouped linear probe; returns balanced-ish top-1 accuracy."""
    if X.shape[1] == 0 or len(np.unique(y)) < 2:
        return float("nan")
    gkf = GroupKFold(n_splits=5)
    accs = []
    for tr, te in gkf.split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, multi_class="multinomial",
                                 n_jobs=-1, random_state=seed)
        clf.fit(sc.transform(X[tr]), y[tr])
        accs.append(float((clf.predict(sc.transform(X[te])) == y[te]).mean()))
    return round(100 * float(np.mean(accs)), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DEFAULTS))
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--preds", required=True, type=Path)
    ap.add_argument("--data_root", type=Path, default=Path("data"))
    ap.add_argument("--max_seqs", type=int, default=250)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    d = np.load(args.preds, allow_pickle=True)
    patch = int(d["patch_size"])
    gts = [np.asarray(g) for g in d["gt"]]
    prs = [np.asarray(p) for p in d["pred"]]
    names = [str(n) for n in d["names"]]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = args.data_root / args.dataset
    model, K, _ = build_model(args.dataset, root / "groundTruth")
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.to(device).eval()

    idx = np.linspace(0, len(names) - 1, min(args.max_seqs, len(names))).astype(int)
    Xr, Xl, Xc, Y, G = [], [], [], [], []
    with torch.no_grad():
        for gi, i in enumerate(idx):
            gt = gts[i]
            if len(gt) < patch:
                continue
            feats = np.load(root / "features" / names[i])
            x = torch.tensor(feats, dtype=torch.float).unsqueeze(0).to(device)
            _ = model(x, torch.ones(x.size(), device=device))
            N, C, T, V, M = x.size()
            lat = model.latent.view(N * M, V, model.latent_dim, T)
            lat = lat.permute(0, 3, 1, 2).reshape(T, -1).cpu().numpy()
            raw = feats.transpose(1, 0, 2, 3).reshape(T, -1)

            yl = patch_labels(gt, patch)
            Xr.append(patch_rows(raw, patch))
            Xl.append(patch_rows(lat, patch))
            Xc.append(prs[i][::patch][:len(yl)])
            Y.append(yl)
            G.append(np.full(len(yl), gi))

    Xr = np.concatenate(Xr); Xl = np.concatenate(Xl)
    Xc = np.concatenate(Xc); Y = np.concatenate(Y); G = np.concatenate(G)
    n = min(len(Xr), len(Xl), len(Xc), len(Y))
    Xr, Xl, Xc, Y, G = Xr[:n], Xl[:n], Xc[:n], Y[:n], G[:n]

    res = {"dataset": args.dataset, "ckpt": str(args.ckpt), "patch_size": patch,
           "codebook_size": int(K), "n_patches": int(n),
           "n_gt_classes": int(len(np.unique(Y)))}

    # ---- D8a: what the bijection costs, at patch level -------------------
    maj = {}
    for c in np.unique(Xc):
        v, cnt = np.unique(Y[Xc == c], return_counts=True)
        maj[c] = v[np.argmax(cnt)]
    many_to_one = np.array([maj[c] for c in Xc])
    res["d8a_patch_level"] = dict(
        many_to_one_acc=round(100 * float((many_to_one == Y).mean()), 2),
        majority_class_baseline=round(
            100 * float(np.bincount(Y).max() / len(Y)), 2),
        n_codes_used=int(len(np.unique(Xc))),
        n_distinct_classes_claimed=int(len(set(maj.values()))),
    )

    # ---- D8b: probes -----------------------------------------------------
    onehot = np.eye(int(K))[Xc.astype(int)]
    res["d8b_probes"] = dict(
        raw_features=probe(Xr, Y, G),
        latent_prequant=probe(Xl, Y, G),
        code_onehot=probe(onehot, Y, G),
        chance_majority=res["d8a_patch_level"]["majority_class_baseline"],
    )

    # ---- D8c: partition quality / collapse -------------------------------
    cnt = np.bincount(Xc.astype(int), minlength=int(K)).astype(float)
    p = cnt / cnt.sum()
    res["d8c_codebook"] = dict(
        nmi_code_vs_gt=round(float(normalized_mutual_info_score(Y, Xc)), 3),
        perplexity=round(float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum())), 2),
        max_perplexity=int(K),
        top1_code_share=round(float(p.max()), 3),
        dead_codes=int((cnt == 0).sum()),
    )

    print(json.dumps(res, indent=2))
    out = args.out or Path(f"results/diagnostics/what_codes_encode_{args.dataset}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
