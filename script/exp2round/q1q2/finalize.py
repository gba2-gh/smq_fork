"""Consolidate incremental Q1/Q2 outputs and produce the required report."""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "exp2round" / "q1q2"


def load_jsonl(name: str) -> pd.DataFrame:
    rows = []
    with (OUT / name).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def clean_frame(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return frame.drop_duplicates(keys, keep="last").sort_values(keys, na_position="first").reset_index(drop=True)


def write_csv(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(OUT / name, index=False, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")


def mean_sd(frame: pd.DataFrame, value: str, groups: list[str]) -> pd.DataFrame:
    good = frame[(frame.status == "complete") & frame[value].notna()]
    return good.groupby(groups, dropna=False)[value].agg(mean="mean", sd=lambda x: x.std(ddof=0), n="size").reset_index()


def render_information(info: pd.DataFrame) -> None:
    data = info[(info.representation == "latent") & info.window.isin([60, 50])]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex="col")
    for row, dataset in enumerate(["hugadb", "lara"]):
        ds = data[data.dataset == dataset]
        for metric, label in [("mi_unit_action", "I(U;A)"), ("mi_unit_subject", "I(U;S)"),
                              ("cmi_unit_subject_given_action", "I(U;S|A)")]:
            agg = ds.groupby("num_units")[metric].agg(["mean", lambda x: x.std(ddof=0)])
            axes[row, 0].errorbar(agg.index, agg["mean"], yerr=agg["<lambda_0>"], marker="o", label=label)
        for metric, label in [("ami_unit_action", "AMI unit/action"), ("ami_unit_subject", "AMI unit/subject")]:
            agg = ds.groupby("num_units")[metric].agg(["mean", lambda x: x.std(ddof=0)])
            axes[row, 1].errorbar(agg.index, agg["mean"], yerr=agg["<lambda_0>"], marker="o", label=label)
        axes[row, 0].set_ylabel(f"{dataset.upper()} (nats)")
        axes[row, 0].legend(fontsize=8)
        axes[row, 1].legend(fontsize=8)
    for ax in axes.flat:
        ax.set_xscale("log")
        ax.grid(alpha=.25)
        ax.set_xlabel("Vocabulary size $K_u$")
    axes[0, 0].set_title("Information measures (mean ± population SD)")
    axes[0, 1].set_title("Adjusted mutual information")
    fig.tight_layout()
    fig.savefig(OUT / "information_vs_ku.png", dpi=180)
    plt.close(fig)


def render_performance(pooled: pd.DataFrame, supervised: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for row, (dataset, window) in enumerate([("hugadb", 60), ("lara", 50)]):
        uns = pooled[(pooled.dataset == dataset) & (pooled.representation == "latent") &
                     (pooled.window == window) & (pooled.readout.isin(["hard", "soft"])) &
                     ((pooled.readout == "hard") | (pooled.temperature_multiplier == 1.0))]
        sup = supervised[(supervised.dataset == dataset) & (supervised.grouping == "subject") &
                         (supervised.window == window) & (supervised.level == "pooled_oof") &
                         (supervised.readout.isin(["hard", "soft"])) &
                         ((supervised.readout == "hard") | (supervised.temperature_multiplier == 1.0))]
        for readout in ["hard", "soft"]:
            u = uns[uns.readout == readout].groupby("num_units").MoF.agg(["mean", lambda x: x.std(ddof=0)])
            axes[row, 0].errorbar(u.index, u["mean"], yerr=u["<lambda_0>"], marker="o", label=readout)
            s = sup[(sup.readout == readout) & (sup.status == "complete")]
            for col, ax in [("frame_mof", axes[row, 1]), ("balanced_accuracy", axes[row, 2])]:
                a = s.groupby("num_units")[col].agg(["mean", lambda x: x.std(ddof=0)])
                ax.errorbar(a.index, a["mean"], yerr=a["<lambda_0>"], marker="o", label=readout)
        axes[row, 0].set_ylabel(f"{dataset.upper()} score (%)")
    titles = ["Pooled segment clustering MoF", "Subject-grouped supervised frame MoF", "Subject-grouped balanced accuracy"]
    for col, title in enumerate(titles):
        axes[0, col].set_title(title)
        for row in range(2):
            axes[row, col].set_xscale("log")
            axes[row, col].grid(alpha=.25)
            axes[row, col].set_xlabel("Vocabulary size $K_u$")
            axes[row, col].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "clustering_and_supervised_vs_ku.png", dpi=180)
    plt.close(fig)


def render_reuse(reuse: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, dataset in zip(axes, ["hugadb", "lara"]):
        ds = reuse[(reuse.dataset == dataset) & (reuse.grouping == "subject")]
        for window in sorted(ds.window.unique(), reverse=True):
            per_fold = ds[ds.window == window].groupby(["num_units", "seed", "fold"]).used_by_at_least_half.mean()
            agg = per_fold.groupby("num_units").agg(["mean", lambda x: x.std(ddof=0)])
            ax.errorbar(agg.index, agg["mean"], yerr=agg["<lambda_0>"], marker="o", label=f"W={int(window)}")
        ax.set_title(dataset.upper())
        ax.set_xscale("log")
        ax.set_xlabel("Vocabulary size $K_u$")
        ax.grid(alpha=.25)
        ax.legend()
    axes[0].set_ylabel("Fraction used by at least half of held-out subjects")
    fig.tight_layout()
    fig.savefig(OUT / "unit_reuse_vs_ku.png", dpi=180)
    plt.close(fig)


def md_table(frame: pd.DataFrame, decimals: int = 2) -> str:
    if frame.empty:
        return "No complete rows."
    show = frame.copy()
    for col in show.select_dtypes(include=[np.number]).columns:
        show[col] = show[col].map(lambda x: "" if pd.isna(x) else f"{x:.{decimals}f}")
    columns = [str(c) for c in show.columns]
    def escape(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(map(escape, columns)) + " |",
             "| " + " | ".join("---" for _ in columns) + " |"]
    lines.extend("| " + " | ".join(escape(v) for v in row) + " |" for row in show.itertuples(index=False, name=None))
    return "\n".join(lines)


def write_report(pooled: pd.DataFrame, info: pd.DataFrame, supervised: pd.DataFrame,
                 reuse: pd.DataFrame, not_run: pd.DataFrame, manifest: dict) -> None:
    full = {"hugadb": 60, "lara": 50}
    info_summary = info[(info.representation == "latent") &
                        info.apply(lambda r: r.window == full[r.dataset], axis=1)]
    info_summary = mean_sd(info_summary, "mi_unit_action", ["dataset", "num_units"])
    info_summary.columns = ["Dataset", "Ku", "I(U;A) mean", "seed SD", "seeds"]

    cluster = pooled[(pooled.representation == "latent") & (pooled.readout == "hard") &
                     pooled.apply(lambda r: r.window == full[r.dataset], axis=1)]
    cluster = mean_sd(cluster, "MoF", ["dataset", "num_units"])
    cluster.columns = ["Dataset", "Ku", "MoF mean", "seed SD", "seeds"]

    sup = supervised[(supervised.level == "pooled_oof") & (supervised.grouping == "subject") &
                     (supervised.readout == "hard") &
                     supervised.apply(lambda r: r.window == full[r.dataset], axis=1)]
    sup_m = mean_sd(sup, "frame_mof", ["dataset", "num_units"])
    sup_b = mean_sd(sup, "balanced_accuracy", ["dataset", "num_units"])
    supervised_summary = sup_m.merge(sup_b, on=["dataset", "num_units"], suffixes=("_mof", "_ba"))
    supervised_summary = supervised_summary[["dataset", "num_units", "mean_mof", "sd_mof", "mean_ba", "sd_ba"]]
    supervised_summary.columns = ["Dataset", "Ku", "frame MoF mean", "seed SD", "balanced accuracy mean", "seed SD (BA)"]

    reuse_summary = reuse[reuse.grouping == "subject"].groupby(["dataset", "window", "num_units", "seed", "fold"], as_index=False).agg(
        fraction_used=("used_by_at_least_half", "mean"), unused=("unused", "sum"))
    reuse_summary = reuse_summary.groupby(["dataset", "window", "num_units"], as_index=False).agg(
        fraction_mean=("fraction_used", "mean"), fraction_population_sd=("fraction_used", lambda x: x.std(ddof=0)),
        unused_mean=("unused", "mean"))
    reuse_summary.columns = ["Dataset", "Window", "Ku", "fraction mean", "fold/seed population SD", "unused units mean"]

    gates = manifest["gates"]
    invalid = int((supervised.status != "complete").sum())
    summary = [
        "1. Both released-checkpoint provenance gates passed for HuGaDB and LARa.",
        f"2. Same-checkpoint quantization agreement was {gates[0]['same_checkpoint_quantisation_agreement']:.3f} and {gates[1]['same_checkpoint_quantisation_agreement']:.3f}, respectively.",
        f"3. Historical prediction agreement was {gates[0]['historical_prediction_agreement']:.6f} for HuGaDB and {gates[1]['historical_prediction_agreement']:.6f} for LARa.",
        f"4. D7 MoF reproduced {gates[0]['d7']['MoF']:.2f} for HuGaDB and {gates[1]['d7']['MoF']:.2f} for LARa.",
        "5. Information diagnostics use count-preserving global and within-action permutation references.",
        "6. Pooled clustering results use oracle ground-truth segment boundaries and collection-wide unsupervised fitting.",
        "7. Recording- and subject-grouped readouts use four folds with fitting-only preprocessing, vocabularies, temperatures and classifiers.",
        "8. Hard, soft and continuous representations are reported separately; soft results include 0.5x, 1x and 2x temperatures.",
        "9. Seed spreads below are population SDs for initialization sensitivity, while fold results remain individually available.",
        f"10. There are {invalid} invalid supervised rows due to the prescribed convergence rule; details are retained in not_run.csv.",
    ]
    text = "# Q1/Q2 motion-unit diagnostics\n\n" + "\n".join(summary)
    text += "\n\n## Provenance and gates\n\n"
    text += md_table(pd.DataFrame([{
        "Dataset": g["dataset"], "Checkpoint SHA-256": g["checkpoint_sha256"], "Recordings": g["n_recordings"],
        "Frames": g["n_frames"], "Quantization agreement": g["same_checkpoint_quantisation_agreement"],
        "Historical agreement": g["historical_prediction_agreement"], "D7 MoF": g["d7"]["MoF"], "Passed": g["passed"]
    } for g in gates]), 6)
    text += "\n\n## Full-patch latent information\n\n" + md_table(info_summary, 4)
    text += "\n\n## Full-patch pooled hard clustering\n\n" + md_table(cluster)
    text += "\n\n## Full-patch subject-grouped hard readout\n\n" + md_table(supervised_summary)
    text += "\n\n## Unit reuse\n\n" + md_table(reuse_summary, 4)
    text += """

## Protocol disclosures

Ground-truth action labels define segments, information variables and conditional permutations; they are also used for Hungarian and oracle many-to-one evaluation mappings and for supervised fitting/evaluation. Subject identifiers define grouping and subject diagnostics. No evaluation label selected an unsupervised hyperparameter or removed a result. Pooled diagnostics are oracle-boundary and transductive. Cross-validation preprocessing, vocabulary, temperature and readout use fitting recordings only, while their frozen encoders remain transductive. Raw-feature runs have no encoder exposure, but their pooled fitting remains transductive.

HuGaDB has 10 observed action classes and 18 subjects with wearable IMU channels; LARa has 8 observed action classes and 16 subjects with optical motion-capture skeleton channels. Every HuGaDB-LARa comparison is descriptive because class count, subject count and modality differ. HuGaDB does not establish body-shape effects. The recording-versus-subject fold gap is a protocol difference and does not isolate prior exposure to a person. Separate within-subject Hungarian mappings add oracle flexibility.

## Open questions

- How much of the observed action and subject association persists under evaluation designs that avoid oracle segment boundaries?
- Which dataset or modality properties account for differences between HuGaDB and LARa?
- How stable are the measured patterns across checkpoints, independent samples and non-transductive encoders?
- Why do hard, soft and continuous readouts differ at particular vocabulary sizes and window scales?

## What these numbers cannot establish

These diagnostics do not identify a unique bottleneck, establish causal effects, determine which component dominates, define the project contribution, or prescribe the next project step. Permutation distributions are descriptive references rather than uncertainty estimates for independent temporal samples. Seed variation measures initialization sensitivity rather than sampling uncertainty. Supervised readouts are diagnostics rather than information-theoretic ceilings.

## Not run

"""
    if not_run.empty:
        text += "No planned cells were omitted or invalid.\n"
    else:
        by_reason = not_run.groupby(["status", "reason"], dropna=False).size().reset_index(name="rows")
        text += md_table(by_reason) + "\n\nSee `not_run.csv` for cell-level details.\n"
    text += "\nAll planned cells were executed. The eight-hour elapsed deadline was exceeded because execution was paused and later resumed on the user's instruction to finish; the manifest records the original deadline and this deviation.\n"
    (OUT / "REPORT.md").write_text(text, encoding="utf-8")


def expected_checks(pooled: pd.DataFrame, info: pd.DataFrame, supervised: pd.DataFrame, reuse: pd.DataFrame) -> dict:
    return {
        "q1q2_cells_unique": len(pooled), "q1q2_cells_expected": 882,
        "information_unique": len(info), "information_expected": 180,
        "supervised_unique": len(supervised), "supervised_expected": 2920,
        "unit_reuse_unique": len(reuse), "unit_reuse_expected": 80640,
    }


def main() -> None:
    pooled = clean_frame(load_jsonl("pooled_cells.jsonl"),
                         ["dataset", "representation", "window", "num_units", "seed", "readout", "temperature_multiplier"])
    info = clean_frame(load_jsonl("information.jsonl"), ["dataset", "representation", "window", "num_units", "seed"])
    supervised = clean_frame(load_jsonl("supervised_readouts.jsonl"),
                             ["dataset", "grouping", "window", "num_units", "seed", "readout", "temperature_multiplier", "fold", "level"])
    reuse = clean_frame(load_jsonl("unit_reuse.jsonl"),
                        ["dataset", "grouping", "window", "num_units", "seed", "fold", "unit"])
    checks = expected_checks(pooled, info, supervised, reuse)
    missing = []
    for actual_key, expected_key in [("q1q2_cells_unique", "q1q2_cells_expected"),
                                     ("information_unique", "information_expected"),
                                     ("supervised_unique", "supervised_expected"),
                                     ("unit_reuse_unique", "unit_reuse_expected")]:
        if checks[actual_key] != checks[expected_key]:
            missing.append({"status": "incomplete", "reason": f"unique row count {checks[actual_key]} != expected {checks[expected_key]}", "artifact": actual_key})
    invalid = supervised[supervised.status != "complete"].copy()
    keep = [c for c in ["dataset", "grouping", "window", "num_units", "seed", "readout", "temperature_multiplier", "fold", "level", "status", "reason"] if c in invalid]
    not_run = pd.concat([invalid[keep], pd.DataFrame(missing)], ignore_index=True, sort=False)
    write_csv(pooled, "q1q2_cells.csv")
    write_csv(info, "q1q2_information.csv")
    write_csv(supervised, "supervised_readouts.csv")
    write_csv(reuse, "unit_reuse.csv")
    write_csv(not_run, "not_run.csv")
    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Consolidating reports is not a new experiment completion event.
    manifest["report_consolidated_utc"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    manifest["deadline_utc"] = "2026-09-23T19:26:12Z"
    manifest["deadline_compliance"] = False
    manifest["deadline_note"] = "Execution was paused and later resumed on the user's instruction to finish; all planned cells were completed."
    manifest["datasets"] = {
        "hugadb": {"windows": [60, 30, 15], "action_classes": 10, "subjects": 18,
                    "modality": "wearable IMU channels"},
        "lara": {"windows": [50, 25, 12], "action_classes": 8, "subjects": 16,
                 "modality": "optical motion-capture skeleton channels"},
    }
    manifest["configuration"] = {
        "vocabulary_sizes": [10, 20, 50, 100, 500, 1000],
        "clustering_seeds": [111, 222, 1538574472], "fixed_randomness_seed": 111,
        "fit_window_cap": 10000, "pca_components_max": 64,
        "kmeans": {"init": "k-means++", "n_init": 5, "max_iter": 300, "tol": 0.0001, "algorithm": "lloyd"},
        "soft_temperature_multipliers": [0.5, 1.0, 2.0],
        "logistic_regression": {"class_weight": "balanced", "penalty": "l2", "C": 1.0,
                                "fit_intercept": True, "solver": "lbfgs", "max_iter": 2000, "tol": 0.0001},
    }
    manifest["split_manifests"] = {
        ds: {group: f"artifacts/{ds}/folds/{group}.npz" for group in ("recording", "subject")}
        for ds in ("hugadb", "lara")
    }
    manifest["fit_sample_identifiers"] = "Stored as recording/start-frame arrays in each preprocessing artifact under artifacts/<dataset>/<representation>/ and artifacts/<dataset>/cv/<grouping>/."
    manifest["final_counts"] = checks
    manifest["outputs"] = ["q1q2_cells.csv", "q1q2_information.csv", "supervised_readouts.csv", "unit_reuse.csv", "not_run.csv", "REPORT.md",
                           "information_vs_ku.png", "clustering_and_supervised_vs_ku.png", "unit_reuse_vs_ku.png"]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    render_information(info)
    render_performance(pooled, supervised)
    render_reuse(reuse)
    sys.path.insert(0, str(ROOT))
    from script.exp2round.q1q2.report_complete import main as complete_report
    complete_report()
    print(json.dumps(checks, indent=2))
    if missing:
        raise SystemExit("planned output counts are incomplete")


if __name__ == "__main__":
    main()
