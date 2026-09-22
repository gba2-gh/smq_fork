"""Render results/seg/tables.md from the exp1/exp2 summary JSONs (no hand-copied numbers)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEG = ROOT / "results" / "seg"
DS = ["lara", "babel1", "babel2", "babel3"]
NM = dict(lara="LARa", babel1="BABEL-1", babel2="BABEL-2", babel3="BABEL-3")


def J(name):
    p = SEG / name
    return json.load(open(p)) if p.exists() else None


def f(e, d=3, pct=False):
    if e is None:
        return "—"
    s = 100 if pct else 1
    return f"{e['point'] * s:.{d}f} [{e['lo'] * s:.{d}f}, {e['hi'] * s:.{d}f}]"


def sig(e):
    return "**" if (e["lo"] > 0 or e["hi"] < 0) else ""


def cd(e, d=3, pct=False):
    s = sig(e)
    return f"{s}{f(e, d, pct)}{s}"


out = []
E1 = {d: J(f"exp1_summary_{d}.json") for d in DS}
E2 = {d: J(f"exp2_summary_{d}.json") for d in DS}
E1D = {d: J(f"exp1_summary_{d}_dur.json") for d in DS}

# ---------------------------------------------------------------- E1
out.append("### T1. Common-anchor retrieval — class-balanced precision@10 [95% interval over participants (LARa) / recordings (BABEL)]\n")
out.append("Interval on each level is its own resampling interval (not paired). Mean over 3 k-means seeds (random: 10 randomizations × 3 seeds). Chance = random-ranking expectation under the same group exclusion.\n")
out.append("| Dataset | readout | Fixed | Oracle | Random (oracle-matched) | Original SMQ | chance |")
out.append("|---|---|---|---|---|---|---|")
for d in DS:
    s = E1[d]
    if not s:
        continue
    c = s["conditions"]
    for mode, lab in (("cont", "continuous 8-bin"), ("q", "quantized prototype")):
        smq = f(c["smq"].get(f"ca_{mode}_p")) if mode == "q" else "—"
        out.append(f"| {NM[d]} | {lab} | {f(c['fixed'][f'ca_{mode}_p'])} | {f(c['oracle'][f'ca_{mode}_p'])} | "
                   f"{f(c['random'][f'ca_{mode}_p'])} | {smq} | {c['fixed'][f'ca_{mode}_chance']['point']:.3f} |")

out.append("\n### T2. Paired contrasts (bold = 95% interval excludes 0)\n")
out.append("Difference in class-balanced precision@10 (common anchors).\n")
out.append("| Dataset | readout | Oracle − Fixed | Oracle − Random | Random − Fixed |")
out.append("|---|---|---|---|---|")
for d in DS:
    s = E1[d]
    for mode, lab in (("cont", "continuous"), ("q", "quantized")):
        pc = s["paired_contrasts"]
        k = f"ca_{mode}_p"
        out.append(f"| {NM[d]} | {lab} | {cd(pc['oracle-fixed'][k])} | {cd(pc['oracle-random'][k])} | {cd(pc['random-fixed'][k])} |")

out.append("\n### T3. Secondary endpoints — paired contrasts\n")
out.append("NMI: frame-level cluster/label NMI, mean over folds. MoF(transfer): train-fitted majority map (fitting labels). MoF(Hungarian): fold-level benchmark mapping (evaluation labels). MoF in points.\n")
out.append("| Dataset | metric | Oracle − Fixed | Oracle − Random | Random − Fixed |")
out.append("|---|---|---|---|---|")
for d in DS:
    pc = E1[d]["paired_contrasts"]
    for k, lab, pct in (("cont_nmi", "NMI", False), ("cont_ari", "ARI", False), ("cont_mof_transfer", "MoF, train-fitted map (pts)", True),
                        ("cont_mof_hungarian", "MoF, fold Hungarian (pts)", True)):
        out.append(f"| {NM[d]} | {lab} | {cd(pc['oracle-fixed'][k], 3 if not pct else 1, pct)} | "
                   f"{cd(pc['oracle-random'][k], 3 if not pct else 1, pct)} | {cd(pc['random-fixed'][k], 3 if not pct else 1, pct)} |")

out.append("\n### T4. Segment-level retrieval (eligible segments only) and mixed-label sensitivity — paired contrasts on class-balanced precision@10\n")
out.append("seg80 = segments with action purity ≥ 0.8 (same rule for every condition); seg00 = all segments incl. mixed-label (majority label as relevance).\n")
out.append("| Dataset | set | readout | Oracle − Fixed | Oracle − Random | Random − Fixed |")
out.append("|---|---|---|---|---|---|")
for d in DS:
    pc = E1[d]["paired_contrasts"]
    for st in ("seg80", "seg00"):
        for mode in ("cont", "q"):
            k = f"{st}_{mode}_p"
            out.append(f"| {NM[d]} | {st} | {mode} | {cd(pc['oracle-fixed'][k])} | {cd(pc['oracle-random'][k])} | {cd(pc['random-fixed'][k])} |")

out.append("\n### T5. Continuous vs quantized (common anchors, precision@10 point estimates) and share of the continuous lift kept after quantization\n")
out.append("| Dataset | condition | continuous lift over chance | quantized lift over chance | quantized / continuous |")
out.append("|---|---|---|---|---|")
for d in DS:
    c = E1[d]["conditions"]
    for cond in ("fixed", "oracle", "random"):
        a, b = c[cond]["ca_cont_lift"]["point"], c[cond]["ca_q_lift"]["point"]
        out.append(f"| {NM[d]} | {cond} | {a:.3f} | {b:.3f} | {b / a:.2f} |")

out.append("\n### T6. Sources of variability at the point sample (SD)\n")
out.append("Clustering-seed SD = SD over 3 k-means seeds (random: mean within-randomization SD). Randomization SD = SD over 10 duration-permutation draws of the seed-averaged score. Compare with the paired intervals above.\n")
out.append("| Dataset | metric | Fixed seed SD | Oracle seed SD | Random seed SD | Random randomization SD |")
out.append("|---|---|---|---|---|---|")
for d in DS:
    c = E1[d]["conditions"]
    for k in ("ca_cont_p", "ca_q_p", "cont_nmi"):
        out.append(f"| {NM[d]} | {k} | {c['fixed'][k]['sd_across_kmeans_seeds']:.4f} | {c['oracle'][k]['sd_across_kmeans_seeds']:.4f} | "
                   f"{c['random'][k]['sd_across_kmeans_seeds']:.4f} | {c['random'][k]['sd_across_randomisations']:.4f} |")

out.append("\n### T7. Informative-null sensitivity: recordings with ≥ 4 oracle segments and duration CV ≥ 0.25\n")
out.append("| Dataset | recordings kept / total | readout | Oracle − Fixed | Oracle − Random | Random − Fixed |")
out.append("|---|---|---|---|---|---|")
for d in DS:
    s = E1[d]
    sub = s["informative_null_subset"]
    if "paired_contrasts" not in sub:
        out.append(f"| {NM[d]} | {sub['n_recordings']} / {s['n_recordings']} | too few | | | |")
        continue
    for k, lab in (("ca_cont_p", "continuous"), ("ca_q_p", "quantized")):
        pc = sub["paired_contrasts"]
        out.append(f"| {NM[d]} | {sub['n_recordings']} / {s['n_recordings']} | {lab} | {cd(pc['oracle-fixed'][k])} | {cd(pc['oracle-random'][k])} | {cd(pc['random-fixed'][k])} |")

out.append("\n### T8. Null-partition informativeness (oracle-matched random)\n")
out.append("| Dataset | recordings | single segment | < 4 segments | ≥ half of randomizations identical to oracle | duration CV < 0.25 |")
out.append("|---|---|---|---|---|---|")
for d in DS:
    n = E1[d]["checks"]["null_summary"]
    out.append(f"| {NM[d]} | {n['recordings']} | {n['single_segment']} | {n['fewer_than_4_segments']} | "
               f"{n['majority_of_randomisations_identical_to_oracle']} | {n['duration_cv_below_025']} |")

out.append("\n### T9. Per-class coverage at the point sample (segment-level, purity ≥ 0.8, continuous): eligible queries per class and per-class precision\n")
for d in DS:
    s = E1[d]
    cn = s["class_names"]
    out.append(f"\n**{NM[d]}**\n")
    out.append("| condition | " + " | ".join(cn) + " |")
    out.append("|---|" + "---|" * len(cn))
    for cond in ("fixed", "oracle", "random"):
        pcl = s["per_class"][cond]["seg80_cont"]
        out.append(f"| {cond}: queries | " + " | ".join(f"{n:.0f}" for n in pcl["n_queries"]) + " |")
        out.append(f"| {cond}: precision (chance) | " + " | ".join(
            (f"{p:.2f} ({c:.2f})" if p is not None else "—") for p, c in zip(pcl["precision"], pcl["chance"])) + " |")

out.append("\n### T10. Original SMQ vs controlled fixed-window representation (paired; Fixed − SMQ)\n")
out.append("| Dataset | quantized retrieval p@10 | NMI | F1@50 (Hungarian) | Edit (Hungarian) |")
out.append("|---|---|---|---|---|")
for d in DS:
    if E2[d]:
        pc = E2[d]["paired_contrasts"]["fixed-smq"]
        out.append(f"| {NM[d]} | {cd(pc['ca_q_p'])} | {cd(pc['cont_nmi'])} | {cd(pc['F1_50_h'], 2)} | {cd(pc['Edit_h'], 2)} |")

if any(E1D.values()):
    out.append("\n### T11. Sensitivity: duration-weighted k-means fitting (Experiment 1, common-anchor p@10, paired)\n")
    out.append("| Dataset | readout | Oracle − Fixed | Oracle − Random | Random − Fixed |")
    out.append("|---|---|---|---|---|")
    for d in DS:
        s = E1D[d]
        if not s:
            continue
        for mode, lab in (("cont", "continuous"), ("q", "quantized")):
            pc = s["paired_contrasts"]; k = f"ca_{mode}_p"
            out.append(f"| {NM[d]} | {lab} | {cd(pc['oracle-fixed'][k])} | {cd(pc['oracle-random'][k])} | {cd(pc['random-fixed'][k])} |")

# ---------------------------------------------------------------- E2
out.append("\n## Experiment 2\n")
out.append("### T12. Segmental metrics by condition (fold-level Hungarian mapping; mean over seeds/randomizations; [95% interval])\n")
out.append("| Dataset | condition | F1@10 | F1@25 | **F1@50** | Edit | MoF |")
out.append("|---|---|---|---|---|---|---|")
for d in DS:
    s = E2[d]
    if not s:
        continue
    for c, lab in (("smq", "Original SMQ"), ("fixed", "Fixed 1 s (controlled)"), ("dp", "DP variable (label-free)"),
                   ("dp_random", "Random, DP-matched")):
        t = s["conditions"][c]
        out.append(f"| {NM[d]} | {lab} | {f(t['F1_10_h'], 1)} | {f(t['F1_25_h'], 1)} | {f(t['F1_50_h'], 1)} | "
                   f"{f(t['Edit_h'], 1)} | {f(t['cont_mof_hungarian'], 1, True)} |")

out.append("\n### T13. Paired contrasts, Hungarian mapping (bold = interval excludes 0)\n")
out.append("| Dataset | contrast | ΔF1@10 | ΔF1@25 | **ΔF1@50** | ΔEdit | ΔMoF (pts) |")
out.append("|---|---|---|---|---|---|---|")
for d in DS:
    s = E2[d]
    if not s:
        continue
    for k, lab in (("dp-fixed", "DP − Fixed"), ("dp-dp_random", "DP − DP-random"), ("dp_random-fixed", "DP-random − Fixed")):
        pc = s["paired_contrasts"][k]
        out.append(f"| {NM[d]} | {lab} | {cd(pc['F1_10_h'], 1)} | {cd(pc['F1_25_h'], 1)} | {cd(pc['F1_50_h'], 1)} | "
                   f"{cd(pc['Edit_h'], 1)} | {cd(pc['cont_mof_hungarian'], 1, True)} |")

out.append("\n### T14. Same contrasts under the train-fitted mapping (descriptive; see caveat on label collapse)\n")
out.append("| Dataset | contrast | ΔF1@50 | ΔEdit | ΔMoF (pts) |")
out.append("|---|---|---|---|---|")
for d in DS:
    s = E2[d]
    if not s:
        continue
    for k, lab in (("dp-fixed", "DP − Fixed"), ("dp-dp_random", "DP − DP-random"), ("dp_random-fixed", "DP-random − Fixed")):
        pc = s["paired_contrasts"][k]
        out.append(f"| {NM[d]} | {lab} | {cd(pc['F1_50_t'], 1)} | {cd(pc['Edit_t'], 1)} | {cd(pc['cont_mof_transfer'], 1, True)} |")

out.append("\n### T15. Continuation rule (PROTOCOL §5)\n")
out.append("(a) ΔF1@50 ≥ 2.0 with interval excluding 0; (b) ΔEdit ≥ 0 and ΔMoF ≥ −1 pt; (c1) random explains < ½ of the gain; (c2) DP − DP-random interval excludes 0. ADVANCE = a∧b∧c1∧c2.\n")
out.append("| Dataset | mapping | a | b | c1 | c2 | ADVANCE |")
out.append("|---|---|---|---|---|---|---|")
for d in DS:
    s = E2[d]
    if not s:
        continue
    for mp, lab in (("hungarian_mapping", "Hungarian (primary)"), ("train_fitted_mapping", "train-fitted")):
        r = s["continuation_rule"][mp]
        yn = lambda b: "yes" if b else "no"
        out.append(f"| {NM[d]} | {lab} | {yn(r['a_gain_ge_2pts_and_ci_excludes_0'])} | {yn(r['b_edit_ge_0_and_mof_ge_minus1'])} | "
                   f"{yn(r['c_random_explains_less_than_half'])} | {yn(r['c_dp_beats_random_ci_excludes_0'])} | **{yn(r['ADVANCE'])}** |")

out.append("\n### T16. Common-anchor retrieval and NMI, DP vs Fixed vs Oracle (paired), and the share of the oracle gain recovered\n")
out.append("| Dataset | readout | DP − Fixed | Oracle − Fixed | ratio (DP gain / oracle gain) |")
out.append("|---|---|---|---|---|")
for d in DS:
    s = E2[d]
    if not s:
        continue
    for k, lab in (("ca_cont_lift", "continuous p@10"), ("ca_q_lift", "quantized p@10"), ("cont_nmi", "NMI")):
        r = s["oracle_advantage_recovered_by_dp"][k]
        ratio = "n/a" if r["ratio_point"] is None else f"{r['ratio_point']:.2f}"
        out.append(f"| {NM[d]} | {lab} | {cd(r['dp_minus_fixed'])} | {cd(r['oracle_minus_fixed'])} | {ratio} |")
out.append("\nA ratio is meaningful only when both gains are positive and the oracle gain's interval excludes 0.\n")

out.append("\n### T17. Class-agnostic boundary precision / recall / F1 (one-to-one matching, ±0.5 s), point estimates\n")
out.append("Boundaries = changes in the frame-level cluster sequence (\"code change\"); \"cut\" = raw segment edges.\n")
out.append("| Dataset | condition | code-change P | R | F1 | cut P | R | F1 |")
out.append("|---|---|---|---|---|---|---|---|")
for d in DS:
    s = E2[d]
    if not s:
        continue
    for c, lab in (("smq", "Original SMQ"), ("fixed", "Fixed"), ("dp", "DP"), ("dp_random", "DP-random"), ("oracle", "Oracle")):
        t = s["conditions"][c]
        g = lambda k: f"{t[k]['point']:.1f}"
        out.append(f"| {NM[d]} | {lab} | {g('bcode_P_tol0.5')} | {g('bcode_R_tol0.5')} | {g('bcode_F1_tol0.5')} | "
                   f"{g('bcut_P_tol0.5')} | {g('bcut_R_tol0.5')} | {g('bcut_F1_tol0.5')} |")
out.append("\nDP − Fixed paired difference in raw-cut boundary F1 (±0.5 s):\n")
out.append("| Dataset | Δ cut F1 | Δ code-change F1 |")
out.append("|---|---|---|")
for d in DS:
    s = E2[d]
    if s:
        pc = s["paired_contrasts"]["dp-fixed"]
        out.append(f"| {NM[d]} | {cd(pc['bcut_F1_tol0.5'], 1)} | {cd(pc['bcode_F1_tol0.5'], 1)} |")

out.append("\n### T18. DP segmentation descriptives (no success criterion)\n")
out.append("| Dataset | segments/s (grid = ~1) | DP duration median / p10–p90 (s) | GT duration median / p10–p90 (s) | at min bound (0.25 s) | at max bound (10 s) | λ per fold | rate matched on fitting set (±2%) |")
out.append("|---|---|---|---|---|---|---|---|")
for d in DS:
    s = E2[d]
    if not s:
        continue
    ds_ = s["descriptives"]
    a, b = ds_["dp_duration_s"], ds_["gt_duration_s"]
    lam = ", ".join(f"{v:.2f}" for v in ds_["lambda_per_fold"].values())
    out.append(f"| {NM[d]} | {ds_['dp_rate_per_s']:.3f} | {a['median']:.2f} / {a['p10']:.2f}–{a['p90']:.2f} | "
               f"{b['median']:.2f} / {b['p10']:.2f}–{b['p90']:.2f} | {100 * ds_['frac_segments_at_min_bound']:.1f}% | "
               f"{100 * ds_['frac_segments_at_max_bound']:.2f}% | {lam} | {'yes' if ds_['rate_matched_all_folds'] else 'no'} |")

out.append("\n### T19. Clustering-seed SD and randomization SD, F1@50 (Hungarian)\n")
out.append("| Dataset | Fixed seed SD | DP seed SD | DP-random seed SD | DP-random randomization SD |")
out.append("|---|---|---|---|---|")
for d in DS:
    s = E2[d]
    if s:
        c = s["conditions"]
        out.append(f"| {NM[d]} | {c['fixed']['F1_50_h']['sd_across_kmeans_seeds']:.2f} | {c['dp']['F1_50_h']['sd_across_kmeans_seeds']:.2f} | "
                   f"{c['dp_random']['F1_50_h']['sd_across_kmeans_seeds']:.2f} | {c['dp_random']['F1_50_h']['sd_across_randomisations']:.2f} |")

(SEG / "tables.md").write_text("\n".join(out), encoding="utf-8")
print("wrote", SEG / "tables.md", len(out), "lines")
