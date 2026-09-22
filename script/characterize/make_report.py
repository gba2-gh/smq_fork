"""Render characterization results and their limitations from saved artifacts."""
import csv
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/characterization"


def read_csv(name):
    return list(csv.DictReader((OUT/name).open(encoding="utf-8")))


def fmt(value, digits=3):
    try:
        x = float(value)
        return f"{x:.{digits}f}" if np.isfinite(x) else "—"
    except (ValueError, TypeError):
        return "—"


def table(lines, headers, data):
    lines.extend(["", "| " + " | ".join(headers) + " |", "|" + "---|"*len(headers)])
    lines.extend("| " + " | ".join(str(x) for x in row) + " |" for row in data)
    lines.append("")


def main():
    summary = read_csv("primary_summary.csv")
    cover = read_csv("primary_coverage.csv")
    percode = read_csv("primary_per_code.csv")
    audit = json.loads((OUT/"audit.json").read_text())
    manifest = json.loads((OUT/"manifest.json").read_text())
    seeds = manifest["seeds"]
    stats = {(r["seed"],r["grouping"],r["subset"],r["weighting"],r["metric"]):r for r in summary}
    cov = {(r["seed"],r["grouping"],r["subset"]):r for r in cover}
    md = ["# What do SMQ's discrete codes represent?",
"",
"**Characterization completed: 21 September 2026.** Frozen one-second K=C baselines, three seeds, HuGaDB and LARa. No model training or downstream experiments were performed.",
"",
"**Verdict:** this study does **not establish reproducible local movement categories beyond static posture**. On LARa, the frequency-weighted posture-controlled movement advantage changes sign across seeds and the fixed matching rules retain fewer than 0.7% of anchor candidates. The across-action subset is smaller still. This is an inconclusive characterization of movement reuse, not evidence that the codes are meaningless or exclusively posture-based. Assignments are highly stable under the tested small noise perturbations. HuGaDB's skeletal-posture question is untestable from its IMU-only inputs; as requested, it contributes stability and signal exemplars only.",
"",
"The main limitation is identifiable: there is very little matched support for comparing different codes at nearly identical posture within the same donor recording and across people. The available observations cannot distinguish absence of movement information from the restrictions of this matched population.",
"",
"## 1. Frozen experimental unit and exposure",
"",
"The [protocol](PROTOCOL.md) was saved before grouping outcomes. The [manifest](manifest.json) records its SHA256, all six checkpoint/export hashes, exact model configurations, and participant splits. Models are the T1 epoch-30 baselines at seeds 1538574472, 111 and 222: LARa K=8/W=50, HuGaDB K=10/W=60. Released weights, subject-split weights and enlarged codebooks were not substituted after seeing results.",
"",
"The encoder has two stages with kernel-3 convolutions at dilations 1,2,4: a **29-frame receptive field, radius 14**. Patch p represents native preprocessed rows [pW,(p+1)W); code assignment additionally depends on 14 rows of context on each side. Only complete odd-indexed patches with complete context are eligible, so even the assignment-dependency intervals of successive eligible patches do not overlap. LARa's original preprocessing samples every fourth 200-Hz row; native index t corresponds to original row 4t. HuGaDB uses the pipeline's nominal 60 Hz.",
"",
"All selected SMQ models trained on all native recordings. The calibration/evaluation split here is **unseen only to descriptor scaling and posture k-means**, not to SMQ. It cannot demonstrate generalization to participants unseen during unsupervised training."]
    data=[]
    for ds in ("lara","hugadb"):
        c=json.loads((OUT/f"{ds}_calibration.json").read_text())
        m=manifest["datasets"][ds]
        data.append([ds,len(m["recordings"]),len(m["calibration_subjects"]),len(m["evaluation_subjects"]),
                     c["calibration_patches"],c["evaluation_patches"],"hip-relative xyz" if ds=="lara" else "IMU signals; no joint posture"])
    table(md,["Dataset","Recordings","Calibration people","Evaluation people","Calibration patches","Evaluation patches","Descriptor scope"],data)
    lm=manifest["datasets"]["lara"]
    md.extend([f"LARa calibration participants: {', '.join(lm['calibration_subjects'])}. Primary participant dyads: "+", ".join("/".join(x) for x in lm["primary_dyads"])+f". Participant {lm['unpaired_subjects'][0]} remains in stability/exemplars but is unpaired for primary inference. These dyads contain 16,874 eligible patches; sampling at most 12 evenly spaced anchors per recording gives **3,432 anchor candidates** per seed and analysis.",
"",
"LARa positional channels 3:6 are root-relative to joint 21, verified zero at the root throughout the data. The first three channels are preserved in model inputs; their exact physical meaning is not inferred from the variable name in preprocessing. No rotations, mirroring or input normalization changes were introduced. Native coordinate units are retained without assuming millimeters. HuGaDB contains six IMU channels at six sensor sites, not joint positions. One extreme sensor value at recording `HuGaDB_v2_various_10_08`, frame 21, affects patch 0; that patch is excluded by the prespecified complete-context/odd-patch rule, including from descriptor scaling. Model inputs/checkpoints were not repaired.",
"",
"## 2. Measurement and implementation checks",
"",
"LARa posture is the patch-mean configuration of the 21 non-root joints. Coordinate scaling is fitted to calibration patch means, with a floor of 5% of their median nonzero SD. Ordered displacement is position(t)−position(0); forward-difference velocity retains sign and time order. Both use the same coordinate scales. The two fixed-length RMS distances are normalized by calibration cross-person median distances: **0.801269** for displacement and **2.751475** for velocity. The primary composite is their equal-weighted mean. Distances and differences are dimensionless relative to these declared scales; raw standardized velocity has units of scaled coordinate per second.",
"",
"The sensitivity distance is forward-only endpoint-constrained DTW with a ±5-frame band (0.1 s), retaining joint identity, displacement direction and velocity. It permits modest local speed variation, not time reversal. It minimizes summed local squared costs and reports the square root of their mean on that path; it is not an exact minimum-average-path objective.",
"",
"Controlled examples were checked before grouping: all have the same mean posture. Identical motion scores zero. Opposite direction gives composite **1.033** versus **0.068** for modest speed variation; corresponding DTW distances are **1.072** and **0.081**. Each displacement and velocity component has the same ordering. These checks establish the required basic discrimination, not validation against human judgments. [Metric-check values](metric_checks.json).",
"",
"Stability inference encodes the complete raw context and quantizes only its central represented patch, avoiding a shifted quantizer-grid origin. **9,927** clean crop/checkpoint assignments match the original full-recording exports exactly. Common raw frames under temporal shift have exactly equal coordinates/signals and recomputed aligned descriptors. This directly validates the practical receptive-field/crop alignment used here.",
"",
"## 3. Primary results: movement after posture and participant controls",
"",
"For an anchor, the same-code donor must come from the other participant of its fixed dyad and lie within posture RMS 0.50. The different-code control must come from the **same donor recording**, lie within 0.25 of that donor and 0.50 of the anchor, and match the anchor–donor posture distance within 0.05. No patch is reused within an analysis. If the nearest same-code donor has no valid control, the anchor is unmatched; calipers were not relaxed. The action-restricted analysis requires both donor and control majority labels to differ from the anchor's.",
"",
"**Effect = control movement distance − same-code movement distance. Positive favors same-code movement similarity.** The following is the primary table for both datasets. Numbers are frequency-weighted within seed. Intervals resample entire participant dyads, retaining dependence among all their recordings and both pair endpoints; they are separate from model-seed variability."])
    primary=[]
    for grouping in ("smq","posture_kmeans"):
        for subset in ("all_actions","different_actions"):
            for seed in seeds:
                key=(str(seed),grouping,subset)
                r=stats[(*key,"frequency","composite")]; c=cov[key]
                primary.append(["LARa","SMQ" if grouping=="smq" else "Posture k-means",seed,
                    "all" if subset=="all_actions" else "different actions",
                    f"{r['n']} ({100*float(c['matched_fraction']):.2f}%)",c["participant_dyads"],
                    fmt(r["same"]),fmt(r["control"]),f"{fmt(r['delta'])} [{fmt(r['ci_low'])}, {fmt(r['ci_high'])}]"])
    primary.append(["HuGaDB","SMQ","all 3","—","unavailable","—","—","—","No skeletal posture/trajectory observations"])
    table(md,["Dataset","Grouping","Seed","Action restriction","Matched / 3,432 anchors","Dyads with matches","Same-code D","Control D","Difference [95% cluster interval]"],primary)
    md.extend(["Only **three of five dyads** contribute to the all-action SMQ estimates, and two or three to the across-action estimates. With so few informative clusters, percentile intervals are descriptive and fragile. For example, seed 111's positive across-action interval comes from only 12 triples in two dyads; it is not robust evidence of reusable movement categories. Empty bootstrap draws are excluded and their count is recorded. Seeds reuse the evaluation collection, so their observations must not be pooled as independent replications.",
"",
"Static-posture k-means is fitted only on calibration posture, separately with the same three seeds and K=8. It has only 7–10 matched triples without action restrictions and 3–9 across actions. Its residual effects also vary, particularly in the across-action subset. Different groupings admit different matched populations, so directly subtracting the two methods' aggregate effects would not be a controlled test of learned movement information.",
"",
"### Component distances and temporal-alignment sensitivity",
"",
"Each cell below is same-code / control / paired difference; displacement and velocity are standardized-coordinate RMS distances before their calibration median normalization. The last column gives the normalized DTW difference and its cluster interval. Both components and every control result are available in [primary_summary.csv](primary_summary.csv)."])
    component=[]
    for subset in ("all_actions","different_actions"):
        for seed in seeds:
            key=(str(seed),"smq",subset,"frequency")
            row=[seed,"all" if subset=="all_actions" else "different actions"]
            for metric in ("displacement","velocity"):
                r=stats[(*key,metric)]
                row.append(" / ".join(fmt(r[k]) for k in ("same","control","delta")))
            r=stats[(*key,"dtw")]
            row.append(f"{fmt(r['delta'])} [{fmt(r['ci_low'])}, {fmt(r['ci_high'])}]")
            component.append(row)
    table(md,["Seed","Subset","Displacement: same / control / Δ","Velocity: same / control / Δ","DTW Δ [95% interval]"],component)
    md.extend(["### Frequency weighting, code balance and seed variability",
"",
"Frequency weighting gives each accepted anchor triple equal weight; code balance gives each represented code equal weight. This is frequency in the **matched sample**, not inverse-probability extrapolation to all native patches. Missing codes cannot receive an estimable movement effect."])
    spread=[]
    for grouping in ("smq","posture_kmeans"):
        for subset in ("all_actions","different_actions"):
            for weighting in ("frequency","code_balanced"):
                vals=[float(stats[(str(s),grouping,subset,weighting,"composite")]["delta"]) for s in seeds]
                spread.append([grouping,subset,weighting," / ".join(fmt(x) for x in vals),f"{np.mean(vals):.3f} ± {np.std(vals,ddof=1):.3f}"])
    table(md,["Grouping","Subset","Weighting","Effects: default / 111 / 222","Across-seed mean ± SD"],spread)
    md.extend(["These SDs describe three fixed model/control fits. They are not uncertainty across evaluation people, which is reported in the cluster intervals. The code-balanced across-action point estimate is positive in all three seeds; that is a weak favorable signal worth recording. It covers only 3–5 represented codes and 2–3 informative dyads per seed, while frequency-weighted estimates change sign. It therefore does not establish a robust population-level movement-reuse effect.",
"",
"### Coverage and matching quality",
"",
"Across SMQ seeds, about 1,995–2,004 anchors fail the same-code posture caliper in the unrestricted analysis; another 1,403–1,412 have no different-code control meeting the fixed rules. These failure categories account for nearly all unmatched anchors. Native orientation/body configuration, fixed participant pairing and the same-recording requirement can all limit support. Low support does not show that the codes are wrong or posture-only."])
    quality=[]
    for r in cover:
        if r["grouping"]!="smq":continue
        quality.append([r["seed"],r["subset"],r["matched"],r["recordings"],r["codes_represented"],
            fmt(r["posture_same_mean"]),fmt(r["posture_control_mean"]),fmt(r["posture_balance_mean"]),fmt(r["action_purity_mean"])])
    table(md,["Seed","Subset","Triples","Recordings","Codes / 8","Mean posture same","Mean posture control","Control−same posture","Mean minimum action purity"],quality)
    md.extend(["Posture distance imbalance is small by the frozen absolute caliper but not identically zero; control posture is slightly farther on average in most cells. There is no causal adjustment for remaining within-caliper differences. Each reported triple includes its three raw intervals, subject/recording IDs, action labels/purity and all distances in [primary_pairs.csv](primary_pairs.csv). [Coverage and exclusion counts](primary_coverage.csv).",
"",
"The across-action subset uses majority labels, including potentially `None` and `Synchronization`, so not every different-label triple is a clear transition across substantive action contexts. The following purely descriptive counts expose this limitation without changing the frozen analysis:"])
    purity=[]
    for r in audit["pairing_coverage"]:
        if r["grouping"]=="smq" and r["subset"]=="different_actions":
            purity.append([r["seed"],r["n"],r["triples_excluding_none_sync"],r["triples_excluding_none_sync_purity80"]])
    table(md,["Seed","Across-label triples","No None/Synchronization in any member","Also ≥80% label purity in all members"],purity)
    md.extend(["## 4. Assignment stability, conditional on actual descriptor change",
"",
"At most six fixed eligible patches per evaluation recording were tested: **1,896 LARa patches / 316 recordings / 11 people**, and **1,413 HuGaDB patches / 243 recordings / 12 people**. Three HuGaDB patches lack complete +2-frame shifted context, leaving 1,410 for that condition. Pose/signal noise uses the same deterministic draw across model seeds. No Hungarian remapping of code IDs is used.",
"",
"LARa Gaussian xyz noise is 0.1% or 0.5% of calibration RMS root-relative position, **316.285 native coordinate units**: input SDs about 0.316 and 1.581. Root subtraction preserves zero hip and leaves the first three input channels unchanged. It increases effective non-root noise; measured full-skeleton RMS relative to the scale is about 0.138%/0.690%. HuGaDB uses each native channel/sensor's calibration SD; it is signal noise, not pose noise. Gaussian additions are floating point even though HuGaDB files store integers. Measured amplitudes are recorded in the validation JSONs.",
"",
"Agreement below is across-seed mean ± sample SD in percentage points. Descriptor changes are the mean over paired evaluation windows; they are seed-independent because inputs and perturbations are shared. HuGaDB quantities are signal-mean / signal-displacement / signal-derivative analogues, not skeletal measurements."])
    stable=[]
    for ds in ("lara","hugadb"):
        ss=read_csv(f"stability_{ds}_summary.csv")
        for perturb in ("shift_2","noise_0.001","noise_0.005"):
            rr=[r for r in ss if r["perturbation"]==perturb]
            aa=[100*float(r["agreement"]) for r in rr]
            stable.append([ds,perturb,rr[0]["n"],f"{np.mean(aa):.3f} ± {np.std(aa,ddof=1):.3f}",
                *[fmt(rr[0][k]) for k in ("posture_change","displacement_change","velocity_change","composite_change")]])
    table(md,["Dataset","Perturbation","Patches/seed","Agreement %, mean ± seed SD","Posture/signal mean Δ","Displacement Δ","Velocity/derivative Δ","Composite descriptor Δ"],stable)
    md.extend(["A +2-frame shift is 40 ms on LARa and nominally 33.3 ms on HuGaDB. Full relative-time windows have changed content and phase, whereas their explicitly aligned shared raw frames agree exactly. HuGaDB's composite change is about **1.079 calibration-distance units**, so its approximately 8% assignment changes do not establish instability under negligible motion change. LARa's 0.5% pose noise causes a large velocity-descriptor change despite essentially unchanged code IDs; this shows resistance to that high-frequency perturbation, not proof of semantic invariance.",
"",
"The tables below retain per-seed participant-cluster 95% intervals for shift agreement and show code-balanced shift agreement. Full noise intervals, descriptor intervals and per-code frequencies/agreements are exported. A bootstrap interval [100%,100%] when no flips were observed is an empirical resampling result, not proof of zero population error."])
    intervals=[]
    for ds in ("lara","hugadb"):
        ss=read_csv(f"stability_{ds}_summary.csv")
        balanced=read_csv(f"stability_{ds}_code_balanced.csv")
        for r in ss:
            if r["perturbation"]!="shift_2":continue
            b=next(x for x in balanced if x["seed"]==r["seed"] and x["perturbation"]=="shift_2")
            intervals.append([ds,r["seed"],f"{100*float(r['agreement']):.2f} [{100*float(r['agreement_ci_low']):.2f}, {100*float(r['agreement_ci_high']):.2f}]",
                              f"{100*float(b['agreement']):.2f} [{100*float(b['agreement_ci_low']):.2f}, {100*float(b['agreement_ci_high']):.2f}]"])
    table(md,["Dataset","Seed","Frequency-weighted shift agreement % [95% interval]","Code-balanced agreement % [95% interval]"],intervals)
    md.extend(["## 5. Movement and signal exemplars without outcome selection",
"",
"[Open the montage index](../../figures/characterization/exemplars/index.html). It contains **54 montages**, one for every code of every checkpoint, with **324 full-patch occurrences**. Every montage includes six distinct evaluation participants; none of the 54 codes is absent. Available evaluation-patch counts range from 97 to 6,821 across codebooks. SHA256 ordering first maximizes participant diversity, then recording diversity. [Selection, counts and exact frame provenance](exemplars.csv).",
"",
"LARa plots all 21 non-root joint tracks through the complete second, with purple-to-yellow time, start/end markers, and signed displacement heatmaps. They preserve the native coordinate system and do not infer missing skeleton connectivity. HuGaDB plots all 36 complete ordered sensor traces, with calibration-only standardization explicitly restricted to display. Near-flat traces below mean near-flat at that shared display scale, not literally zero motion.",
"",
"The fixed qualitative subset is code 0, the most frequent code, and the least frequent nonempty code in the default-seed model (deduplicated). It was chosen by the protocol, not by visual coherence:"])
    table(md,["Dataset/code","Eligible patches","Observed common structure and variation"],[
        ["LARa 0",2120,"Mix of small changes and substantial articulated movement; no visually uniform direction."],
        ["LARa 6, most frequent",5824,"Nearly stationary configurations and larger trajectories coexist; no single obvious transition."],
        ["LARa 3, least frequent",617,"Some nearly stationary examples, others strong arcs; opposite signed displacement components appear across people."],
        ["HuGaDB 3, most frequent",1755,"All six selected occurrences have nearly flat sensor traces at the shared scale."],
        ["HuGaDB 0, least frequent",340,"Varied transients and one nearly flat example; no uniform waveform apparent."]])
    md.extend(["These descriptions are qualitative evidence of heterogeneous occurrences, not semantic labels or a complete assessment of within-code consistency. They neither validate a recurring primitive nor prove its absence. A code number in one seed has no established correspondence to the same number in another seed.",
"",
"## 6. Relative support for the three explanations",
"",
"The explanations are not mutually exclusive:"])
    table(md,["Explanation","Evidence in this run","Supported conclusion"],[
        ["Movement-sensitive representation","SMQ residual movement effects positive for two seeds and negative for one; very low matching support; DTW does not supply uniform positive evidence.","Reproducible cross-person movement categories beyond posture are not established."],
        ["Posture-sensitive representation","Tight same-recording different-code posture controls are rare; static-posture control also has low coverage; examples mix static configurations and dynamics.","Posture remains a plausible contributor, but this run cannot show that posture is the dominant or sole organizing variable."],
        ["Nuisance-sensitive or unstable partition","Code agreement is near100% under declared small noise; temporal shifts change HuGaDB signal descriptors substantially; all subjects were seen by SMQ.","Strong small-noise assignment instability is not supported. Participant/recording dependence and genuinely unseen-person reproducibility remain unresolved."]])
    md.extend(["The primary contrast controls participant and recording identity between its two alternatives by design. It does not estimate all recording-specific variation, and no direct within-person versus across-person matched effect was prespecified. The limited match yield is not itself a nuisance-dependence effect. Root-relative xyz controls also leave any information in the first three native LARa channels unmeasured.",
"",
"**Supported:** the frozen models can be examined with exactly aligned raw patches; the chosen movement distances distinguish direction from modest speed differences on controlled examples; tested small-noise code agreement is high; eligible code occurrences are distributed across people; the strict matched comparison has very limited support and no consistently signed movement benefit across seeds.",
"",
"**Unsupported:** that SMQ learns reusable local movement primitives, that it merely quantizes static posture, that poor action recognition makes its codes meaningless, that sharing codes across labels proves reuse, or that these results demonstrate variable-duration syllables, hierarchy, or unseen-subject generalization.",
"",
"## 7. Audit, deviations and deliverables",
"",
"The only substantive scope adjustment was the user-approved exclusion of HuGaDB's skeletal-posture primary test. Protocol calipers, model selection and sampling were not changed after outcomes. Technical implementation corrections materialized cached code arrays once for efficient reading and applied HuGaDB Gaussian noise in floating point; final stability outputs use the corrected implementation. Raw-to-float32 cache rounding on LARa was validated after the declared cast and recorded, not treated as a data mismatch.",
"",
f"The [artifact audit](audit.json) validates all **{audit['primary_analyses_validated']}** matched analyses, **{audit['primary_triples_validated']}** triples across analyses (not mutually independent across seeds/subsets), all **{audit['primary_summary_rows_validated']}** primary summary rows, frozen checkpoint/protocol hashes, matching calipers, no within-analysis patch reuse, cross-participant and donor-recording constraints, action restrictions, 54 montage files and their participant coverage. It also records descriptive action-label quality counts. Both primary and stability intervals resample clusters rather than individual pairs. Only two or three primary dyads contain matches, so uncertainty estimates should not be used as strong hypothesis tests.",
"",
"Reproduce from the repository root with the existing `smq` Python environment:",
"",
"```powershell",
"$smqPython = 'C:/Users/gzaz976/AppData/Local/anaconda3/envs/smq/python.exe'",
"& $smqPython -B script/characterize/prepare.py",
"& $smqPython -B script/characterize/primary.py",
"& $smqPython -B script/characterize/stability.py --dataset lara",
"& $smqPython -B script/characterize/stability.py --dataset hugadb",
"& $smqPython -B script/characterize/exemplars.py",
"& $smqPython -B script/characterize/audit.py",
"& $smqPython -B script/characterize/make_report.py",
"```",
"",
"[Protocol](PROTOCOL.md) · [Frozen manifest](manifest.json) · [Primary distances and intervals](primary_summary.csv) · [Coverage/exclusions](primary_coverage.csv) · [All per-code primary results](primary_per_code.csv) · [Matched triples](primary_pairs.csv) · [LARa stability](stability_lara_summary.csv) · [HuGaDB stability](stability_hugadb_summary.csv) · [LARa per-code stability](stability_lara_per_code.csv) · [HuGaDB per-code stability](stability_hugadb_per_code.csv) · [Montages](../../figures/characterization/exemplars/index.html).",
"",
"This run stops at characterization. Resolving the unanswered question would require a separately designed comparison with adequate matched support and, for HuGaDB, actual pose observations; neither is supplied by declaring the present sparse effects decisive.",
"",
"## Appendix: every LARa SMQ code, unrestricted-action primary comparison",
"",
"Counts expose codes with no matched evidence. Intervals below are exploratory, especially with fewer than three informative dyads. The separate CSV includes the action-restricted subset, posture control, and both distance components for every code."])
    pc=[]
    for r in percode:
        if r["grouping"]=="smq" and r["subset"]=="all_actions" and r["metric"]=="composite":
            pc.append([r["seed"],r["code"],r["evaluation_occurrences"],r["anchor_candidates"],r["matched"],r["participant_dyads"],
                f"{fmt(r['delta'])} [{fmt(r['ci_low'])}, {fmt(r['ci_high'])}]"])
    table(md,["Seed","Code","Evaluation occurrences","Candidate anchors","Matched triples","Informative dyads","Movement Δ [95% interval]"],pc)
    (OUT/"REPORT.md").write_text("\n".join(md)+"\n",encoding="utf-8")
    # Compact standalone primary table; HuGaDB's missing estimand is explicit.
    pt=["# Primary posture-controlled movement results", "", "Positive Δ means same-code movements are more similar. See REPORT.md for sparse-coverage and cluster-uncertainty limitations."]
    table(pt,["Dataset","Grouping","Seed","Subset","Matched coverage","Dyads","Same","Control","Δ [95% interval]"],primary)
    (OUT/"PRIMARY_RESULTS.md").write_text("\n".join(pt)+"\n",encoding="utf-8")
    print("Wrote REPORT.md and PRIMARY_RESULTS.md")


if __name__=="__main__":main()
