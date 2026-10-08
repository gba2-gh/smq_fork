"""Tests for phase 5. Run: python -m script.week5.timescale_inspection.tests5"""

from __future__ import annotations

import contextlib
import io
import json
import shutil
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from script.quantized_input_comparison import fit as fitmod  # noqa: E402
from script.quantized_input_comparison import masking  # noqa: E402
from script.week5.raw_kmeans_cross_dataset import datasets as D  # noqa: E402
from script.week5.raw_kmeans_cross_dataset import prepare as PREP  # noqa: E402
from script.week5.timescale_inspection import gates5 as G  # noqa: E402
from script.week5.timescale_inspection import report5 as R5r  # noqa: E402
from script.week5.timescale_inspection import run_phase5 as R5  # noqa: E402
from script.week5.timescale_inspection import tokenizers as TOK  # noqa: E402


def _runner(run_id: str, hours: float | None = None, smoke: bool = False) -> R5.Runner:
    args = Namespace(run_id=run_id, hours=hours, radius="current", context_run="rkx_v2", direct_run="ts1_v1", smoke=smoke,
                     cpu=True, dry_run=False)
    return R5.Runner(args)


@contextlib.contextmanager
def scratch_runner(**kw):
    run_id = "_tests5_scratch"
    out = ROOT / "results" / "week5" / "timescale_inspection" / run_id
    shutil.rmtree(out, ignore_errors=True)
    try:
        yield _runner(run_id, **kw)
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_g1_generalized_path_reproduces_the_rkx_v2_baseline():
    """Gate G1 itself: T0 through the generalized preparation equals the saved rkx_v2 BABEL-2 seed-111 artifacts."""
    if not (ROOT / "results/week5/raw_kmeans_cross_dataset/rkx_v2/prep/babel2__global__seed111").exists():
        return True  # reference artifacts not on this machine
    with tempfile.TemporaryDirectory() as tmp:
        res = G.gate_g1(Path(tmp) / "g1")
        assert res["status"] == "passed", res
    return True


def test_whitening_off_equals_fit_population_and_whitening_on_whitens():
    rng = np.random.default_rng(0)
    raw = {f"r{i}": dict(flat=(rng.normal(size=(40, 30)) * np.linspace(1, 6, 30)).astype(np.float32),
                         lengths=np.full(40, 4, dtype=np.int32), num_patches=40) for i in range(4)}
    ids = [(n, p) for n in raw for p in range(40)]
    a = fitmod.fit_population("x", raw, ids, 8, 6, 3)
    b = TOK.fit_population("x", raw, ids, 8, 6, 3, whiten=False)
    assert np.array_equal(a.pca.components_, b.pca.components_)
    assert np.array_equal(a.kmeans.cluster_centers_, b.kmeans.cluster_centers_)
    w = TOK.fit_population("x", raw, ids, 8, 6, 3, whiten=True)
    feats = np.concatenate([fitmod.project_pca(w, r["flat"]) for r in raw.values()])
    assert np.allclose(feats.std(axis=0), 1.0, atol=0.1)  # whitened: unit variance per component
    assert w.kmeans.cluster_centers_.shape == (6, 8)
    return True


def _skeleton(T=40, V=6, seed=0):
    return np.random.default_rng(seed).normal(size=(T, V, 3)).cumsum(0).astype(np.float32)


def test_relational_features_shape_and_rigid_motion_invariance():
    pos = _skeleton()
    d = TOK.frame_distances(pos)
    assert d.shape == (40, 15)  # 6 joints -> 15 pairs
    theta = 0.7
    rot = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]], dtype=np.float32)
    moved = pos @ rot.T + np.array([5.0, -3.0, 2.0], dtype=np.float32)
    assert np.allclose(TOK.frame_distances(moved), d, atol=1e-3)
    out = TOK.window_stats(d, 8)
    assert out.shape == (5, 45)  # mean, sd, mean |delta| over 15 pairs
    assert np.allclose(TOK.window_stats(TOK.frame_distances(moved), 8), out, atol=1e-3)
    return True


def test_window_features_use_only_the_windows_own_frames_and_handle_a_partial_window():
    d = TOK.frame_distances(_skeleton(T=43))
    base = TOK.window_stats(d, 8)
    assert base.shape == (6, 45)  # 43 frames -> 5 full windows plus a 3-frame tail
    d2 = d.copy()
    d2[:8] += 7.0  # perturb window 0 only
    changed = TOK.window_stats(d2, 8)
    assert np.allclose(changed[1:], base[1:], atol=1e-5)  # no window depends on the previous window's frames (float32 summation order aside)
    assert not np.allclose(changed[0], base[0])
    tail = TOK.window_stats(d[40:], 8)  # the tail computed on its own equals the last row
    assert np.allclose(tail[0], base[5], atol=1e-6)
    assert np.all(np.isfinite(base))
    return True


def test_positions_only_equals_the_raw_patch_restricted_to_channels_3_to_5():
    spec = D.SPECS["lara"]
    names = D.recording_names(spec)[:1]
    raw = PREP.extract_raw(spec, names, "global")[names[0]]["flat"]
    pos = TOK.extract_positions_only(spec, names)[names[0]]["flat"]
    P = raw.shape[0]
    assert pos.shape == (P, spec.window * 3 * 22)
    assert np.array_equal(raw.reshape(P, spec.window, 6, 22)[:, :, 3:6], pos.reshape(P, spec.window, 3, 22))
    return True


def test_t3_and_t3b_differ_only_in_pca_dimension_and_t3b_differs_from_t0_only_in_channels():
    t3, t3b, t0 = (TOK.SPECS[k] for k in ("t3_lara_pos256", "t3b_lara_pos64", "t0_raw"))
    drop = {"key", "label", "pca_dim", "family", "datasets", "selectable"}

    def core(s, extra=()):
        return {k: v for k, v in vars(s).items() if k not in drop | set(extra)}

    assert core(t3) == core(t3b) and t3.pca_dim == 256 and t3b.pca_dim == 64
    assert core(t3b, ("source",)) == core(t0, ("source",)) and t3b.pca_dim == t0.pca_dim
    assert t3b.source == "raw_positions" and t0.source == "raw" and t3b.whiten is False
    return True


def test_smq_latent_extraction_shape_and_checkpoint_load():
    import torch
    dev = torch.device("cpu")
    spec = D.SPECS["hugadb"]
    names = D.recording_names(spec)[:2]
    out = TOK.extract_smq(spec, names, dev)
    for n in names:
        T = D.frame_counts(spec, [n])[n]
        P = int(np.ceil(T / spec.window))
        assert out[n]["flat"].shape == (P, spec.window * 6 * 16)  # 6 sensors x 16 latent dims per frame
        assert int(out[n]["lengths"].sum()) == T
        assert np.all(np.isfinite(out[n]["flat"]))
    model, state = TOK.load_smq("hugadb", dev)
    assert all(torch.equal(model.state_dict()[k], state[k]) for k in state)
    return True


def test_selection_reads_babel2_candidates_and_never_returns_t4():
    scores = {"t0_raw": [14.0, 15.0, 16.0], "t1_whiten": [20.0, 21.0, 22.0], "t2_relational": [18.0, 18.0, 18.0],
              "t4_smq": [99.0, 99.0, 99.0], "t3_lara_pos256": [98.0, 98.0, 98.0]}
    assert R5.select_tokenizer(scores) == "t1_whiten"
    assert R5.select_tokenizer({"t0_raw": [30.0, 30.0, 30.0], "t1_whiten": [20.0] * 3, "t2_relational": [18.0] * 3}) == "t0_raw"
    assert R5.select_tokenizer({"t0_raw": [30.0, 30.0], "t1_whiten": [20.0] * 3, "t2_relational": [18.0] * 3}) is None  # incomplete
    assert all(it.cell.tok == "t1_whiten" and it.cell.dataset in ("babel2", "babel3") for it in R5.stage_items("D", selected="t1_whiten"))
    assert [it.cell.dataset for it in R5.stage_items("D", selected="t1_whiten")][:1] == ["babel2"]  # development set first
    assert "t4_smq" not in TOK.SELECTABLE
    return True


def test_t4_job_uses_guard_14_span_16_and_the_masks_then_score_targets():
    job = R5.make_job(R5.mk_cell("babel2", "t4_smq", 111))
    assert (job.guard_radius, job.mask_span, job.config_name) == (14, 16, "long_mask")
    ref = R5.make_job(R5.mk_cell("babel2", "t1_whiten", 111))
    assert (ref.guard_radius, ref.mask_span, ref.config_name) == (0, 8, "reference")
    assert R5.make_job(R5.mk_cell("babel2", "t1_whiten", 111, span=2)).mask_span == 2
    spec = D.SPECS["babel2"]
    chunk = 128

    def scored(span: int, draws: int = 60) -> float:
        rng = np.random.default_rng(0)
        starts = np.arange(chunk, dtype=np.int32) * spec.window
        lengths = np.full(chunk, spec.window, dtype=np.int32)
        sel = elig = 0
        for _ in range(draws):
            s, e = masking.build_eligible_mask(starts, lengths, chunk, chunk * spec.window, rng, radius=14, fraction=0.30, span=span)
            sel += int(s.sum())
            elig += int(e.sum())
        return elig / max(sel, 1)

    s2, s4, s8, s16 = scored(2), scored(4), scored(8), scored(16)
    assert s16 > s8 > s4 and s16 > 0.40 and s2 < 0.05 and s4 < 0.10, (s2, s4, s8, s16)  # the plan's measured ordering
    return True


def test_probe_labelling_all_some_none():
    assert R5r.probe_label([True, True, True]) == "all rise"
    assert R5r.probe_label([False, False, False]) == "none rise"
    assert R5r.probe_label([True, False, True]).startswith("some rise (2/3)")
    assert R5r.probe_label([]) == "n/a"
    return True


def test_stage_order_is_respected_under_an_expired_budget():
    assert R5.STAGE_ORDER == ("G", "A", "B", "C1", "P1", "C2", "D", "E", "P2")
    with scratch_runner(hours=1e-9) as r:
        assert r.expired()
        counts: dict = {}
        r.run_item(R5.stage_items("A")[0], counts, "A")
        assert counts == {"deadline": 3} and not list((r.out / "units").iterdir())  # nothing started
        assert [len(R5.stage_items(s)) for s in ("A", "B", "C1", "C2")] == [18, 2, 9, 3]
        assert len(R5.stage_items("D", selected="t1_whiten")) == 12 and len(R5.stage_items("E", selected="t1_whiten")) == 12
        assert R5.stage_items("D") == [] and R5.stage_items("E") == []  # no selection, no span/window stage
    return True


def test_resume_reuses_verified_units_and_rejects_a_changed_identity():
    with scratch_runner() as r:
        cell = R5.mk_cell("babel2", "t1_whiten", 111)
        ident = r.unit_identity(cell, "centre", None)
        d = r.unit_dir(cell, "centre")
        d.mkdir(parents=True)
        (d / "info.json").write_text(json.dumps(dict(status="done", identity=ident, ladder=[])))
        assert r.unit_done(cell, "centre", ident) is not None
        assert r.unit_done(cell, "centre", "other") is None
        assert r.unit_identity(R5.mk_cell("babel2", "t1_whiten", 222), "centre", None) != ident
        assert r.unit_identity(R5.mk_cell("babel2", "t1_whiten", 111, span=4), "context", "sha") != r.unit_identity(cell, "context", "sha")
        # the manifest refuses a run directory written by a different cell set / implementation
        m = json.loads(r.manifest_path.read_text())
        m["identity"]["implementation_hash"] = "stale"
        r.manifest_path.write_text(json.dumps(m))
        try:
            _runner("_tests5_scratch")
        except SystemExit:
            return True
        raise AssertionError("a stale manifest was accepted")


def test_cell_ids_gates_and_blocking():
    base = R5.mk_cell("babel2", "t1_whiten", 111)
    assert base.cell_id == "babel2__t1_whiten__seed111" and base.is_default and base.guard == 0
    assert R5.mk_cell("babel2", "t1_whiten", 111, span=4).cell_id.endswith("__span4")
    assert R5.mk_cell("babel2", "t1_whiten", 111, window=4).prep_id.endswith("__w4")
    assert R5.mk_cell("babel2", "t1_whiten", 111, span=4).prep_id == base.prep_id  # a span variant shares the preparation
    assert R5.mk_cell("babel2", "t4_smq", 111).is_default and R5.mk_cell("babel2", "t4_smq", 111).guard == 14
    with scratch_runner() as r:
        r.gates = {"g1": dict(status="passed"), "g2_babel2": dict(status="failed"), "g3": dict(status="failed")}
        assert r.blocked(R5.mk_cell("babel2", "t4_smq", 111)) and "G2" in r.blocked(R5.mk_cell("babel2", "t4_smq", 111))
        assert r.blocked(R5.mk_cell("babel2", "t1_whiten", 111)) is None  # T1 does not depend on G2 or G3
        assert "G3" in r.blocked(R5.mk_cell("lara", "t2_relational", 111))
        assert r.blocked(R5.mk_cell("babel2", "t2_relational", 111)) is None  # BABEL's T2 does not depend on G3
        r.gates["g1"] = dict(status="failed")
        assert "G1" in r.blocked(R5.mk_cell("babel2", "t1_whiten", 111))
    for tok in ("t1_whiten", "t2_relational", "t3_lara_pos256", "t3b_lara_pos64", "t4_smq"):
        for it in (i for st in ("A", "B", "C1", "C2") for i in R5.stage_items(st)):
            if it.cell.tok == tok:
                assert it.cell.dataset in TOK.SPECS[tok].datasets, (tok, it.cell.dataset)  # no undefined combination
    return True


def test_dry_run_projection_fits_the_declared_budget_shape():
    with scratch_runner() as r:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            r.dry_run()
        text = buf.getvalue()
    assert "projected total" in text and "stage A: 18 cells" in text and "stage C1: 9 cells" in text
    return True


def test_report_writes_and_applies_the_rules_on_synthetic_units():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "phase5"
        (out / "units").mkdir(parents=True)
        (out / "probes").mkdir()
        (out / "span_diag").mkdir()
        (out / "manifest.json").write_text(json.dumps(dict(run_id="t", status="finished", identity=dict(radius="current", scale="full"))))
        (out / "selection.json").write_text(json.dumps(dict(selected="t1_whiten")))

        def unit(ds, tok, arm, seed, f1, window=None, span=None):
            window = window or D.SPECS[ds].window
            span = span or TOK.SPECS[tok].span
            d = out / "units" / f"{ds}__{tok}__{arm}__{seed}__{window}__{span}"
            d.mkdir()
            (d / "info.json").write_text(json.dumps(dict(
                status="done", dataset=ds, tokenizer=tok, seed=seed, arm=arm, window=window, span=span,
                ladder=[dict(rung_steps=4000, MoF=f1 + 10, Edit=f1 + 5, F1_10=f1 + 8, F1_25=f1 + 4, F1_50=f1)],
                segment_stats=dict(mean_pred_segments=4.0, mean_gt_segments=5.0))))

        for i, s in enumerate(R5.SEEDS):
            unit("babel2", "t1_whiten", "centre", s, 25.0 + i)  # well above the T0 baseline
            unit("babel2", "t1_whiten", "context", s, 30.0 + i)
            unit("babel2", "t1_whiten", "context", s, 40.0 + i, span=4)
        text = R5r.write_report(out).read_text(encoding="utf-8")
        assert "## H6" in text and "## H7" in text and "Span (stage D)" in text or "Span (stage D)" in text or "## Span" in text
        if (ROOT / "results/week5/raw_kmeans_cross_dataset/rkx_v2/segmentation.csv").exists():
            assert "babel2 / T1 centre" in text and "beats T0 by the bar" in text
            assert "H2′" in text
    return True


def test_norm_row_handles_both_key_styles_and_results_metrics_matches_f1():
    from script.week5.timescale_inspection import report5 as R5r
    ladder_row = dict(MoF=10.0, Edit=20.0, F1_10=30.0, F1_25=40.0, F1_50=50.0, rung_steps=4000)
    csv_row = {"MoF": "10.0", "Edit": "20.0", "F1@10": "30.0", "F1@25": "40.0", "F1@50": "50.0"}
    assert R5r._norm_row(ladder_row) == R5r._norm_row(csv_row) == dict(MoF=10.0, Edit=20.0, F1_10=30.0, F1_25=40.0, F1_50=50.0)
    return True


ALL_TESTS = [
    test_norm_row_handles_both_key_styles_and_results_metrics_matches_f1,
    test_whitening_off_equals_fit_population_and_whitening_on_whitens,
    test_relational_features_shape_and_rigid_motion_invariance,
    test_window_features_use_only_the_windows_own_frames_and_handle_a_partial_window,
    test_positions_only_equals_the_raw_patch_restricted_to_channels_3_to_5,
    test_t3_and_t3b_differ_only_in_pca_dimension_and_t3b_differs_from_t0_only_in_channels,
    test_selection_reads_babel2_candidates_and_never_returns_t4,
    test_t4_job_uses_guard_14_span_16_and_the_masks_then_score_targets,
    test_probe_labelling_all_some_none,
    test_stage_order_is_respected_under_an_expired_budget,
    test_resume_reuses_verified_units_and_rejects_a_changed_identity,
    test_cell_ids_gates_and_blocking,
    test_dry_run_projection_fits_the_declared_budget_shape,
    test_report_writes_and_applies_the_rules_on_synthetic_units,
    test_smq_latent_extraction_shape_and_checkpoint_load,
    test_g1_generalized_path_reproduces_the_rkx_v2_baseline,
]

if __name__ == "__main__":
    failed = 0
    for t in ALL_TESTS:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            import traceback
            failed += 1
            print(f"FAIL {t.__name__}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
    print(f"{len(ALL_TESTS) - failed}/{len(ALL_TESTS)} passed")
    sys.exit(1 if failed else 0)
