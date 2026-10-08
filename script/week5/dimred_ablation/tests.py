"""Tests for the dimensionality-reduction ablation. Run: python -m script.week5.dimred_ablation.tests"""

from __future__ import annotations

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
from script.week5.dimred_ablation import prepare as PR  # noqa: E402
from script.week5.dimred_ablation import reducers as RD  # noqa: E402
from script.week5.dimred_ablation import report as REP  # noqa: E402
from script.week5.dimred_ablation import run_dimred as RUN  # noqa: E402
from script.week5.timescale_inspection import tokenizers as TOK  # noqa: E402


def _raw(n_rec=6, per=60, dim=30, window=4, seed=0, offsets=0.0):
    rng = np.random.default_rng(seed)
    out = {}
    for i in range(n_rec):
        flat = rng.normal(size=(per, dim)) * np.linspace(1, 6, dim) + offsets * rng.normal(size=dim)
        lengths = np.full(per, window, dtype=np.int32)
        lengths[-1] = window - 1  # a terminal partial window, excluded from fitting
        out[f"r{i}"] = dict(flat=flat.astype(np.float32), lengths=lengths, num_patches=per)
    return out


def _key(variant: str, alpha: float | None = None, num_codes: int = 6, seed: int = 3, window: int = 4) -> PR.PrepKey:
    v = RD.BY_KEY[variant]
    return PR.PrepKey("babel2", variant, "relational", window, 5, seed, v.method, v.dim,
                      v.alpha(False) if alpha is None else alpha, v.var, v.standardize, v.recenter, v.sfa_pre, v.solver, v.n_landmarks,
                      num_codes=num_codes)


def _ids(raw, window=4):
    return fitmod.complete_window_ids(fitmod.window_lengths(raw), window)


def test_base_equals_the_anchor_tokenizer_path_with_and_without_whitening():
    raw = _raw(dim=80)
    ids = _ids(raw)
    for whiten in (False, True):
        key = _key("base", alpha=float(whiten))
        a = TOK.fit_population("x", raw, ids, 64, 6, 3, whiten=whiten)
        b = PR.fit_population("x", raw, ids, key)
        assert np.array_equal(a.pca.components_, b.reducer.sk_pca.components_)
        assert np.array_equal(a.kmeans.cluster_centers_, b.kmeans.cluster_centers_)
        for rec in raw.values():
            fa, ca = fitmod.assign_kmeans(a, rec["flat"])
            fb = PR.project(b, rec["flat"])
            assert np.array_equal(fa, fb) and np.array_equal(ca, b.kmeans.predict(fb))
    return True


def test_alpha_resolution_per_anchor():
    assert RD.BY_KEY["base"].alpha(False) == 0.0 and RD.BY_KEY["base"].alpha(True) == 1.0
    assert RD.BY_KEY["whiten_flip"].alpha(False) == 1.0 and RD.BY_KEY["whiten_flip"].alpha(True) == 0.0
    assert RD.BY_KEY["whiten_half"].alpha(True) == 0.5
    assert RD.BY_KEY["sfa64"].alpha(False) == 1.0 and RD.BY_KEY["identity"].alpha(True) == 0.0
    assert PR.make_key("babel3", "base", 111).alpha == 1.0 and PR.make_key("lara", "base", 111).alpha == 0.0
    return True


def test_half_whitening_scales_variance_to_the_square_root_of_the_eigenvalue():
    x = np.random.default_rng(1).normal(size=(2000, 20)) * np.linspace(1, 10, 20)
    red, z = RD.fit_reducer(RD.BY_KEY["whiten_half"], 0.5, x, seed=0)
    full, _ = RD.fit_reducer(RD.Variant("p", "A", "pca", 16), 0.0, x, seed=0)
    ev = full.sk_pca.explained_variance_
    assert np.allclose(z.var(axis=0, ddof=1)[:16], np.sqrt(ev), rtol=1e-6)
    return True


def test_variance_criterion_picks_the_smallest_sufficient_dimension():
    x = np.random.default_rng(2).normal(size=(3000, 40)) * np.linspace(0.2, 8, 40)
    red, z = RD.fit_reducer(RD.BY_KEY["var95"], 0.0, x, seed=0)
    d = red.info["chosen_dim"]
    assert z.shape[1] == d and red.info["cumulative_at_dim"] >= 0.95
    assert abs(RD.retained_variance(red, x) - red.info["cumulative_at_dim"]) < 1e-6
    smaller, _ = RD.fit_reducer(RD.Variant("p", "C", "pca", d - 1), 0.0, x, seed=0)
    assert RD.retained_variance(smaller, x) < 0.95
    return True


def test_identity_and_random_projection():
    x = np.random.default_rng(3).normal(size=(500, 30)).astype(np.float32)
    red, z = RD.fit_reducer(RD.BY_KEY["identity"], 0.0, x, seed=0)
    assert z is x and red.out_dim == 30 and RD.retained_variance(red, x) == 1.0
    r1, z1 = RD.fit_reducer(RD.Variant("r", "B", "random", 8), 0.0, x, seed=5)
    r2, _ = RD.fit_reducer(RD.Variant("r", "B", "random", 8), 0.0, x, seed=5)
    r3, _ = RD.fit_reducer(RD.Variant("r", "B", "random", 8), 0.0, x, seed=6)
    assert z1.shape == (500, 8) and np.array_equal(r1.components_, r2.components_)
    assert not np.array_equal(r1.components_, r3.components_)
    _, zw = RD.fit_reducer(RD.Variant("r", "B", "random", 8), 1.0, x, seed=5)
    assert np.allclose(zw.std(axis=0, ddof=1), 1.0)
    return True


def test_within_recording_pca_ignores_a_recording_offset_axis():
    """Recordings differ by a large constant offset along coordinate 0; within each recording coordinate 1 varies most.
    Plain PCA's first axis is the offset; within-recording PCA's is coordinate 1."""
    rng = np.random.default_rng(4)
    rows, means = [], []
    for r in range(20):
        off = np.zeros(6)
        off[0] = rng.normal() * 20
        x = rng.normal(size=(100, 6)) * np.array([0.5, 3, 1, 1, 1, 1]) + off
        rows.append(x)
        means.append(np.repeat(x.mean(axis=0)[None], 100, axis=0))
    x, m = np.concatenate(rows), np.concatenate(means)
    plain, _ = RD.fit_reducer(RD.Variant("p", "A", "pca", 1), 0.0, x, seed=0)
    wr, _ = RD.fit_reducer(RD.Variant("w", "B", "wrpca", 1), 0.0, x, seed=0, sample_rec_means=m)
    assert abs(plain.basis()[0, 0]) > 0.95 and abs(wr.basis()[0, 1]) > 0.95
    assert wr.info["between_recording_share"] < 0.05
    return True


def test_slow_features_find_the_slow_signal_under_fast_high_variance_noise():
    rng = np.random.default_rng(5)
    t = np.arange(4000)
    slow = np.sign(np.sin(t / 150.0))  # piecewise-constant "actions"
    fast = rng.normal(size=(4000, 5)) * 5  # high variance, no persistence
    mix = rng.normal(size=(6, 6))
    x = np.column_stack([slow, fast]) @ mix
    x = (x - x.mean(0)) / x.std(0)
    prev, nxt = x[:-1], x[1:]
    sfa, z = RD.fit_reducer(RD.Variant("s", "B", "sfa", 1, whiten="on", sfa_pre=6), 1.0, x, seed=0,
                            pairs_prev=prev, pairs_next=nxt)
    pca, zp = RD.fit_reducer(RD.Variant("p", "A", "pca", 1), 0.0, x, seed=0)
    assert abs(np.corrcoef(z[:, 0], slow)[0, 1]) > 0.95
    assert abs(np.corrcoef(zp[:, 0], slow)[0, 1]) < 0.5
    assert abs(z[:, 0].var(ddof=1) - 1.0) < 1e-6  # whitened by construction
    return True


def test_kernel_pca_unfolds_a_nonlinear_manifold_and_is_pca_like_for_a_wide_kernel():
    rng = np.random.default_rng(6)
    t = rng.uniform(0, 3 * np.pi, 3000)  # a spiral: one nonlinear degree of freedom plus small noise
    x = np.column_stack([t * np.cos(t), t * np.sin(t), rng.normal(size=3000) * 0.05])
    red, z = RD.fit_reducer(RD.Variant("k", "M", "kpca", 2, n_landmarks=600), 0.0, x, seed=0)
    assert z.shape == (3000, 2) and red.info["sigma"] > 0 and 0 < red.info["spectrum_share"] <= 1
    # kernel axes are harmonics along the curve, not monotone in t: several of them together recover t, linear PCA cannot
    from sklearn.linear_model import LinearRegression
    red8, z8 = RD.fit_reducer(RD.Variant("k", "M", "kpca", 8, n_landmarks=600), 0.0, x, seed=0)
    _, zp8 = RD.fit_reducer(RD.Variant("p", "A", "pca", 8), 0.0, x, seed=0)
    assert LinearRegression().fit(z8, t).score(z8, t) > 0.9 > 0.5 > LinearRegression().fit(zp8, t).score(zp8, t)
    assert np.allclose(red.transform(x[:50]), z[:50])  # out-of-sample rows reproduce the fitted projection
    # data that is nearly linear relative to the kernel bandwidth: the first kernel axis follows the first PCA axis
    y = np.random.default_rng(7).normal(size=(1500, 8)) * np.linspace(5, 1, 8)
    wide = RD.Variant("k", "M", "kpca", 1, n_landmarks=500)
    pca1, zp = RD.fit_reducer(RD.Variant("p", "A", "pca", 1), 0.0, y, seed=0)
    _, zw = RD.fit_reducer(wide, 0.0, y, seed=0)
    assert abs(np.corrcoef(zw[:, 0], zp[:, 0])[0, 1]) > 0.8
    _, zu = RD.fit_reducer(wide, 1.0, y, seed=0)  # alpha 1 whitens: unit variance on the fitting sample
    assert abs(zu[:, 0].var() - 1.0) < 0.3
    _, z0 = RD.fit_reducer(wide, 0.0, y, seed=0)  # unwhitened keeps PCA scaling: z0 = sqrt(variance) * whitened z
    ratio = z0[:, 0] / zu[:, 0]
    assert np.allclose(ratio, ratio[0], rtol=1e-6) and abs(z0[:, 0].var() / ratio[0] ** 2 - 1.0) < 0.3
    return True


def test_recentring_zeroes_each_recordings_complete_window_mean():
    raw = _raw(offsets=10.0)
    rc = PR.recentred(raw, 4)
    for n, rec in rc.items():
        complete = rec["lengths"] == 4
        assert np.allclose(rec["flat"][complete].mean(axis=0), 0.0, atol=1e-4)
        assert not np.allclose(raw[n]["flat"][complete].mean(axis=0), 0.0, atol=1e-2)
    return True


def test_every_variant_fits_projects_and_saves():
    raw = _raw(n_rec=8, per=120, dim=300)  # 300-d, so PCA-256 and SFA's 128-d pre-whitening are well posed
    ids = _ids(raw)
    tmp = Path(tempfile.mkdtemp())
    try:
        for v in RD.VARIANTS:
            src = PR.recentred(raw, 4) if v.recenter else raw
            pop = PR.fit_population(f"dr_{v.key}", src, ids, _key(v.key, num_codes=8))
            f = PR.project(pop, src["r0"]["flat"])
            assert f.dtype == np.float32 and f.shape[1] == pop.reducer.out_dim
            want = {"identity": 300, "var95": pop.reducer.out_dim}.get(v.key, v.dim)
            assert pop.reducer.out_dim == want, (v.key, pop.reducer.out_dim)
            assert 0.0 < pop.info["retained_variance"] <= 1.0 + 1e-9
            PR.save_population(tmp, pop)
            assert (tmp / "preprocessing" / f"dr_{v.key}_reducer.npz").exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return True


def test_no_scaling_variant_only_centres():
    raw = _raw(dim=20)
    pop = PR.fit_population("x", raw, _ids(raw), _key("nostd_pca64"))
    assert pop.scaler.scale_ is None and pop.reducer.out_dim == 20  # min(64, D)
    return True


def test_cells_jobs_and_anchors():
    cells = RUN.stage_cells("M")
    assert len(cells) == 5 * 4 * 3 and [c.dataset for c in cells[:15]] == ["babel2"] * 15
    assert {c.variant for c in cells} == {"base", "identity", "random64", "sfa64", "kpca64"}
    assert {c.variant for c in RUN.stage_cells("S")} == {"pca_full"} and RD.DEFAULT_STAGES == ("M",)
    assert PR.make_key("lara", "pca_full", 111).solver == "full" and PR.make_key("lara", "base", 111).solver == "randomized"
    hug = RUN.make_job(RUN.Cell("hugadb", "base", 111))
    assert (hug.guard_radius, hug.mask_span, hug.context, hug.stride, hug.arm) == (14, 8, 128, 64, "context_id")
    lara = RUN.make_job(RUN.Cell("lara", "pca16", 222))
    assert (lara.guard_radius, lara.mask_span, lara.family) == (0, 2, "dr_pca16")
    assert RUN.Cell("babel3", "base", 111).arm == "pca" and RUN.Cell("babel2", "base", 111).arm == "context"
    smoke = [c for st in RD.STAGES for c in RUN.stage_cells(st, (111,), smoke=True)]
    assert {c.variant for c in smoke if c.dataset == "babel2"} == set(RD.BY_KEY)
    return True


def test_dry_run_writes_nothing_and_projects_cost():
    rid = "dr_test_dryrun_tmp"
    out = ROOT / "results" / "week5" / "dimred_ablation" / rid
    assert not out.exists()
    args = Namespace(run_id=rid, hours=None, stages=None, datasets=None, gates_only=False, smoke=False, cpu=True,
                     dry_run=True)
    RUN.Runner(args).dry_run()
    assert not out.exists()
    return True


def test_report_reads_units_and_applies_the_bar():
    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / "units").mkdir()
        (tmp / "diagnostics").mkdir()
        json.dump(dict(status="finished", smoke=False), open(tmp / "manifest.json", "w"))
        f1 = {"base": (30, 31, 29), "pca16": (35, 35, 34), "pca32": (31, 28, 30)}
        for variant, vals in f1.items():
            for s, v in zip(REP.SEEDS, vals):
                d = tmp / "units" / f"babel2__{variant}__seed{s}__context"
                d.mkdir()
                row = dict(MoF=40.0, Edit=30.0, F1_10=40.0, F1_25=35.0, F1_50=float(v), rung_steps=4000)
                json.dump(dict(status="done", dataset="babel2", variant=variant, seed=s, ladder=[row],
                               segment_stats=dict(mean_pred_segments=3.0, mean_gt_segments=3.5)),
                          open(d / "info.json", "w"))
        text = REP.write_report(tmp).read_text(encoding="utf-8")
        units = REP.load_units(tmp)
        base = {s: REP.last_row(i)["F1_50"] for s, i in units[("babel2", "base")].items()}
        p16 = REP.paired({s: REP.last_row(i)["F1_50"] for s, i in units[("babel2", "pca16")].items()}, base)
        p32 = REP.paired({s: REP.last_row(i)["F1_50"] for s, i in units[("babel2", "pca32")].items()}, base)
        assert REP.verdict(p16, 3) == "**better**" and REP.verdict(p32, 3) == "no clear difference"
        assert "pca16" in text and "## babel2" in text
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return True


ALL_TESTS = [
    test_base_equals_the_anchor_tokenizer_path_with_and_without_whitening,
    test_alpha_resolution_per_anchor,
    test_half_whitening_scales_variance_to_the_square_root_of_the_eigenvalue,
    test_variance_criterion_picks_the_smallest_sufficient_dimension,
    test_identity_and_random_projection,
    test_within_recording_pca_ignores_a_recording_offset_axis,
    test_slow_features_find_the_slow_signal_under_fast_high_variance_noise,
    test_kernel_pca_unfolds_a_nonlinear_manifold_and_is_pca_like_for_a_wide_kernel,
    test_recentring_zeroes_each_recordings_complete_window_mean,
    test_every_variant_fits_projects_and_saves,
    test_no_scaling_variant_only_centres,
    test_cells_jobs_and_anchors,
    test_dry_run_writes_nothing_and_projects_cost,
    test_report_reads_units_and_applies_the_bar,
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
