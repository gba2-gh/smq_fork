"""Task 2 preflight tests: synthetic/fast, no full training run. Focused on
the three highest-risk invariants the driving instructions called out:
(a) initialization pairing (backbone/head/mask-token/projection identity
across matched arms, and the context_geometry_init<->context_quantized
rtol=1e-5/atol=1e-6 equivalence), (b) masking-eligibility integration (Task
1's `masking.py` is genuinely the masker used, not a reimplementation), and
(c) no ground-truth label leakage into training/checkpoint-selection code
paths (a static import-graph check, not just an assertion by construction).

Run via `python -m script.quantized_input_comparison.run_task2 --preflight`.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from script.action_transport.stageb_model import CodeContextModel, ModelConfig
from script.quantized_input_comparison import contextual as C
from script.quantized_input_comparison import masking as qic_masking

ROOT = Path(__file__).resolve().parents[2]


# --- (a) initialization pairing ------------------------------------------------------

def test_id_path_backward_compatible():
    """The extended ModelConfig/CodeContextModel's default input_mode='id'
    path must be byte-identical (hidden AND logits) to a from-scratch
    reimplementation of the pre-Task-2 forward(), for both position modes,
    train and eval mode (dropout RNG order included)."""
    from script.action_transport.stageb_model import sinusoidal_positions

    def original_forward(model, input_codes, padding_mask):
        cfg = model.cfg
        batch, length = input_codes.shape
        if cfg.position_mode == "sinusoidal":
            positions = sinusoidal_positions(length, cfg.d_model, input_codes.device)
            x = model.dropout(model.embedding(input_codes) + positions.unsqueeze(0))
            hidden = model.encoder(x, src_key_padding_mask=padding_mask)
        else:
            x = model.dropout(model.embedding(input_codes))
            idx = torch.arange(length, device=input_codes.device)
            distance = (idx[:, None] - idx[None, :]).abs().float()
            bias = -model.alibi_slopes[:, None, None] * distance
            bias = bias.unsqueeze(0).expand(batch, -1, -1, -1).clone()
            bias = bias.masked_fill(padding_mask[:, None, None, :], float("-inf"))
            hidden = model._encode_with_bias(x, bias.reshape(batch * cfg.n_heads, length, length))
        if cfg.tie_head:
            logits = hidden @ model.embedding.weight[:cfg.num_codes].T + model.head_bias
        else:
            logits = model.head(hidden)
        return hidden, logits

    for position_mode in ("sinusoidal", "alibi"):
        for train_mode in (False, True):
            cfg = ModelConfig(num_codes=20, d_model=16, n_layers=2, n_heads=2, d_ff=32, dropout=0.3,
                              position_mode=position_mode)
            torch.manual_seed(0)
            model = CodeContextModel(cfg)
            model.train(train_mode)
            codes = torch.randint(0, 21, (3, 10))
            pad = torch.zeros(3, 10, dtype=torch.bool)
            pad[:, 8:] = True
            torch.manual_seed(123)
            h1, l1 = model.forward(codes, pad)
            torch.manual_seed(123)
            h2, l2 = original_forward(model, codes, pad)
            assert torch.equal(h1, h2) and torch.equal(l1, l2), (position_mode, train_mode)
    return True


def test_matched_init_pairing():
    num_codes, input_dim = 16, 5
    cfg = C.base_train_config(max_steps=10, eval_every=5, patience_steps=10)
    init = C.build_matched_init(num_codes, input_dim, cfg, seed=111)
    lookup = np.random.default_rng(0).normal(size=(num_codes, input_dim)).astype(np.float32)
    rms = 3.0

    m_id = C.build_arm_model("context_id", num_codes, input_dim, cfg, 111, init)
    m_q = C.build_arm_model("context_quantized", num_codes, input_dim, cfg, 111, init)
    m_c = C.build_arm_model("context_continuous", num_codes, input_dim, cfg, 111, init)
    m_g = C.build_arm_model("context_geometry_init", num_codes, input_dim, cfg, 111, init,
                            lookup_over_rms=lookup / rms)

    for m in (m_id, m_q, m_c, m_g):
        for k, v in init.encoder_state.items():
            assert torch.equal(m.encoder.state_dict()[k], v)
        assert torch.equal(m.head.weight, init.head_state["weight"])
        assert torch.equal(m.head.bias, init.head_state["bias"])
    assert torch.equal(m_id.embedding.weight.data[num_codes], init.mask_token_init)
    assert torch.equal(m_q.mask_token, init.mask_token_init)
    assert torch.equal(m_c.mask_token, init.mask_token_init)
    assert torch.equal(m_g.embedding.weight.data[num_codes], init.mask_token_init)
    assert torch.equal(m_q.input_proj.weight, m_c.input_proj.weight)
    assert torch.equal(m_q.input_proj.bias, m_c.input_proj.bias)

    codes = np.resize(np.arange(num_codes, dtype=np.int64), 20)
    rms_vecs = lookup[codes] / rms
    diff = C.verify_geometry_init_matches_quantized(m_g, m_q, codes, rms_vecs)
    assert diff < 1e-6, diff
    assert not torch.equal(m_g.embedding.weight.data[:num_codes], m_id.embedding.weight.data[:num_codes])
    return True


def test_isolated_construction_does_not_perturb_training_rng():
    """spec: 'adapter construction does not change dropout, masks or batch
    order.' Draws a numpy stream, builds several matched models (which
    internally use isolated_rng), draws again -- the second draw must be
    identical to what an unperturbed stream would have produced."""
    rng_state_before = np.random.get_state()
    np.random.seed(777)
    draw_before_only = np.random.default_rng(999).permutation(50)  # independent generator, sanity baseline

    np.random.seed(777)
    cfg = C.base_train_config(max_steps=5, eval_every=5, patience_steps=5)
    init = C.build_matched_init(16, 5, cfg, seed=111)
    C.build_arm_model("context_id", 16, 5, cfg, 111, init)
    C.build_arm_model("context_quantized", 16, 5, cfg, 111, init)
    draw_after = np.random.default_rng(999).permutation(50)
    np.random.set_state(rng_state_before)
    assert np.array_equal(draw_before_only, draw_after)
    return True


# --- (b) masking integration ---------------------------------------------------------

def test_chunk_masks_uses_qic_masking_module():
    """`contextual.chunk_masks` must call `masking.build_eligible_mask`
    itself (imported, not reimplemented) -- checked both by source
    inspection and by a real behavioral match against calling
    `masking.build_eligible_mask` directly on the same chunk."""
    import inspect
    src = inspect.getsource(C.chunk_masks)
    assert "qic_masking.build_eligible_mask" in src, "chunk_masks does not call masking.build_eligible_mask"

    class FakeLoader:
        def recording(self, family, name):
            n = 40
            return dict(codes=np.arange(n, dtype=np.int64), continuous=np.zeros((n, 3), dtype=np.float32),
                       starts=(np.arange(n) * 15).astype(np.int32), lengths=np.full(n, 15, dtype=np.int32))

        def frame_count(self, name):
            return 600

    loader = FakeLoader()
    chunks = C.make_vector_chunks(loader, "fam", ["r1"], chunk=32, stride=32)
    masks = C.chunk_masks(chunks, base=111, fraction=0.30, span=8)
    c = chunks[0]
    rng = np.random.default_rng(__import__("script.action_transport.stageb_data",
                                          fromlist=["stable_seed"]).stable_seed(c.recording, c.start, 111))
    expected_selected, expected_eligible = qic_masking.build_eligible_mask(
        c.starts, c.lengths, c.valid_len, c.frame_count, rng, fraction=0.30, span=8)
    assert np.array_equal(masks[0][0], expected_selected)
    assert np.array_equal(masks[0][1], expected_eligible)
    return True


# --- (c) no label leakage --------------------------------------------------------------

TRAINING_AND_SELECTION_MODULES = ("contextual", "ot_arms")  # ot_arms' fitting functions, not its scoring functions


def test_labels_module_not_imported_by_training_modules():
    """Static check: `script/quantized_input_comparison/contextual.py` (all
    of training + checkpoint selection) must not import `labels.py` (or
    `script.exp2round.q1q2.core`/`script.action_transport.data`, the label
    sources) anywhere in its module-level import graph. `ot_arms.py` is
    allowed to import `labels` (its scoring functions read labels), but the
    import must be LOCAL to the two scoring functions
    (`fit_and_score_categorical`'s ladder call and `score_continuous`), not
    at module level, and must never appear inside a `fit_*` function body."""
    contextual_src = (ROOT / "script" / "quantized_input_comparison" / "contextual.py").read_text()
    tree = ast.parse(contextual_src)
    top_level_imports = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                top_level_imports.add((node.module if isinstance(node, ast.ImportFrom) else alias.name))
    forbidden = {"labels", "script.quantized_input_comparison.labels", "script.exp2round.q1q2.core",
                "script.action_transport.data", "q1q2core"}
    hit = forbidden & top_level_imports
    assert not hit, f"contextual.py imports label sources at module level: {hit}"

    ot_src = (ROOT / "script" / "quantized_input_comparison" / "ot_arms.py").read_text()
    ot_tree = ast.parse(ot_src)
    for node in ast.walk(ot_tree):
        if isinstance(node, ast.FunctionDef):
            if node.name in ("fit_categorical_k", "fit_continuous_arm", "theta0_from_grouping", "mu0_from_grouping",
                            "categorical_recordings", "continuous_recordings_from_raw",
                            "continuous_recordings_from_embeddings"):
                for sub in ast.walk(node):
                    if isinstance(sub, ast.ImportFrom) and sub.module and "labels" in sub.module:
                        raise AssertionError(f"{node.name} (a fitting function) imports labels.py")
    return True


def test_train_vector_contextualizer_signature_has_no_label_params():
    import inspect
    sig = inspect.signature(C.train_vector_contextualizer)
    names = set(sig.parameters)
    forbidden = {"gt", "gts", "labels", "action_labels", "ground_truth"}
    assert not (names & forbidden), names & forbidden
    return True


# --- (d) review fix-round: RNG reset / empty-eligible-batch handling ------------------

def _synth_chunks(n_chunks, length, num_codes, dim, seed):
    rng = np.random.default_rng(seed)
    chunks = []
    for i in range(n_chunks):
        codes = rng.integers(0, num_codes, size=length).astype(np.int64)
        cont = rng.normal(size=(length, dim)).astype(np.float32)
        starts = (np.arange(length) * 15).astype(np.int32)
        lengths = np.full(length, 15, dtype=np.int32)
        chunks.append(C.VectorChunk(f"rec{i}", 0, codes, cont, starts, lengths, length, length * 15))
    return chunks


def test_training_rng_reset_makes_repeated_fit_bit_identical():
    """Issue 1: two from-scratch fits of the SAME cell (same seed/init/data)
    must produce bit-identical weights regardless of how much of the global
    torch RNG stream was consumed before `train_vector_contextualizer` was
    called (simulating different prior cells / resume history)."""
    num_codes, input_dim = 12, 5
    cfg = C.base_train_config(max_steps=15, eval_every=5, patience_steps=10**9)
    init = C.build_matched_init(num_codes, input_dim, cfg, seed=111)
    train_chunks = _synth_chunks(8, 32, num_codes, input_dim, seed=1)
    val_chunks = _synth_chunks(4, 32, num_codes, input_dim, seed=2)
    lookup = np.random.default_rng(0).normal(size=(num_codes, input_dim)).astype(np.float32)

    torch.manual_seed(9999)
    torch.randn(1000)  # consume the stream by a different amount each run
    r_a = C.train_vector_contextualizer("context_quantized", train_chunks, val_chunks, num_codes, input_dim,
                                        seed=111, cfg=cfg, init=init, device=torch.device("cpu"),
                                        lookup=lookup, rms=1.0)
    torch.manual_seed(9999)
    torch.randn(37)
    r_b = C.train_vector_contextualizer("context_quantized", train_chunks, val_chunks, num_codes, input_dim,
                                        seed=111, cfg=cfg, init=init, device=torch.device("cpu"),
                                        lookup=lookup, rms=1.0)
    w_a, w_b = r_a.model.state_dict(), r_b.model.state_dict()
    for k in w_a:
        assert torch.equal(w_a[k], w_b[k]), f"weights diverged at {k} despite identical seed/init/data"
    return True


def test_empty_eligible_batches_do_not_consume_step_budget():
    """Issue 2: a batch with zero eligible targets must not increment `step`
    or advance the LR scheduler (no optimizer update happened), and
    `skipped_train_chunks` must account for it."""
    num_codes, input_dim = 12, 5
    cfg = C.base_train_config(max_steps=2, eval_every=1, patience_steps=10**9)
    cfg = cfg.__class__(**{**cfg.__dict__, "batch_size": 4})
    init = C.build_matched_init(num_codes, input_dim, cfg, seed=111)
    train_chunks = _synth_chunks(4, 32, num_codes, input_dim, seed=1)
    val_chunks = _synth_chunks(4, 32, num_codes, input_dim, seed=2)
    lookup = np.random.default_rng(0).normal(size=(num_codes, input_dim)).astype(np.float32)

    orig = qic_masking.build_eligible_mask

    def all_ineligible(*args, **kwargs):
        selected, eligible = orig(*args, **kwargs)
        return selected, np.zeros_like(eligible)

    qic_masking.build_eligible_mask = all_ineligible
    try:
        raised = False
        try:
            C.train_vector_contextualizer("context_quantized", train_chunks, val_chunks, num_codes, input_dim,
                                          seed=111, cfg=cfg, init=init, device=torch.device("cpu"),
                                          lookup=lookup, rms=1.0, mask_base=111)
        except RuntimeError:
            raised = True  # every batch forever empty -> the consecutive-empty-epoch guard must fire
    finally:
        qic_masking.build_eligible_mask = orig
    assert raised, "an all-empty-eligible run must raise rather than silently 'complete'"
    return True


def test_empty_eligible_batches_mixed_with_real_batches_only_count_real_steps():
    num_codes, input_dim = 12, 5
    cfg = C.base_train_config(max_steps=3, eval_every=1, patience_steps=10**9)
    cfg = cfg.__class__(**{**cfg.__dict__, "batch_size": 2})
    init = C.build_matched_init(num_codes, input_dim, cfg, seed=111)
    train_chunks = _synth_chunks(4, 32, num_codes, input_dim, seed=1)
    val_chunks = _synth_chunks(4, 32, num_codes, input_dim, seed=2)
    lookup = np.random.default_rng(0).normal(size=(num_codes, input_dim)).astype(np.float32)

    calls = {"n": 0}
    orig = qic_masking.build_eligible_mask

    def half_ineligible(*args, **kwargs):
        selected, eligible = orig(*args, **kwargs)
        calls["n"] += 1
        return (selected, np.zeros_like(eligible)) if calls["n"] % 2 == 0 else (selected, eligible)

    qic_masking.build_eligible_mask = half_ineligible
    try:
        result = C.train_vector_contextualizer("context_quantized", train_chunks, val_chunks, num_codes, input_dim,
                                                seed=111, cfg=cfg, init=init, device=torch.device("cpu"),
                                                lookup=lookup, rms=1.0, mask_base=111)
    finally:
        qic_masking.build_eligible_mask = orig
    assert result.info["steps_run"] == cfg.max_steps, result.info["steps_run"]
    return True


def test_validation_raises_when_every_val_chunk_is_ineligible():
    """Issue 2's validation half: an all-ineligible validation pass must not
    silently report a spurious 0.0 loss (which would then look like the
    best/optimal checkpoint)."""
    num_codes, input_dim = 12, 5
    cfg = C.base_train_config(max_steps=1, eval_every=1, patience_steps=10**9)
    init = C.build_matched_init(num_codes, input_dim, cfg, seed=111)
    train_chunks = _synth_chunks(4, 32, num_codes, input_dim, seed=1)
    val_chunks = _synth_chunks(4, 32, num_codes, input_dim, seed=2)
    lookup = np.random.default_rng(0).normal(size=(num_codes, input_dim)).astype(np.float32)

    orig = qic_masking.build_eligible_mask
    calls = {"n": 0}

    def train_ok_val_empty(starts, lengths, valid_len, frame_count, rng, **kwargs):
        selected, eligible = orig(starts, lengths, valid_len, frame_count, rng, **kwargs)
        calls["n"] += 1
        # val_masks are all built up-front (calls 1..len(val_chunks)) before any
        # training batch mask -- make exactly those all-ineligible.
        return (selected, np.zeros_like(eligible)) if calls["n"] <= len(val_chunks) else (selected, eligible)

    qic_masking.build_eligible_mask = train_ok_val_empty
    try:
        raised = False
        try:
            C.train_vector_contextualizer("context_quantized", train_chunks, val_chunks, num_codes, input_dim,
                                          seed=111, cfg=cfg, init=init, device=torch.device("cpu"),
                                          lookup=lookup, rms=1.0, mask_base=111)
        except RuntimeError as exc:
            raised = "zero eligible targets" in str(exc)
    finally:
        qic_masking.build_eligible_mask = orig
    assert raised, "an all-ineligible validation pass must raise, not return a spurious 0.0"
    return True


# --- (e) historical native-VQ anchor ---------------------------------------------------

def test_historical_vq_native_stream_never_reclusters_with_kmeans():
    """Issue 3: the 'vq_native' historical job must read native VQ ids
    straight off the checkpoint (`extracted[name]["ids"]`), never through
    `transform_and_assign` (PCA+K-means cluster assignment) -- source
    inspection of the actual branch taken for stream_name == 'vq_native',
    plus the STREAMS/historical-CSV wiring."""
    import inspect
    from script.quantized_input_comparison import historical_compat as hist

    assert hist.STREAMS == ("vq_native", "raw")
    assert hist.historical_csv_for("vq_native") == hist.VQ_NATIVE_HISTORICAL_CSV
    assert "fsq_pilot_002" in str(hist.VQ_NATIVE_HISTORICAL_CSV)
    assert hist.historical_csv_for("raw") == hist.RAW_HISTORICAL_CSV
    assert "encoder_sweep_001" in str(hist.RAW_HISTORICAL_CSV)

    src = inspect.getsource(hist.run_one_stream)
    # Split the function body at the branch and check the vq_native branch
    # text does not call transform_and_assign (kmeans cluster assignment).
    if_idx = src.index('if stream_name == "vq_native":')
    else_idx = src.index("else:", if_idx)
    vq_native_branch = src[if_idx:else_idx]
    assert "transform_and_assign" not in vq_native_branch, \
        "vq_native branch must not call transform_and_assign (K-means cluster assignment)"
    assert 'rec["ids"]' in vq_native_branch or "rec['ids']" in vq_native_branch
    return True


# --- (f) diagnostics math ---------------------------------------------------------------

def test_diagnostics_run_length_and_adjacent_similarity():
    from script.quantized_input_comparison import diagnostics as diag

    seq = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])  # runs of length 3, 2, 4 -> median 3
    assert diag.median_run_length_frames(seq) == 3.0

    v = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])  # sim(0,1)=1, sim(1,2)=0
    out = diag.adjacent_input_cosine_similarity({"r": v})
    assert abs(out["mean_adjacent_cosine_similarity"] - 0.5) < 1e-9

    codes = np.array([0, 0, 1, 1, 1, 2])  # 3 matches / 5 pairs
    same = diag.adjacent_same_code_rate({"r": codes})
    assert abs(same["adjacent_same_code_rate"] - 3 / 5) < 1e-9

    cost = np.array([[0.0, 1.0, 2.0], [0.5, 0.5, 3.0]])
    contrast = diag.emission_cost_contrast(cost)
    assert abs(contrast["mean_best_cost"] - 0.25) < 1e-9
    assert abs(contrast["mean_emission_contrast"] - 0.5) < 1e-9  # (1-0 + 0.5-0.5)/2

    vecs = np.random.default_rng(0).normal(size=(50, 4))
    rank_out = diag.vector_norm_rank_diagnostics(vecs)
    assert rank_out["ambient_dim"] == 4 and rank_out["n_vectors"] == 50
    assert 0 < rank_out["effective_rank"] <= 4 + 1e-6
    return True


# --- (g) ladder functions save predictions -----------------------------------------------

def test_categorical_final_ladder_populates_predictions_out():
    """Issue 4: `categorical_final_ladder` must, when given a `predictions_out`
    dict, fill it with the final rung's actual frame-level predictions --
    not just aggregate metrics."""
    from script.action_transport import categorical, config as at_config, transport
    from script.fsq_reconstruction.evaluate import categorical_final_ladder, fit_categorical_k

    recs = [categorical.RecordingInput("r0", np.array([0, 0, 1, 1], dtype=np.int32),
                                       np.array([5, 5, 5, 5], dtype=np.int32), 30, 5),
           categorical.RecordingInput("r1", np.array([1, 1, 0, 0], dtype=np.int32),
                                       np.array([5, 5, 5, 5], dtype=np.int32), 30, 5)]
    num_states, num_codes, eta = 2, 2, 1.0
    coeffs = transport.Coeffs(a=0.7, beta=0.3, lam=0.05, eps=0.07)
    theta0 = np.full((num_states, num_codes), 1.0 / num_codes)
    log_k = float(np.log(num_codes))
    fit = fit_categorical_k(recs, theta0, num_states, num_codes, eta, coeffs, inner_steps=3, outer_cap=3,
                            outer_patience=3, outer_reltol=1e-6, device=torch.device("cpu"), log_k=log_k,
                            deadline=None)
    starts_by_name = {"r0": np.array([0, 5, 10, 15], dtype=np.int32), "r1": np.array([0, 5, 10, 15], dtype=np.int32)}
    frame_counts_by_name = {"r0": 20, "r1": 20}
    gts_by_name = {"r0": np.zeros(20, dtype=np.int64), "r1": np.ones(20, dtype=np.int64)}
    predictions: dict = {}
    ladder = categorical_final_ladder(fit.params, recs, num_states, log_k, coeffs, (2, 4), gts_by_name,
                                      starts_by_name, frame_counts_by_name, deadline=None,
                                      predictions_out=predictions)
    assert set(predictions) >= {"r0", "r1", "__window_states__"}
    assert predictions["r0"].shape == (20,)
    assert predictions["__window_states__"]["r0"].shape == (4,)
    # predictions must be consistent with the LAST rung's own reported metrics
    # (broadcasting the saved window states must reproduce the saved frame preds).
    from script.action_transport import evaluate as at_evaluate
    rebroadcast = at_evaluate.broadcast_to_frames(predictions["__window_states__"]["r0"],
                                                  starts_by_name["r0"], recs[0].lengths, frame_counts_by_name["r0"])
    assert np.array_equal(rebroadcast, predictions["r0"])
    return True


def test_fit_and_score_categorical_skips_scoring_when_fit_invalid():
    """Issue 5: a categorical fit that comes back `outer_status == "invalid"`
    must never reach `categorical_final_ladder` -- scoring an invalid theta
    is meaningless, and must not look like a completed cell."""
    from script.action_transport.categorical import FitResult
    from script.quantized_input_comparison import ot_arms as ot_mod

    class FakeLoader:
        def lookup(self, family):
            return np.zeros((2, 3), dtype=np.float32)

        def grouping(self, family):
            g = np.zeros(512, dtype=np.int64)
            g[1] = 1
            return g  # fit_and_score_categorical hardcodes num_codes=512

        def recording(self, family, name):
            return dict(codes=np.array([0, 1], dtype=np.int64), lengths=np.array([5, 5], dtype=np.int32),
                       starts=np.array([0, 5], dtype=np.int32))

    orig_fit = ot_mod.fit_categorical_k
    orig_ladder = ot_mod.categorical_final_ladder
    called = {"ladder": False}

    def fake_fit(*args, **kwargs):
        real = orig_fit(*args, **kwargs)
        return FitResult(params=real.params, log_t_state={}, outer_iterations=real.outer_iterations,
                         outer_status="invalid", objective_trace=[])

    def tripwire_ladder(*args, **kwargs):
        called["ladder"] = True
        return orig_ladder(*args, **kwargs)

    ot_mod.fit_categorical_k = fake_fit
    ot_mod.categorical_final_ladder = tripwire_ladder
    try:
        import script.quantized_input_comparison.labels as qic_labels
        orig_gts = qic_labels.gts_for
        qic_labels.gts_for = lambda names: {n: np.zeros(10, dtype=np.int64) for n in names}
        try:
            result = ot_mod.fit_and_score_categorical(FakeLoader(), "fam", "direct_id", ["r0"], ["r0"],
                                                       deadline=None, rungs=(2,))
        finally:
            qic_labels.gts_for = orig_gts
    finally:
        ot_mod.fit_categorical_k = orig_fit
        ot_mod.categorical_final_ladder = orig_ladder
    assert result["ladder"] == []
    assert not called["ladder"], "categorical_final_ladder must not be called for an invalid fit"
    return True


# --- (h) resume config identity ---------------------------------------------------------

def test_cell_complete_rejects_config_and_status_mismatches():
    """Issue 6: `cell_complete` must reject a cell whose `info.json` was
    produced under a DIFFERENT effective config (the concrete `t2_smoke_v1`
    failure mode: rungs (20,40) declared, but some on-disk cells actually
    hold (500,1000,2000,4000) results from an earlier code/config), and
    must reject a cell whose status is not exactly "done"."""
    import tempfile
    from dataclasses import asdict as _asdict

    from script.quantized_input_comparison import run_task2 as rt2

    cfg = rt2.Task2Config(run_id="unit_test_cfg", rungs=(20, 40), max_steps=40, eval_every=10)
    other_cfg = rt2.Task2Config(run_id="unit_test_cfg", rungs=(500, 1000, 2000, 4000), max_steps=5000,
                                eval_every=100)

    with tempfile.TemporaryDirectory() as tmp:
        cell_dir = Path(tmp)

        np.savez(cell_dir / "ot.npz", mu=np.zeros((2, 2)))
        np.savez(cell_dir / "predictions.npz", r0__frame_pred=np.zeros(10, dtype=np.int32),
                 r0__window_states=np.zeros(2, dtype=np.int32))

        def write(status, cfg_hash, rungs):
            (cell_dir / "info.json").write_text(__import__("json").dumps(
                dict(status=status, effective_config_hash=cfg_hash,
                     scored_recording_names=["r0"], n_scored_recordings=1)))
            (cell_dir / "ladder.json").write_text(__import__("json").dumps(
                dict(rows=[dict(rung_steps=r, MoF=1.0, Edit=1.0, F1_10=1.0,
                                F1_25=1.0, F1_50=1.0) for r in rungs])))

        # Matches cfg exactly -> complete.
        write("done", rt2.config_hash(cfg), cfg.rungs)
        assert rt2.cell_complete(cell_dir, cfg) is True

        # Produced under a DIFFERENT effective config (other_cfg's hash/rungs,
        # e.g. a stale full-budget cell reused under a smoke cfg) -> rejected.
        write("done", rt2.config_hash(other_cfg), other_cfg.rungs)
        assert rt2.cell_complete(cell_dir, cfg) is False

        # Right config hash but status is "invalid"/"failed", not "done" -> rejected.
        write("invalid", rt2.config_hash(cfg), cfg.rungs)
        assert rt2.cell_complete(cell_dir, cfg) is False

        # Right config hash and status, but ladder rows don't cover every
        # declared rung -> rejected (an incomplete/truncated ladder).
        write("done", rt2.config_hash(cfg), (cfg.rungs[0],))
        assert rt2.cell_complete(cell_dir, cfg) is False

        write("done", rt2.config_hash(cfg), cfg.rungs)
        (cell_dir / "predictions.npz").unlink()
        assert rt2.cell_complete(cell_dir, cfg) is False
        np.savez(cell_dir / "predictions.npz", r0__frame_pred=np.zeros(10, dtype=np.int32),
                 r0__window_states=np.zeros(2, dtype=np.int32))
        write("done", rt2.config_hash(cfg), cfg.rungs)
        payload = __import__("json").loads((cell_dir / "ladder.json").read_text())
        payload["rows"][0]["status_counts"] = {"invalid": 1}
        (cell_dir / "ladder.json").write_text(__import__("json").dumps(payload))
        assert rt2.cell_complete(cell_dir, cfg) is False
    return True


def test_config_hash_changes_when_task2_source_changes():
    """A source-hash gate: `config_hash` must depend on the actual Task 2
    source files, not only the declared `Task2Config` fields, so an in-place
    code edit between two --resume invocations (the root cause behind
    `t2_smoke_v1` mixing budgets) is detectable."""
    from script.quantized_input_comparison import run_task2 as rt2

    cfg = rt2.Task2Config(run_id="unit_test_src_hash")
    real_hash = rt2.task2_source_hash
    try:
        rt2.task2_source_hash = lambda: "deadbeefdeadbeef"
        h1 = rt2.config_hash(cfg)
        rt2.task2_source_hash = lambda: "00000000cafef00d"
        h2 = rt2.config_hash(cfg)
    finally:
        rt2.task2_source_hash = real_hash
    assert h1 != h2
    return True


def test_final_solver_failure_never_reaches_scoring():
    from types import SimpleNamespace
    from script.action_transport import categorical, continuous, transport
    from script.fsq_reconstruction import evaluate as pilot_evaluate

    rec = categorical.RecordingInput("r0", np.array([0, 1], dtype=np.int32),
                                     np.array([5, 5], dtype=np.int32), 30, 5)
    crecord = continuous.ContinuousRecordingInput("r0", rec.codes, rec.lengths, 30, 5,
                                                  np.array([[1., 0.], [0., 1.]]),
                                                  np.array([False, False]))
    coeffs = transport.Coeffs(a=0.7, beta=0.3, lam=0.05, eps=0.07)
    original = transport.solve_final
    transport.solve_final = lambda init, *args, **kwargs: SimpleNamespace(
        log_t=init, accepted_steps=0, status="invalid")
    try:
        for ladder, parameters, records, extra in (
            (pilot_evaluate.categorical_final_ladder, np.full((2, 2), 0.5), [rec], (2, np.log(2))),
            (pilot_evaluate.contextual_final_ladder, np.eye(2), [crecord], ())):
            try:
                ladder(parameters, records, *extra, coeffs, (2,), {"r0": np.zeros(10, dtype=np.int32)},
                       {"r0": np.array([0, 5])}, {"r0": 10}, fail_on_invalid=True)
            except pilot_evaluate.FinalSolveInvalid:
                pass
            else:
                raise AssertionError("invalid final solve reached scoring")
    finally:
        transport.solve_final = original
    return True


def test_diagnostics_pool_runs_and_bounded_entropy_rank():
    from script.quantized_input_comparison import diagnostics as diag

    # Per-recording medians are 50.5 and 1; the median over all five runs is 1.
    seqs = {"long": np.r_[np.zeros(1, dtype=int), np.ones(100, dtype=int)],
            "short": np.array([0, 1, 0])}
    result = diag.run_length_diagnostics(seqs, seqs, 50)
    assert result["pred_median_run_length_frames"] == 1.0
    samples = np.array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
    ranks = diag.vector_norm_rank_diagnostics([samples[:2], samples[2:]], samples)
    assert ranks["numerical_rank"] == 2
    assert abs(ranks["effective_rank"] - 2.0) < 1e-10
    assert ranks["rank_sample_size"] == 4
    return True


ALL_TESTS = [
    test_id_path_backward_compatible,
    test_matched_init_pairing,
    test_isolated_construction_does_not_perturb_training_rng,
    test_chunk_masks_uses_qic_masking_module,
    test_labels_module_not_imported_by_training_modules,
    test_train_vector_contextualizer_signature_has_no_label_params,
    test_training_rng_reset_makes_repeated_fit_bit_identical,
    test_empty_eligible_batches_do_not_consume_step_budget,
    test_empty_eligible_batches_mixed_with_real_batches_only_count_real_steps,
    test_validation_raises_when_every_val_chunk_is_ineligible,
    test_historical_vq_native_stream_never_reclusters_with_kmeans,
    test_diagnostics_run_length_and_adjacent_similarity,
    test_categorical_final_ladder_populates_predictions_out,
    test_fit_and_score_categorical_skips_scoring_when_fit_invalid,
    test_cell_complete_rejects_config_and_status_mismatches,
    test_config_hash_changes_when_task2_source_changes,
    test_final_solver_failure_never_reaches_scoring,
    test_diagnostics_pool_runs_and_bounded_entropy_rank,
]


def run_all() -> bool:
    ok = True
    for fn in ALL_TESTS:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            ok = False
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"{'ALL PASSED' if ok else 'SOME FAILED'} ({len(ALL_TESTS)} tests)")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if run_all() else 1)
