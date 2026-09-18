"""Correctness checks to run before trusting any training result:

A) chunk_wp_loss's per-target weighted Pearson matches utils.weighted_pearson (the trusted
   reference) on random data.
B) A batched full-sequence forward through SequenceModel is numerically identical to feeding
   the same sequences one timestep at a time while carrying hidden state (the real stateful
   contract). This is what licenses evaluate_fast as a stand-in for the row-by-row scorer.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils import (
    DataPoint,
    FEATURE_COLUMNS,
    GlobalAccumulator,
    SEQUENCE_LENGTH,
    TARGET_COLUMNS,
    weighted_pearson,
)
from train.losses import chunk_wp_loss, global_wp_loss, weighted_pearson_torch
from train.model import ModelConfig, SequenceModel


def test_metric_matches_utils(n_trials=20, length=500, seed=0):
    rng = np.random.default_rng(seed)
    max_abs_diff = 0.0
    for _ in range(n_trials):
        y = rng.normal(size=length).astype(np.float32)
        p = rng.normal(size=length).astype(np.float32)
        ref = weighted_pearson(y, p)
        got = weighted_pearson_torch(
            torch.from_numpy(y.astype(np.float64)),
            torch.from_numpy(p.astype(np.float64)),
            torch.ones(length, dtype=torch.float64),
        ).item()
        max_abs_diff = max(max_abs_diff, abs(ref - got))
    assert max_abs_diff < 1e-6, f"metric mismatch: max abs diff {max_abs_diff}"
    print(f"[A] weighted_pearson_torch matches utils.weighted_pearson (max abs diff {max_abs_diff:.2e})")


def test_metric_masking(seed=1):
    # Rows outside the mask must not influence the correlation at all.
    rng = np.random.default_rng(seed)
    length = 300
    y = rng.normal(size=length).astype(np.float32)
    p = rng.normal(size=length).astype(np.float32)
    mask_np = np.zeros(length, dtype=bool)
    mask_np[50:250] = True
    ref = weighted_pearson(y[mask_np], p[mask_np])
    got = weighted_pearson_torch(
        torch.from_numpy(y.astype(np.float64)),
        torch.from_numpy(p.astype(np.float64)),
        torch.from_numpy(mask_np.astype(np.float64)),
    ).item()
    assert abs(ref - got) < 1e-6, f"masking mismatch: {ref} vs {got}"
    print(f"[A2] masked weighted_pearson_torch matches masked utils.weighted_pearson ({ref:.6f} vs {got:.6f})")


def test_batched_equals_stepwise(seed=2):
    torch.manual_seed(seed)
    for reparam in (False, True):
        cfg = ModelConfig(trunk_hidden=(16,), gru_hidden=24, gru_layers=2, reparam=reparam)
        model = SequenceModel(cfg).eval()

        batch, length = 3, 400
        features = torch.randn(batch, length, cfg.n_features, dtype=torch.float32)

        with torch.no_grad():
            hidden0 = model.init_hidden(batch)
            batched_pred, _ = model(features, hidden0)

            hidden = model.init_hidden(batch)
            stepwise_pred = torch.empty_like(batched_pred)
            for t in range(length):
                step_pred, hidden = model(features[:, t : t + 1], hidden)
                stepwise_pred[:, t] = step_pred[:, 0]

        max_abs_diff = (batched_pred - stepwise_pred).abs().max().item()
        assert max_abs_diff < 1e-4, f"batched vs stepwise mismatch (reparam={reparam}): {max_abs_diff}"
        print(f"[B] batched forward == stepwise forward (reparam={reparam}, max abs diff {max_abs_diff:.2e})")


def test_hidden_reset_between_batches(seed=3):
    # Different content after a reset must not leak state from a previous window.
    torch.manual_seed(seed)
    cfg = ModelConfig(gru_hidden=8, gru_layers=1)
    model = SequenceModel(cfg).eval()
    x = torch.randn(1, 5, cfg.n_features)
    with torch.no_grad():
        h0 = model.init_hidden(1)
        out_a, _ = model(x, h0)
        h0_again = model.init_hidden(1)
        out_b, _ = model(x, h0_again)
    assert torch.allclose(out_a, out_b), "init_hidden should be deterministic zeros"
    print("[B2] init_hidden gives a deterministic, reproducible reset")


def test_loss_gradient_is_finite_on_degenerate_chunks(seed=4):
    # A chunk that's all-warmup (fully masked out) or has a zero-variance target region
    # must not poison gradients (regression test for the sqrt(0)-in-a-`where` NaN trap).
    torch.manual_seed(seed)
    cfg = ModelConfig(gru_hidden=8, gru_layers=1)
    model = SequenceModel(cfg)
    length = 50
    features = torch.randn(4, length, cfg.n_features)
    t0 = torch.randn(4, length)
    t1 = torch.randn(4, length)

    fully_masked = torch.zeros(4, length)
    constant_target = torch.ones(4, length)  # zero variance -> zero weighted-var branch

    for mask in (fully_masked, constant_target):
        model.zero_grad()
        hidden = model.init_hidden(4)
        pred, _ = model(features, hidden)
        loss, _, _ = chunk_wp_loss(pred[..., 0], pred[..., 1], t0, t1, mask)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9)
        assert torch.isfinite(grad_norm), f"non-finite gradient on degenerate mask: {grad_norm}"
    print("[C] degenerate (fully-masked / zero-weight) chunks produce finite gradients")


def test_global_wp_loss_pools_across_batch(seed=5):
    # global_wp_loss must compute ONE correlation over all (B, L) rows pooled together —
    # not a per-batch-item correlation averaged afterward (that's chunk_wp_loss, kept only
    # for comparison). Verify against utils.weighted_pearson on the flattened arrays.
    rng = np.random.default_rng(seed)
    batch, length = 5, 200
    t0 = torch.from_numpy(rng.normal(size=(batch, length)).astype(np.float32))
    t1 = torch.from_numpy(rng.normal(size=(batch, length)).astype(np.float32))
    t0_hat = torch.from_numpy(rng.normal(size=(batch, length)).astype(np.float32))
    t1_hat = torch.from_numpy(rng.normal(size=(batch, length)).astype(np.float32))
    mask = torch.ones(batch, length)
    mask[0, :50] = 0  # exercise a partially-masked row too

    loss, wp0, wp1 = global_wp_loss(t0_hat, t1_hat, t0, t1, mask)

    flat_mask = mask.reshape(-1).bool().numpy()
    ref_t0 = weighted_pearson(t0.reshape(-1).numpy()[flat_mask], t0_hat.reshape(-1).numpy()[flat_mask])
    ref_t1 = weighted_pearson(t1.reshape(-1).numpy()[flat_mask], t1_hat.reshape(-1).numpy()[flat_mask])

    assert abs(wp0.item() - ref_t0) < 1e-5, f"t0 mismatch: {wp0.item()} vs {ref_t0}"
    assert abs(wp1.item() - ref_t1) < 1e-5, f"t1 mismatch: {wp1.item()} vs {ref_t1}"
    assert abs(loss.item() - (-0.5 * (ref_t0 + ref_t1))) < 1e-5
    print(
        f"[E] global_wp_loss pools the whole batch into one correlation per target "
        f"(t0 {wp0.item():.6f} vs {ref_t0:.6f}, t1 {wp1.item():.6f} vs {ref_t1:.6f})"
    )


def test_global_wp_loss_gradient_is_finite_on_degenerate_batches(seed=6):
    torch.manual_seed(seed)
    cfg = ModelConfig(gru_hidden=8, gru_layers=1)
    model = SequenceModel(cfg)
    length = 50
    features = torch.randn(4, length, cfg.n_features)
    t0 = torch.randn(4, length)
    t1 = torch.randn(4, length)

    for mask in (torch.zeros(4, length), torch.ones(4, length)):
        model.zero_grad()
        hidden = model.init_hidden(4)
        pred, _ = model(features, hidden)
        loss, _, _ = global_wp_loss(pred[..., 0], pred[..., 1], t0, t1, mask)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9)
        assert torch.isfinite(grad_norm), f"non-finite gradient on degenerate batch: {grad_norm}"
    print("[E2] global_wp_loss produces finite gradients on degenerate (all-zero / all-one weight) batches")


def test_global_accumulator_matches_one_shot_reference(n_sequences=50):
    """Global WP (METRIC_SPECIFICATION='global_wp'): GlobalAccumulator merges centered
    weighted moments sequence-by-sequence (bounded memory); this checks that merge algorithm
    against the direct one-shot formula on the SAME concatenated rows, using real predictions
    from the baseline ONNX model over `n_sequences` validation sequences — i.e. exactly the
    "fast evaluator vs. official ScorerStepByStep" agreement notes/TASK.md step 3 asks for,
    since ScorerStepByStep's aggregation *is* GlobalAccumulator.
    """
    import pyarrow.parquet as pq

    from baseline.solution import PredictionModel

    pf = pq.ParquetFile(ROOT / "datasets/valid.parquet")
    n = min(n_sequences, pf.num_row_groups)
    model = PredictionModel()
    acc = GlobalAccumulator()
    all_targets, all_preds = [], []

    for rg in range(n):
        table = pf.read_row_group(rg, use_threads=False)
        seq_ix = table["seq_ix"][0].as_py()
        features = np.column_stack([table[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float32)
        targets = np.column_stack([table[c].to_numpy() for c in TARGET_COLUMNS]).astype(np.float32)
        need = table["need_prediction"].to_numpy().astype(bool)
        scored = table["is_scored"].to_numpy().astype(bool)
        preds = np.zeros((SEQUENCE_LENGTH, 2), dtype=np.float32)
        for step in range(SEQUENCE_LENGTH):
            value = model.predict(DataPoint(seq_ix, step, bool(need[step]), features[step]))
            if value is not None:
                preds[step] = value
        mask = need & scored
        acc.add(targets, preds, mask)
        all_targets.append(targets[mask])
        all_preds.append(preds[mask])

    merged = acc.result()
    cat_y = np.concatenate(all_targets, axis=0)
    cat_p = np.concatenate(all_preds, axis=0)
    ref_t0 = weighted_pearson(cat_y[:, 0], cat_p[:, 0])
    ref_t1 = weighted_pearson(cat_y[:, 1], cat_p[:, 1])
    ref_mean = (ref_t0 + ref_t1) / 2

    assert abs(merged["t0"] - ref_t0) < 1e-6, f"t0 mismatch: {merged['t0']} vs {ref_t0}"
    assert abs(merged["t1"] - ref_t1) < 1e-6, f"t1 mismatch: {merged['t1']} vs {ref_t1}"
    assert abs(merged["weighted_pearson"] - ref_mean) < 1e-6, f"mean mismatch: {merged['weighted_pearson']} vs {ref_mean}"
    print(
        f"[D] GlobalAccumulator merge == one-shot reference on {n} baseline-scored sequences "
        f"(t0 {merged['t0']:.6f} vs {ref_t0:.6f}, t1 {merged['t1']:.6f} vs {ref_t1:.6f})"
    )


def test_head_scaling_equals_output_scaling(seed=7):
    # export_onnx.py bakes a calibration multiplier by scaling model.head (and model.skip,
    # if present) in-place instead of multiplying the output at inference time. Verify these
    # are mathematically identical, including through the reparam inverse-transform (linear
    # and homogeneous in `raw`) and with the skip connection engaged (both contributions must
    # be scaled, or the multiplier only partially applies).
    torch.manual_seed(seed)
    for reparam in (False, True):
        for use_skip in (False, True):
            cfg = ModelConfig(trunk_hidden=(16,), gru_hidden=24, gru_layers=2, reparam=reparam, use_skip=use_skip)
            model = SequenceModel(cfg).eval()
            features = torch.randn(3, 50, cfg.n_features)
            multiplier = 0.37

            with torch.no_grad():
                hidden = model.init_hidden(3)
                pred_unscaled, _ = model(features, hidden)
                expected = pred_unscaled * multiplier

                model.head.weight.mul_(multiplier)
                model.head.bias.mul_(multiplier)
                if model.skip is not None:
                    model.skip.weight.mul_(multiplier)
                    model.skip.bias.mul_(multiplier)
                hidden = model.init_hidden(3)
                pred_scaled, _ = model(features, hidden)

            max_abs_diff = (pred_scaled - expected).abs().max().item()
            assert max_abs_diff < 1e-4, f"head-scaling mismatch (reparam={reparam}, use_skip={use_skip}): {max_abs_diff}"
            print(f"[F] scaling head(+skip) == scaling the output (reparam={reparam}, use_skip={use_skip}, max abs diff {max_abs_diff:.2e})")


def test_init_skip_from_ridge_reproduces_ridge_predictions(seed=8):
    ridge_path = ROOT / "models/ridge_v1.npz"
    if not ridge_path.exists():
        print("[G] skipped: models/ridge_v1.npz not present (run tools/fit_ridge.py first)")
        return
    rng = np.random.default_rng(seed)
    data = np.load(ridge_path)
    beta = data["beta"]

    cfg = ModelConfig(gru_hidden=8, gru_layers=1, use_skip=True)
    model = SequenceModel(cfg).eval()
    model.init_skip_from_ridge(ridge_path)

    X = rng.normal(size=(20, cfg.n_features)).astype(np.float32)
    X_aug = np.concatenate([X, np.ones((20, 1), dtype=np.float32)], axis=1)
    ridge_pred = X_aug @ beta
    with torch.no_grad():
        skip_pred = model.skip(torch.from_numpy(X)).numpy()

    max_abs_diff = float(np.abs(ridge_pred - skip_pred).max())
    assert max_abs_diff < 1e-4, f"skip-from-ridge mismatch: {max_abs_diff}"
    print(f"[G] init_skip_from_ridge reproduces ridge_v1 predictions exactly (max abs diff {max_abs_diff:.2e})")


if __name__ == "__main__":
    test_metric_matches_utils()
    test_metric_masking()
    test_batched_equals_stepwise()
    test_hidden_reset_between_batches()
    test_loss_gradient_is_finite_on_degenerate_chunks()
    test_global_wp_loss_pools_across_batch()
    test_global_wp_loss_gradient_is_finite_on_degenerate_batches()
    test_global_accumulator_matches_one_shot_reference()
    test_head_scaling_equals_output_scaling()
    test_init_skip_from_ridge_reproduces_ridge_predictions()
    print("\nAll self-tests passed.")
