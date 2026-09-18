"""Differentiable weighted-Pearson loss, matching METRIC.md's clipping/weighting exactly."""
from __future__ import annotations

import torch

METRIC_CLIP = 2.0
EPS = 1e-8


def weighted_pearson_torch(
    target: torch.Tensor, pred: torch.Tensor, weight_mask: torch.Tensor, clip: float = METRIC_CLIP
) -> torch.Tensor:
    """target, pred, weight_mask: (..., L). weight_mask is 1.0 on rows to include, 0.0 elsewhere.

    Returns per-leading-dim weighted Pearson correlation, shape (...).
    Rows with zero total weight yield a correlation of 0 (matches METRIC.md).
    """
    y = torch.clamp(target, -clip, clip)
    p = torch.clamp(pred, -clip, clip)
    w = torch.abs(y) * weight_mask
    total = w.sum(dim=-1)
    safe_total = total.clamp_min(EPS)

    y_mean = (w * y).sum(dim=-1) / safe_total
    p_mean = (w * p).sum(dim=-1) / safe_total
    yc = y - y_mean.unsqueeze(-1)
    pc = p - p_mean.unsqueeze(-1)

    cov = (w * yc * pc).sum(dim=-1) / safe_total
    var_y = (w * yc * yc).sum(dim=-1) / safe_total
    var_p = (w * pc * pc).sum(dim=-1) / safe_total

    # sqrt's local gradient is infinite at 0. Even a masked-out `where` branch still
    # backprops `grad_output * 0`, and `0 * inf = nan` — so keep the sqrt argument
    # strictly positive (a fixed tiny constant, not clamp_min, which would zero the
    # gradient exactly at the boundary and reintroduce the same trap).
    denom = torch.sqrt((var_y * var_p).clamp_min(0.0) + EPS * EPS)
    corr = cov / denom

    valid = (total > EPS) & (var_y > EPS) & (var_p > EPS)
    corr = torch.where(valid, corr, torch.zeros_like(corr))
    return corr.clamp(-1.0, 1.0)


def chunk_wp_loss(
    t0_hat: torch.Tensor,
    t1_hat: torch.Tensor,
    t0: torch.Tensor,
    t1: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """OLD / kept for comparison only — do not use for Global WP. All inputs (B, L). mask
    (B, L) is 1.0 where need_prediction (rows eligible for loss).

    Computes one weighted correlation PER BATCH ITEM (per sequence-chunk), then averages
    equally across the batch. This implicitly equalizes each sequence's contribution and
    per-sequence-centers the correlation — exactly the cross-sequence calibration the old
    per-block metric rewarded and Global WP does not (measured: per-sequence z-scoring of
    predictions costs -0.049 Global WP). Returns (loss, mean_wp_t0, mean_wp_t1) where
    loss = -mean(0.5*(wp_t0+wp_t1)) over the batch.
    """
    wp0 = weighted_pearson_torch(t0, t0_hat, mask)
    wp1 = weighted_pearson_torch(t1, t1_hat, mask)
    per_seq = 0.5 * (wp0 + wp1)
    loss = -per_seq.mean()
    return loss, wp0.mean(), wp1.mean()


def global_wp_loss(
    t0_hat: torch.Tensor,
    t1_hat: torch.Tensor,
    t0: torch.Tensor,
    t1: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Matches Global WP (METRIC_SPECIFICATION='global_wp', utils.GlobalAccumulator): pools
    ALL rows in the batch — every sequence and timestep together — into a SINGLE weighted
    correlation per target, then averages the two targets. All inputs (B, L). Sequences and
    timesteps are not renormalized or equalized: a row's influence is exactly its
    |clip(y,-2,2)| weight, matching the metric. This is the batch-local finite-sample
    approximation of the metric's dataset-global pooling.

    Returns (loss, wp_t0, wp_t1) as 0-dim tensors; loss = -0.5*(wp_t0+wp_t1).
    """
    wp0 = weighted_pearson_torch(t0.reshape(-1), t0_hat.reshape(-1), mask.reshape(-1))
    wp1 = weighted_pearson_torch(t1.reshape(-1), t1_hat.reshape(-1), mask.reshape(-1))
    loss = -0.5 * (wp0 + wp1)
    return loss, wp0, wp1
