"""Two validators: a fast batched torch-based one for training-time monitoring, and a thin
wrapper around utils.ScorerStepByStep (the ground truth, row-by-row) for final verification."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import (
    FEATURE_COLUMNS,
    SEQUENCE_LENGTH,
    TARGET_COLUMNS,
    WARMUP,
    DataPoint,
    GlobalAccumulator,
    ScorerStepByStep,
    weighted_pearson,
)
import pyarrow.parquet as pq


def _read_valid_sequence(pf: pq.ParquetFile, row_group: int):
    table = pf.read_row_group(row_group, use_threads=False)
    features = np.column_stack([table[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float32)
    targets = np.column_stack([table[c].to_numpy() for c in TARGET_COLUMNS]).astype(np.float32)
    is_scored = table["is_scored"].to_numpy().astype(bool)
    need = table["need_prediction"].to_numpy().astype(bool)
    return features, targets, need, is_scored


@torch.no_grad()
def _forward_all(model, pf: pq.ParquetFile, row_groups, device, batch_size):
    """Runs the model once over the given row groups; returns a list of
    (targets, need, scored, raw_prediction) per sequence — raw, no multiplier applied."""
    model.eval()
    cached = []
    for start in range(0, len(row_groups), batch_size):
        idx = row_groups[start : start + batch_size]
        feats, targets_list, need_list, scored_list = zip(
            *(_read_valid_sequence(pf, int(i)) for i in idx)
        )
        features = torch.from_numpy(np.stack(feats)).to(device)
        hidden = model.init_hidden(len(idx), device=device)
        preds, _ = model(features, hidden)
        preds = preds.cpu().numpy().astype(np.float32)  # (B, L, 2)
        for b in range(len(idx)):
            cached.append((targets_list[b], need_list[b], scored_list[b], preds[b]))
    return cached


def _score_cached(cached, output_multiplier: float):
    acc_masked = GlobalAccumulator()
    acc_all_rows = GlobalAccumulator()
    for targets, need, scored, raw in cached:
        p = raw * output_multiplier
        acc_masked.add(targets, p, need & scored)
        acc_all_rows.add(targets, p, need)
    return {"wp_masked": acc_masked.result(), "wp_all_rows": acc_all_rows.result()}


def evaluate_fast(
    model,
    parquet_path: str | Path,
    device: torch.device,
    max_sequences: int | None = 200,
    batch_size: int = 32,
    seed: int = 0,
    output_multiplier: float = 1.0,
):
    """Batched full-sequence forward pass (mathematically identical to a step-by-step causal
    pass for equal-length, unpadded sequences — see selftest.py). Pools rows into the SAME
    utils.GlobalAccumulator used by the official ScorerStepByStep, so the two agree exactly
    rather than merely approximately. Returns both required numbers (see METRIC.md /
    notes/CLAUDE.md hard rules):
      - 'wp_masked'   — official: need_prediction AND is_scored
      - 'wp_all_rows' — need_prediction only (ignores is_scored); divergence between the two
        means the model is fitting the visible mask's character rather than the signal.

    `output_multiplier` scales predictions before scoring — Global WP is only scale-invariant
    BEFORE the metric's [-2,2] clip, so a poorly-scaled model can lose a lot of score to
    clipping alone (see notes/LOG.md 2026-09-17 calibration entry). Pick one with
    `sweep_output_multiplier` below rather than guessing.
    """
    pf = pq.ParquetFile(parquet_path)
    n = pf.num_row_groups
    if max_sequences is not None:
        n = min(n, max_sequences)
    row_groups = np.arange(n)
    if max_sequences is not None and max_sequences < pf.num_row_groups:
        row_groups = np.random.default_rng(seed).choice(pf.num_row_groups, size=n, replace=False)

    cached = _forward_all(model, pf, row_groups, device, batch_size)
    return _score_cached(cached, output_multiplier)


def sweep_output_multiplier(
    model,
    parquet_path: str | Path,
    device: torch.device,
    multipliers=(1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2),
    max_sequences: int | None = 200,
    batch_size: int = 32,
    seed: int = 0,
):
    """Runs the model ONCE (the expensive part), then sweeps output multipliers cheaply
    against the cached raw predictions. Returns (best_multiplier, full_table), where
    full_table is a list of {'c', 'wp_masked', 'wp_all_rows'} dicts. Selection is by
    wp_masked (the official metric), per the CLAUDE.md hard rule."""
    pf = pq.ParquetFile(parquet_path)
    n = pf.num_row_groups
    if max_sequences is not None:
        n = min(n, max_sequences)
    row_groups = np.arange(n)
    if max_sequences is not None and max_sequences < pf.num_row_groups:
        row_groups = np.random.default_rng(seed).choice(pf.num_row_groups, size=n, replace=False)

    cached = _forward_all(model, pf, row_groups, device, batch_size)

    table = []
    best_c, best_wp = 1.0, -2.0
    for c in multipliers:
        result = _score_cached(cached, c)
        row = {"c": c, "wp_masked": result["wp_masked"]["weighted_pearson"], "wp_all_rows": result["wp_all_rows"]["weighted_pearson"]}
        table.append(row)
        if row["wp_masked"] > best_wp:
            best_wp, best_c = row["wp_masked"], c
    return best_c, table


class _TorchStepModel:
    """Adapts a SequenceModel to the PredictionModel.predict(data_point) contract, for use
    with utils.ScorerStepByStep (the slow, ground-truth, row-by-row scorer)."""

    def __init__(self, model, device):
        self.model = model.eval()
        self.device = device
        self.hidden = None
        self.seq_ix = None

    @torch.no_grad()
    def predict(self, data_point: DataPoint):
        if data_point.seq_ix != self.seq_ix:
            self.seq_ix = data_point.seq_ix
            self.hidden = self.model.init_hidden(1, device=self.device)
        x = torch.from_numpy(data_point.state.reshape(1, 1, -1)).to(self.device)
        pred, self.hidden = self.model(x, self.hidden)
        if not data_point.need_prediction:
            return None
        return pred[0, 0].cpu().numpy().astype(np.float32)


def evaluate_official(model, parquet_path: str | Path, device: torch.device):
    """Ground-truth row-by-row scorer via utils.ScorerStepByStep. Slow; use sparingly."""
    return ScorerStepByStep(parquet_path).score(_TorchStepModel(model, device))
