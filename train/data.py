"""Streaming TBPTT batches over train.parquet. One row group == one full 20,000-row sequence."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import FEATURE_COLUMNS, SEQUENCE_LENGTH, TARGET_COLUMNS, WARMUP


@dataclass
class Window:
    features: np.ndarray   # (B, L, 112) float32
    t0: np.ndarray         # (B, L) float32
    t1: np.ndarray         # (B, L) float32
    mask: np.ndarray       # (B, L) float32, 1.0 where need_prediction (step_in_seq >= WARMUP)
    is_first: bool         # True if this is the first window of a fresh batch of sequences (reset hidden)


def read_sequence(parquet_file: pq.ParquetFile, row_group: int):
    table = parquet_file.read_row_group(row_group, use_threads=False)
    features = np.column_stack([table[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float32)
    t0 = table[TARGET_COLUMNS[0]].to_numpy().astype(np.float32)
    t1 = table[TARGET_COLUMNS[1]].to_numpy().astype(np.float32)
    return features, t0, t1


class TBPTTLoader:
    """Yields Window objects. Each epoch: shuffle row groups, batch them, slide a length-L
    window across all 20,000 steps of each batch carrying hidden state, then move to the
    next batch of sequences (fresh hidden state, is_first=True)."""

    def __init__(
        self,
        parquet_path: str | Path,
        batch_size: int = 16,
        chunk_len: int = 2000,
        max_row_groups: int | None = None,
        seed: int = 0,
    ):
        self.pf = pq.ParquetFile(parquet_path)
        self.batch_size = batch_size
        self.chunk_len = chunk_len
        self.n_row_groups = self.pf.num_row_groups
        if max_row_groups is not None:
            self.n_row_groups = min(self.n_row_groups, max_row_groups)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        n_batches = self.n_row_groups // self.batch_size
        n_windows = SEQUENCE_LENGTH // self.chunk_len
        return n_batches * n_windows

    def epoch(self):
        order = self.rng.permutation(self.n_row_groups)
        n_batches = self.n_row_groups // self.batch_size
        for b in range(n_batches):
            idx = order[b * self.batch_size : (b + 1) * self.batch_size]
            feats, t0s, t1s = zip(*(read_sequence(self.pf, int(i)) for i in idx))
            features = np.stack(feats)  # (B, 20000, 112)
            t0 = np.stack(t0s)          # (B, 20000)
            t1 = np.stack(t1s)          # (B, 20000)
            step = np.arange(SEQUENCE_LENGTH)
            need_pred = (step >= WARMUP).astype(np.float32)  # (20000,)

            for start in range(0, SEQUENCE_LENGTH, self.chunk_len):
                end = start + self.chunk_len
                mask = np.broadcast_to(need_pred[start:end], (len(idx), end - start))
                yield Window(
                    features=features[:, start:end],
                    t0=t0[:, start:end],
                    t1=t1[:, start:end],
                    mask=mask.astype(np.float32),
                    is_first=(start == 0),
                )
