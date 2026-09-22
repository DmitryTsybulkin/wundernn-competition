"""Streaming TBPTT batches over train.parquet. One row group == one full 20,000-row sequence.

The loader is resumable: `state_dict()` returns the position of the NEXT window to be yielded
(epoch RNG state + batch index + window index) and `load_state_dict()` makes the next
`epoch()` call continue from exactly there, reproducing the same shuffle and the same
batches. The trainer saves this alongside the model/optimizer state (notes/TASK_gpu_prep.md).
"""
from __future__ import annotations

import copy
import sys
from concurrent.futures import ThreadPoolExecutor
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
    """Features are returned in the file's native dtype (float16 in the *_subset files,
    float32 in the originals); Loader upcasts one window at a time, which is exact."""
    table = parquet_file.read_row_group(row_group, use_threads=False)
    features = np.column_stack([table[c].to_numpy() for c in FEATURE_COLUMNS])
    t0 = table[TARGET_COLUMNS[0]].to_numpy().astype(np.float32)
    t1 = table[TARGET_COLUMNS[1]].to_numpy().astype(np.float32)
    return features, t0, t1


class TBPTTLoader:
    """Yields Window objects. Each epoch: shuffle row groups, batch them, slide a length-L
    window across all 20,000 steps of each batch carrying hidden state, then move to the
    next batch of sequences (fresh hidden state, is_first=True).

    Memory: a batch is held in the file's native dtype (float16 for the subsets => 2.3 GB at
    batch 512; float32 originals => 4.6 GB) and only one window is upcast to float32 at a time.
    With prefetch=True the next batch is read on a background thread while the current one is
    being trained on (doubles the resident batch memory; does not touch the RNG, so it cannot
    change what is yielded)."""

    def __init__(
        self,
        parquet_path: str | Path,
        batch_size: int = 16,
        chunk_len: int = 2000,
        max_row_groups: int | None = None,
        seed: int = 0,
        prefetch: bool = False,
    ):
        self.pf = pq.ParquetFile(parquet_path)
        self.batch_size = batch_size
        self.chunk_len = chunk_len
        self.n_row_groups = self.pf.num_row_groups
        if max_row_groups is not None:
            self.n_row_groups = min(self.n_row_groups, max_row_groups)
        self.rng = np.random.default_rng(seed)
        self.prefetch = prefetch
        self._pool = ThreadPoolExecutor(max_workers=1) if prefetch else None
        # Position of the next window to yield, within the current epoch, plus the RNG state
        # the current epoch's permutation was drawn from.
        self._epoch_rng_state = copy.deepcopy(self.rng.bit_generator.state)
        self._next_batch = 0
        self._next_window = 0
        self._resume = None

    def __len__(self):
        n_batches = self.n_row_groups // self.batch_size
        n_windows = SEQUENCE_LENGTH // self.chunk_len
        return n_batches * n_windows

    # ---- resumability -------------------------------------------------------------------
    def state_dict(self) -> dict:
        return {
            "epoch_rng_state": copy.deepcopy(self._epoch_rng_state),
            "next_batch": self._next_batch,
            "next_window": self._next_window,
            "batch_size": self.batch_size,
            "chunk_len": self.chunk_len,
            "n_row_groups": self.n_row_groups,
        }

    def load_state_dict(self, state: dict):
        for k in ("batch_size", "chunk_len", "n_row_groups"):
            if state[k] != getattr(self, k):
                raise ValueError(
                    f"cannot resume: loader {k} was {state[k]} in the checkpoint but is "
                    f"{getattr(self, k)} now (data position would not be reproducible)"
                )
        self._resume = state

    # ---- batches --------------------------------------------------------------------------
    def _load_batch(self, idx):
        first = read_sequence(self.pf, int(idx[0]))
        features = np.empty((len(idx), SEQUENCE_LENGTH, first[0].shape[1]), dtype=first[0].dtype)
        t0 = np.empty((len(idx), SEQUENCE_LENGTH), dtype=np.float32)
        t1 = np.empty((len(idx), SEQUENCE_LENGTH), dtype=np.float32)
        features[0], t0[0], t1[0] = first
        for j in range(1, len(idx)):
            features[j], t0[j], t1[j] = read_sequence(self.pf, int(idx[j]))
        return features, t0, t1

    def epoch(self):
        if self._resume is not None:
            state, self._resume = self._resume, None
            self.rng.bit_generator.state = copy.deepcopy(state["epoch_rng_state"])
            start_batch, start_window = state["next_batch"], state["next_window"]
        else:
            start_batch = start_window = 0
        self._epoch_rng_state = copy.deepcopy(self.rng.bit_generator.state)
        order = self.rng.permutation(self.n_row_groups)
        n_batches = self.n_row_groups // self.batch_size
        n_windows = -(-SEQUENCE_LENGTH // self.chunk_len)  # ceil: a short final window, as before
        step = np.arange(SEQUENCE_LENGTH)
        need_pred = (step >= WARMUP).astype(np.float32)  # (20000,)

        def batch_idx(b):
            return order[b * self.batch_size : (b + 1) * self.batch_size]

        pending = None
        for b in range(start_batch, n_batches):
            if pending is not None:
                features, t0, t1 = pending.result()
            else:
                features, t0, t1 = self._load_batch(batch_idx(b))
            pending = (
                self._pool.submit(self._load_batch, batch_idx(b + 1))
                if self._pool is not None and b + 1 < n_batches else None
            )
            w_from = start_window if b == start_batch else 0
            for wi in range(w_from, n_windows):
                # Position of the NEXT window, set before yielding so a checkpoint taken while
                # this window is being trained on records where to continue.
                if wi + 1 < n_windows:
                    self._next_batch, self._next_window = b, wi + 1
                else:
                    self._next_batch, self._next_window = b + 1, 0
                start = wi * self.chunk_len
                end = min(start + self.chunk_len, SEQUENCE_LENGTH)
                mask = np.broadcast_to(need_pred[start:end], (len(features), end - start))
                yield Window(
                    features=features[:, start:end].astype(np.float32),
                    t0=t0[:, start:end],
                    t1=t1[:, start:end],
                    mask=mask.astype(np.float32),
                    is_first=(wi == 0),
                )
        # Epoch complete: the next epoch starts from the post-permutation RNG state.
        self._epoch_rng_state = copy.deepcopy(self.rng.bit_generator.state)
        self._next_batch = self._next_window = 0
