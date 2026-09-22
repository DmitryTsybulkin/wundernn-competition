"""notes/TASK.md step 1: cache every model's raw (uncalibrated, unclipped) validation
predictions once, alongside targets/need/scored, so every later analysis (per-target
calibration, ensembling) is pure numpy over cached arrays — no re-running any model.

All caches share the same format: {preds, targets, need, scored}, shapes
(1873, 20000, 2) / (1873, 20000, 2) / (1873, 20000) / (1873, 20000).

The baseline's cache already exists at tools/_baseline_raw_preds_valid.npz (from the
2026-09-17 calibration task) and is reused as-is, per TASK.md's instruction.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train.eval import _forward_all
from train.model import SequenceModel
from utils import FEATURE_COLUMNS, TARGET_COLUMNS
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
VALID_PATH = ROOT / "datasets/valid.parquet"

BASELINE_CACHE = ROOT / "tools/_baseline_raw_preds_valid.npz"
TRUNK_CACHE = ROOT / "tools/_trunk_global_full_raw_preds_valid.npz"
RIDGE_CACHE = ROOT / "tools/_ridge_v1_raw_preds_valid.npz"


def cache_trunk_global_full(force: bool = False):
    if TRUNK_CACHE.exists() and not force:
        print(f"using cached predictions at {TRUNK_CACHE}")
        return TRUNK_CACHE

    ckpt = torch.load(ROOT / "runs/02_trunk_global_full/best.pt", map_location="cpu", weights_only=False)
    model = SequenceModel(ckpt["model_cfg"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    pf = pq.ParquetFile(VALID_PATH)
    n = pf.num_row_groups
    t0 = time.time()
    cached = _forward_all(model, pf, np.arange(n), torch.device("cpu"), batch_size=32)
    print(f"forward pass over {n} sequences: {time.time()-t0:.0f}s")

    preds = np.stack([c[3] for c in cached])          # (N, 20000, 2)
    targets = np.stack([c[0] for c in cached])         # (N, 20000, 2)
    need = np.stack([c[1] for c in cached])            # (N, 20000)
    scored = np.stack([c[2] for c in cached])          # (N, 20000)

    np.savez(TRUNK_CACHE, preds=preds, targets=targets, need=need, scored=scored)
    print(f"cached to {TRUNK_CACHE} ({TRUNK_CACHE.stat().st_size / 1e6:.0f} MB)")
    return TRUNK_CACHE


def cache_checkpoint(ckpt_path: Path, out_path: Path, force: bool = False):
    """Generic version of cache_trunk_global_full: any SequenceModel checkpoint (GRU or TCN)."""
    if out_path.exists() and not force:
        print(f"using cached predictions at {out_path}")
        return out_path

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = SequenceModel(ckpt["model_cfg"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    pf = pq.ParquetFile(VALID_PATH)
    n = pf.num_row_groups
    t0 = time.time()
    cached = _forward_all(model, pf, np.arange(n), torch.device("cpu"), batch_size=32)
    print(f"forward pass over {n} sequences ({ckpt_path}): {time.time()-t0:.0f}s", flush=True)

    np.savez(
        out_path,
        preds=np.stack([c[3] for c in cached]), targets=np.stack([c[0] for c in cached]),
        need=np.stack([c[1] for c in cached]), scored=np.stack([c[2] for c in cached]),
    )
    print(f"cached to {out_path} ({out_path.stat().st_size / 1e6:.0f} MB)", flush=True)
    return out_path


def cache_ridge_v1(force: bool = False):
    if RIDGE_CACHE.exists() and not force:
        print(f"using cached predictions at {RIDGE_CACHE}")
        return RIDGE_CACHE

    data = np.load(ROOT / "models/ridge_v1.npz")
    beta = data["beta"]  # (113, 2)

    pf = pq.ParquetFile(VALID_PATH)
    n = pf.num_row_groups
    t0 = time.time()

    all_preds, all_targets, all_need, all_scored = [], [], [], []
    for rg in range(n):
        table = pf.read_row_group(rg, use_threads=False)
        X = np.column_stack([table[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float64)
        X_aug = np.concatenate([X, np.ones((len(X), 1))], axis=1)
        pred = (X_aug @ beta).astype(np.float32)
        targets = np.column_stack([table[c].to_numpy() for c in TARGET_COLUMNS]).astype(np.float32)
        need = table["need_prediction"].to_numpy().astype(bool)
        scored = table["is_scored"].to_numpy().astype(bool)
        all_preds.append(pred)
        all_targets.append(targets)
        all_need.append(need)
        all_scored.append(scored)
    print(f"matmul pass over {n} sequences: {time.time()-t0:.0f}s")

    np.savez(
        RIDGE_CACHE,
        preds=np.stack(all_preds), targets=np.stack(all_targets),
        need=np.stack(all_need), scored=np.stack(all_scored),
    )
    print(f"cached to {RIDGE_CACHE} ({RIDGE_CACHE.stat().st_size / 1e6:.0f} MB)")
    return RIDGE_CACHE


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--checkpoint":
    # usage: python3 tools/predict_cache.py --checkpoint runs/X/best.pt tools/_name_raw_preds_valid.npz
    cache_checkpoint(ROOT / sys.argv[2], ROOT / sys.argv[3])
elif __name__ == "__main__":
    print(f"baseline cache: {'exists, reused' if BASELINE_CACHE.exists() else 'MISSING'} "
          f"({BASELINE_CACHE.stat().st_size / 1e6:.0f} MB)" if BASELINE_CACHE.exists() else "baseline cache MISSING")
    cache_trunk_global_full()
    cache_ridge_v1()
