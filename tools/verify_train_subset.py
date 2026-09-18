"""notes/TASK.md step 4 verification: refit ridge_v1's exact recipe (reusing
tools/fit_ridge.py's functions unmodified) on datasets/train_subset.parquet alone, then
score on the full validation set. Should land close to ridge_v1's 0.586440 — a materially
lower number would mean the 2,500-sequence sampling or the float16 conversion broke
something.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.fit_ridge import LAMBDAS, accumulate, score_on_rowgroups, solve
from utils import FEATURE_COLUMNS, GlobalAccumulator, TARGET_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
SUBSET_PATH = ROOT / "datasets/train_subset.parquet"
VALID_PATH = ROOT / "datasets/valid.parquet"


def main():
    pf = pq.ParquetFile(SUBSET_PATH)
    n = pf.num_row_groups
    n_heldout = max(1, n // 5)
    fit_groups = list(range(n - n_heldout))
    heldout_groups = list(range(n - n_heldout, n))
    print(f"train_subset.parquet: {n} sequences ({pf.schema_arrow.field('i0_p0').type} features)")

    t0 = time.time()
    XtWX_fit, XtWy_fit, n_fit_rows = accumulate(pf, fit_groups)
    print(f"accumulated over {len(fit_groups)} fit sequences ({n_fit_rows:,} rows), {time.time()-t0:.0f}s")

    best_lam, best_score = None, -1.0
    for lam in LAMBDAS:
        beta = solve(XtWX_fit, XtWy_fit, lam)
        score = score_on_rowgroups(pf, heldout_groups, beta)
        print(f"  lambda={lam:<10g} held-out score={score:.6f}")
        if score > best_score:
            best_score, best_lam = score, lam
    print(f"chosen lambda={best_lam:g}")

    XtWX_full, XtWy_full, n_full_rows = accumulate(pf, range(n))
    beta_final = solve(XtWX_full, XtWy_full, best_lam)
    print(f"refit on all {n} sequences ({n_full_rows:,} rows)")

    pf_valid = pq.ParquetFile(VALID_PATH)
    acc = GlobalAccumulator()
    for rg in range(pf_valid.num_row_groups):
        table = pf_valid.read_row_group(rg, use_threads=False)
        X = np.column_stack([table[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float64)
        X_aug = np.concatenate([X, np.ones((len(X), 1))], axis=1)
        y = np.column_stack([table[c].to_numpy() for c in TARGET_COLUMNS]).astype(np.float32)
        need = table["need_prediction"].to_numpy().astype(bool)
        scored = table["is_scored"].to_numpy().astype(bool)
        pred = (X_aug @ beta_final).astype(np.float32)
        acc.add(y, pred, need & scored)

    result = acc.result()
    print(f"\nfull-validation wp_masked (subset-fit ridge) = {result['weighted_pearson']:.6f}")
    print(f"reference (ridge_v1, fit on full-precision train.parquet)         = 0.586440")
    print(f"difference = {result['weighted_pearson'] - 0.586440:+.6f}")


if __name__ == "__main__":
    main()
