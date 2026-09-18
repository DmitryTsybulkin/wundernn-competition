"""notes/TASK.md step 3: a proper weighted-ridge reference model, closed-form, streamed.

Per-row weight w = mean(|clip(t0,-2,2)|, |clip(t1,-2,2)|), matching the task's literal
wording ("weights |clip(y,-2,2)| averaged over the two targets") — a single shared weight
and a single X'WX factorization reused for both targets' normal equations.

Only need_prediction rows (step >= WARMUP) are used, subsampled every 4th row per sequence
for speed. Ridge penalty is selected on a held-out slice of TRAIN (never validation), then
the final coefficients are refit on the full 1,000-sequence sample at that penalty.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import FEATURE_COLUMNS, GlobalAccumulator, N_FEATURES, TARGET_COLUMNS, WARMUP

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "datasets/train.parquet"
VALID_PATH = ROOT / "datasets/valid.parquet"
OUT_PATH = ROOT / "models/ridge_v1.npz"

N_SEQUENCES = 1000
N_HELDOUT = 200  # last N_HELDOUT of the N_SEQUENCES used only to pick the ridge penalty
STRIDE = 4
LAMBDAS = [1e-2, 1e-1, 1.0, 10.0, 1e2, 1e3, 1e4, 1e5]
MULTIPLIERS = [1.0, 0.8, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2]


def load_sequence_rows(pf: pq.ParquetFile, rg: int):
    """Returns (X_aug, y) for need_prediction rows, every STRIDE-th, X_aug has a bias column."""
    table = pf.read_row_group(rg, use_threads=False)
    need = table["need_prediction"].to_numpy().astype(bool)
    idx = np.flatnonzero(need)[::STRIDE]
    X = np.column_stack([table[c].to_numpy() for c in FEATURE_COLUMNS])[idx].astype(np.float64)
    y = np.column_stack([table[c].to_numpy() for c in TARGET_COLUMNS])[idx].astype(np.float64)
    X_aug = np.concatenate([X, np.ones((len(idx), 1))], axis=1)
    return X_aug, y


def accumulate(pf: pq.ParquetFile, row_groups):
    d = N_FEATURES + 1
    XtWX = np.zeros((d, d))
    XtWy = np.zeros((d, 2))
    n_rows = 0
    for rg in row_groups:
        X_aug, y = load_sequence_rows(pf, rg)
        yc = np.clip(y, -2.0, 2.0)
        w = 0.5 * (np.abs(yc[:, 0]) + np.abs(yc[:, 1]))
        XtWX += X_aug.T @ (w[:, None] * X_aug)
        XtWy += X_aug.T @ (w[:, None] * y)
        n_rows += len(y)
    return XtWX, XtWy, n_rows


def solve(XtWX, XtWy, lam):
    d = XtWX.shape[0]
    reg = lam * np.eye(d)
    reg[-1, -1] = 0.0  # do not penalize the bias term
    beta = np.linalg.solve(XtWX + reg, XtWy)
    return beta  # (d, 2)


def weighted_pearson_np(y, p):
    yc = np.clip(y, -2.0, 2.0)
    pc = np.clip(p, -2.0, 2.0)
    w = np.abs(yc)
    total = w.sum()
    if total < 1e-8:
        return 0.0
    ym = np.sum(w * yc) / total
    pm = np.sum(w * pc) / total
    cov = np.sum(w * (yc - ym) * (pc - pm)) / total
    vy = np.sum(w * (yc - ym) ** 2) / total
    vp = np.sum(w * (pc - pm) ** 2) / total
    if vy <= 1e-8 or vp <= 1e-8:
        return 0.0
    return float(np.clip(cov / np.sqrt(vy * vp), -1.0, 1.0))


def score_on_rowgroups(pf, row_groups, beta):
    """need_prediction-only weighted Pearson (both targets averaged), no is_scored (train.parquet
    has no such column) — used only for internal ridge-penalty selection, not an official number."""
    ys, ps = [], []
    for rg in row_groups:
        X_aug, y = load_sequence_rows(pf, rg)
        p = X_aug @ beta
        ys.append(y)
        ps.append(p)
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    return 0.5 * (weighted_pearson_np(y[:, 0], p[:, 0]) + weighted_pearson_np(y[:, 1], p[:, 1]))


def main():
    pf_train = pq.ParquetFile(TRAIN_PATH)
    fit_groups = list(range(N_SEQUENCES - N_HELDOUT))
    heldout_groups = list(range(N_SEQUENCES - N_HELDOUT, N_SEQUENCES))

    t0 = time.time()
    print(f"accumulating over {len(fit_groups)} fit sequences (stride {STRIDE}, need_prediction only)...")
    XtWX_fit, XtWy_fit, n_fit_rows = accumulate(pf_train, fit_groups)
    print(f"  {n_fit_rows:,} rows, {time.time()-t0:.0f}s")

    print(f"sweeping ridge penalty on {len(heldout_groups)} held-out TRAIN sequences...")
    best_lam, best_score = None, -1.0
    for lam in LAMBDAS:
        beta = solve(XtWX_fit, XtWy_fit, lam)
        score = score_on_rowgroups(pf_train, heldout_groups, beta)
        print(f"  lambda={lam:<10g} held-out WP-like score={score:.6f}", flush=True)
        if score > best_score:
            best_score, best_lam = score, lam
    print(f"chosen lambda={best_lam:g} (held-out score {best_score:.6f})")

    print(f"\nrefitting on all {N_SEQUENCES} sequences at lambda={best_lam:g}...")
    t1 = time.time()
    XtWX_full, XtWy_full, n_full_rows = accumulate(pf_train, range(N_SEQUENCES))
    beta_final = solve(XtWX_full, XtWy_full, best_lam)
    print(f"  {n_full_rows:,} rows, {time.time()-t1:.0f}s")

    np.savez(
        OUT_PATH,
        beta=beta_final,
        lam=best_lam,
        n_sequences=N_SEQUENCES,
        stride=STRIDE,
        feature_columns=np.array(FEATURE_COLUMNS),
        target_columns=np.array(TARGET_COLUMNS),
        weight_formula="0.5*(|clip(t0,-2,2)|+|clip(t1,-2,2)|)",
        bias_column_index=N_FEATURES,
    )
    print(f"saved {OUT_PATH}")

    print("\nscoring on FULL validation set (official GlobalAccumulator, need_prediction & is_scored)...")
    pf_valid = pq.ParquetFile(VALID_PATH)
    t2 = time.time()
    accs_masked = {c: GlobalAccumulator() for c in MULTIPLIERS}
    accs_all = {c: GlobalAccumulator() for c in MULTIPLIERS}
    for rg in range(pf_valid.num_row_groups):
        table = pf_valid.read_row_group(rg, use_threads=False)
        X = np.column_stack([table[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float64)
        X_aug = np.concatenate([X, np.ones((len(X), 1))], axis=1)
        y = np.column_stack([table[c].to_numpy() for c in TARGET_COLUMNS]).astype(np.float32)
        need = table["need_prediction"].to_numpy().astype(bool)
        scored = table["is_scored"].to_numpy().astype(bool)
        raw_pred = (X_aug @ beta_final).astype(np.float32)
        for c in MULTIPLIERS:
            p = raw_pred * c
            accs_masked[c].add(y, p, need & scored)
            accs_all[c].add(y, p, need)
    print(f"  {time.time()-t2:.0f}s")

    print("\nridge_v1 full-validation results:")
    for c in MULTIPLIERS:
        rm, ra = accs_masked[c].result(), accs_all[c].result()
        print(f"  c={c:<5} wp_masked={rm['weighted_pearson']:.6f}  wp_all_rows={ra['weighted_pearson']:.6f}")


if __name__ == "__main__":
    main()
