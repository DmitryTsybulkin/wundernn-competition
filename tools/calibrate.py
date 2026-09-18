"""notes/TASK.md step 1-2: confirm the output-clipping effect on the FULL validation set,
then (if confirmed) package a calibrated-baseline submission candidate.

Does NOT modify baseline/solution.py or baseline.onnx. Caches the baseline's raw
(unmultiplied) predictions over all 1,873 validation sequences to disk once, then sweeps
output multipliers against the official GlobalAccumulator.
"""
from __future__ import annotations

import argparse
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baseline.solution import PredictionModel
from utils import DataPoint, FEATURE_COLUMNS, GlobalAccumulator, SEQUENCE_LENGTH, TARGET_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
VALID_PATH = ROOT / "datasets/valid.parquet"
CACHE_PATH = ROOT / "tools/_baseline_raw_preds_valid.npz"
MULTIPLIERS = [1.0, 0.8, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2]


def compute_and_cache_predictions(force: bool = False) -> Path:
    if CACHE_PATH.exists() and not force:
        print(f"using cached predictions at {CACHE_PATH}")
        return CACHE_PATH

    pf = pq.ParquetFile(VALID_PATH)
    n = pf.num_row_groups
    model = PredictionModel()

    all_preds = np.empty((n, SEQUENCE_LENGTH, 2), dtype=np.float32)
    all_targets = np.empty((n, SEQUENCE_LENGTH, 2), dtype=np.float32)
    all_need = np.empty((n, SEQUENCE_LENGTH), dtype=bool)
    all_scored = np.empty((n, SEQUENCE_LENGTH), dtype=bool)

    t_start = time.time()
    for rg in range(n):
        table = pf.read_row_group(rg, use_threads=False)
        seq_ix = table["seq_ix"][0].as_py()
        features = np.column_stack([table[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float32)
        targets = np.column_stack([table[c].to_numpy() for c in TARGET_COLUMNS]).astype(np.float32)
        need = table["need_prediction"].to_numpy().astype(bool)
        scored = table["is_scored"].to_numpy().astype(bool)
        preds = np.zeros((SEQUENCE_LENGTH, 2), dtype=np.float32)
        for step in range(SEQUENCE_LENGTH):
            v = model.predict(DataPoint(seq_ix, step, bool(need[step]), features[step]))
            if v is not None:
                preds[step] = v
        all_preds[rg] = preds
        all_targets[rg] = targets
        all_need[rg] = need
        all_scored[rg] = scored
        if (rg + 1) % 200 == 0:
            print(f"{rg + 1}/{n} sequences, {time.time() - t_start:.0f}s elapsed", flush=True)

    np.savez(
        CACHE_PATH, preds=all_preds, targets=all_targets, need=all_need, scored=all_scored
    )
    print(f"cached raw predictions to {CACHE_PATH} ({CACHE_PATH.stat().st_size / 1e6:.0f} MB)")
    return CACHE_PATH


def sweep(cache_path: Path, multipliers=MULTIPLIERS):
    data = np.load(cache_path)
    preds, targets, need, scored = data["preds"], data["targets"], data["need"], data["scored"]
    n = preds.shape[0]

    rows = []
    for c in multipliers:
        acc_masked, acc_all = GlobalAccumulator(), GlobalAccumulator()
        clipped_frac_num, clipped_frac_den = 0, 0
        for i in range(n):
            p = preds[i] * c
            mask_masked = need[i] & scored[i]
            acc_masked.add(targets[i], p, mask_masked)
            acc_all.add(targets[i], p, need[i])
            clipped_frac_num += int(np.sum(np.abs(p[mask_masked]) > 2.0))
            clipped_frac_den += int(mask_masked.sum())
        rm, ra = acc_masked.result(), acc_all.result()
        row = {
            "c": c,
            "wp_masked": rm["weighted_pearson"],
            "wp_all_rows": ra["weighted_pearson"],
            "frac_clipped_scored": clipped_frac_num / clipped_frac_den if clipped_frac_den else float("nan"),
        }
        rows.append(row)
        print(
            f"c={c:<5} wp_masked={row['wp_masked']:.6f}  wp_all_rows={row['wp_all_rows']:.6f}  "
            f"frac|c*pred|>2 (scored)={row['frac_clipped_scored']:.4f}",
            flush=True,
        )
    return rows


def package_submission(multiplier: float, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline_src = (ROOT / "baseline/solution.py").read_text()

    # Insert the multiplier as a scale applied to the returned prediction, without touching
    # baseline/solution.py itself (kept pristine per TASK.md).
    marker = "        return np.asarray(prediction[0, 0], dtype=np.float32)"
    assert marker in baseline_src, "baseline/solution.py structure changed; update the marker"
    calibrated_src = baseline_src.replace(
        marker,
        f"        return (np.asarray(prediction[0, 0], dtype=np.float32) * np.float32({multiplier})).astype(np.float32)",
    )
    calibrated_src = calibrated_src.replace(
        '"""Inference-only stateful ONNX baseline for the public competition kit."""',
        f'"""Inference-only stateful ONNX baseline, output-calibrated by a constant '
        f'{multiplier}x multiplier (see notes/LOG.md 2026-09-17 calibration entry). Generated '
        f'by tools/calibrate.py from the pristine baseline/solution.py — do not hand-edit."""',
    )
    (out_dir / "solution.py").write_text(calibrated_src)

    import shutil
    shutil.copy(ROOT / "baseline/baseline.onnx", out_dir / "baseline.onnx")

    zip_path = out_dir.parent / f"{out_dir.name}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_dir / "solution.py", "solution.py")
        zf.write(out_dir / "baseline.onnx", "baseline.onnx")
    size_mb = zip_path.stat().st_size / 1e6
    print(f"wrote {zip_path} ({size_mb:.2f} MB, limit 20 MB)")
    assert size_mb < 20
    return zip_path


def verify_determinism_and_throughput(solution_dir: Path, n_rows: int = 200_000):
    sys.path.insert(0, str(solution_dir))
    import importlib
    solution = importlib.import_module("solution")

    pf = pq.ParquetFile(VALID_PATH)

    def run_once():
        model = solution.PredictionModel()
        preds = []
        rows_done = 0
        for rg in range(pf.num_row_groups):
            if rows_done >= n_rows:
                break
            table = pf.read_row_group(rg, use_threads=False)
            seq_ix = table["seq_ix"][0].as_py()
            features = np.column_stack([table[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float32)
            need = table["need_prediction"].to_numpy().astype(bool)
            for step in range(SEQUENCE_LENGTH):
                v = model.predict(DataPoint(seq_ix, step, bool(need[step]), features[step]))
                preds.append(v if v is not None else np.zeros(2, dtype=np.float32))
                rows_done += 1
                if rows_done >= n_rows:
                    break
        return np.stack(preds)

    p1 = run_once()
    p2 = run_once()
    assert np.array_equal(p1, p2), "non-deterministic predictions across runs!"
    print(f"determinism OK: identical predictions across two runs ({len(p1)} rows)")

    model = solution.PredictionModel()
    t_start = time.perf_counter()
    rows_done = 0
    for rg in range(pf.num_row_groups):
        if rows_done >= n_rows:
            break
        table = pf.read_row_group(rg, use_threads=False)
        seq_ix = table["seq_ix"][0].as_py()
        features = np.column_stack([table[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float32)
        need = table["need_prediction"].to_numpy().astype(bool)
        for step in range(SEQUENCE_LENGTH):
            model.predict(DataPoint(seq_ix, step, bool(need[step]), features[step]))
            rows_done += 1
            if rows_done >= n_rows:
                break
    elapsed = time.perf_counter() - t_start
    us_per_row = elapsed / rows_done * 1e6
    projected_minutes = us_per_row * 39_400_000 / 1e6 / 60
    print(
        f"throughput: {rows_done:,} rows in {elapsed:.2f}s -> {us_per_row:.2f} us/row "
        f"(budget ~91 us/row). Projected full test set (1,970 seq, 39.4M rows): "
        f"{projected_minutes:.1f} min (budget 60 min)."
    )
    return us_per_row


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument("--package", type=float, default=None, help="multiplier to package as a submission")
    args = parser.parse_args()

    cache_path = compute_and_cache_predictions(force=args.force_recompute)
    print(f"\nc=1.0 sanity check should match 0.6170522889725805 exactly:\n")
    rows = sweep(cache_path)

    if args.package is not None:
        out_dir = ROOT / "submissions/baseline_calibrated"
        zip_path = package_submission(args.package, out_dir)
        verify_determinism_and_throughput(out_dir)
