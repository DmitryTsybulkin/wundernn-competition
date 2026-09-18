"""notes/TASK.md step 3: ensemble test, properly calibrated this time. Blends the
calibrated baseline with 02_trunk_global_full (shared-a and per-target-a), reports the
prediction-stream correlation, and optionally adds ridge_v1 as a small third component.

Calibration multipliers below are filled in from tools/per_target_calibration.py's output
before running this script.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import weighted_pearson

ROOT = Path(__file__).resolve().parents[1]
CACHES = {
    "baseline": ROOT / "tools/_baseline_raw_preds_valid.npz",
    "trunk_global_full": ROOT / "tools/_trunk_global_full_raw_preds_valid.npz",
    "ridge_v1": ROOT / "tools/_ridge_v1_raw_preds_valid.npz",
}

# From tools/per_target_calibration.py (2026-09-18): per-target gain was negligible
# (<=0.00009, smaller than local plateau noise) for all three models, so shared-c is used
# throughout, matching that script's own verdict. baseline uses the previously-established
# c=0.35 (the conservative, larger-c choice on a flat plateau — the literal grid optimum
# was c=0.3 at 0.673295, only 0.0007 higher, well within plateau noise). trunk_global_full
# and ridge_v1 both have c=1.0 as the unambiguous best (calibration does not help either).
CALIBRATION = {
    "baseline": {"c0": 0.35, "c1": 0.35},
    "trunk_global_full": {"c0": 1.0, "c1": 1.0},
    "ridge_v1": {"c0": 1.0, "c1": 1.0},
}

A_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def load_calibrated(name):
    d = np.load(CACHES[name])
    preds, targets, need, scored = d["preds"], d["targets"], d["need"], d["scored"]
    mask = need & scored
    c0, c1 = CALIBRATION[name]["c0"], CALIBRATION[name]["c1"]
    p0 = preds[..., 0][mask] * c0
    p1 = preds[..., 1][mask] * c1
    y0 = targets[..., 0][mask]
    y1 = targets[..., 1][mask]
    return y0, y1, p0, p1


def wp_mean(y0, y1, p0, p1):
    return 0.5 * (weighted_pearson(y0, p0) + weighted_pearson(y1, p1))


def main():
    assert all(v["c0"] is not None for v in CALIBRATION.values()), (
        "fill in CALIBRATION from tools/per_target_calibration.py's output before running"
    )

    y0_b, y1_b, p0_b, p1_b = load_calibrated("baseline")
    y0_t, y1_t, p0_t, p1_t = load_calibrated("trunk_global_full")
    # Sanity: masks must line up (same valid.parquet, same need&scored selection).
    assert np.array_equal(y0_b, y0_t) and np.array_equal(y1_b, y1_t), "target arrays diverge between caches — mask mismatch"
    y0, y1 = y0_b, y1_b

    corr0 = np.corrcoef(p0_b, p0_t)[0, 1]
    corr1 = np.corrcoef(p1_b, p1_t)[0, 1]
    print(f"prediction-stream correlation (calibrated, scored rows): t0={corr0:.4f}  t1={corr1:.4f}")
    if corr0 > 0.95 and corr1 > 0.95:
        print("-> highly correlated (>0.95 both targets): little room for ensembling to help.")

    print("\n=== shared-a blend: a*baseline + (1-a)*trunk_global_full ===")
    shared_table = []
    for a in A_GRID:
        p0 = a * p0_b + (1 - a) * p0_t
        p1 = a * p1_b + (1 - a) * p1_t
        mean = wp_mean(y0, y1, p0, p1)
        shared_table.append((a, mean))
        print(f"  a={a:.1f}  wp_masked={mean:.6f}")
    best_a, best_shared_wp = max(shared_table, key=lambda r: r[1])
    print(f"  best a={best_a} wp_masked={best_shared_wp:.6f}")

    print("\n=== per-target-a blend: a_t0, a_t1 chosen independently ===")
    wp0_by_a = {a: weighted_pearson(y0, a * p0_b + (1 - a) * p0_t) for a in A_GRID}
    wp1_by_a = {a: weighted_pearson(y1, a * p1_b + (1 - a) * p1_t) for a in A_GRID}
    best_a0 = max(wp0_by_a, key=wp0_by_a.get)
    best_a1 = max(wp1_by_a, key=wp1_by_a.get)
    best_per_target_wp = 0.5 * (wp0_by_a[best_a0] + wp1_by_a[best_a1])
    print(f"  t0 grid: " + "  ".join(f"a={a}:{wp0_by_a[a]:.6f}" for a in A_GRID))
    print(f"  t1 grid: " + "  ".join(f"a={a}:{wp1_by_a[a]:.6f}" for a in A_GRID))
    print(f"  best a_t0={best_a0} (wp={wp0_by_a[best_a0]:.6f}), best a_t1={best_a1} (wp={wp1_by_a[best_a1]:.6f})")
    print(f"  per-target-a wp_masked={best_per_target_wp:.6f}")

    baseline_solo = wp_mean(y0, y1, p0_b, p1_b)
    trunk_solo = wp_mean(y0, y1, p0_t, p1_t)
    print(f"\nsolo scores (post-calibration): baseline={baseline_solo:.6f}  trunk_global_full={trunk_solo:.6f}")
    best_ensemble = max(best_shared_wp, best_per_target_wp)
    best_component = max(baseline_solo, trunk_solo)
    print(f"best ensemble={best_ensemble:.6f}  best solo component={best_component:.6f}  "
          f"gain={best_ensemble - best_component:+.6f}")

    print("\n=== adding ridge_v1 as a small third component ===")
    d_r = np.load(CACHES["ridge_v1"])
    preds_r, targets_r, need_r, scored_r = d_r["preds"], d_r["targets"], d_r["need"], d_r["scored"]
    mask_r = need_r & scored_r
    c0_r, c1_r = CALIBRATION["ridge_v1"]["c0"], CALIBRATION["ridge_v1"]["c1"]
    p0_r = preds_r[..., 0][mask_r] * c0_r
    p1_r = preds_r[..., 1][mask_r] * c1_r
    assert np.array_equal(targets_r[..., 0][mask_r], y0), "ridge cache mask diverges"

    # Fix the two-way blend at its best weight, then sweep a small ridge contribution.
    p0_2way = best_a0 * p0_b + (1 - best_a0) * p0_t
    p1_2way = best_a1 * p1_b + (1 - best_a1) * p1_t
    for r in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]:
        p0_3 = (1 - r) * p0_2way + r * p0_r
        p1_3 = (1 - r) * p1_2way + r * p1_r
        mean = wp_mean(y0, y1, p0_3, p1_3)
        print(f"  ridge_weight={r:.2f}  wp_masked={mean:.6f}")


if __name__ == "__main__":
    main()
