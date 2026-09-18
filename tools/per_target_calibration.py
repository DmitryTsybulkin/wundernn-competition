"""notes/TASK.md step 2: per-target output calibration. The shared-c sweep from the last
task used one multiplier for both targets; the baseline is lopsided (t0 0.587913 vs
t1 0.646192), so an independent c_t0/c_t1 grid search may do better. Also reports the
neighbourhood around the best pair to guard against over-tuning on a flat plateau.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import weighted_pearson

ROOT = Path(__file__).resolve().parents[1]
GRID = [1.0, 0.8, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2]

CACHES = {
    "baseline": ROOT / "tools/_baseline_raw_preds_valid.npz",
    "trunk_global_full": ROOT / "tools/_trunk_global_full_raw_preds_valid.npz",
    "ridge_v1": ROOT / "tools/_ridge_v1_raw_preds_valid.npz",
}


def load(name):
    d = np.load(CACHES[name])
    return d["preds"], d["targets"], d["need"], d["scored"]


def wp_for_target(preds_k, targets_k, mask):
    return weighted_pearson(targets_k[mask], preds_k[mask])


def sweep_shared_and_per_target(name):
    preds, targets, need, scored = load(name)
    mask = need & scored

    # Flatten once (this is the expensive part per-c if redone per grid point; precompute
    # the masked, per-target arrays once).
    y0 = targets[..., 0][mask]
    y1 = targets[..., 1][mask]
    p0_raw = preds[..., 0][mask]
    p1_raw = preds[..., 1][mask]

    def wp_shared(c):
        wp0 = weighted_pearson(y0, p0_raw * c)
        wp1 = weighted_pearson(y1, p1_raw * c)
        return 0.5 * (wp0 + wp1), wp0, wp1

    shared_table = [(c, *wp_shared(c)) for c in GRID]
    best_shared = max(shared_table, key=lambda r: r[1])
    print(f"\n=== {name}: shared c ===")
    for c, mean, wp0, wp1 in shared_table:
        marker = "  <-- best" if c == best_shared[0] else ""
        print(f"  c={c:<5} wp_masked={mean:.6f}  (t0={wp0:.6f} t1={wp1:.6f}){marker}")

    # Independent per-target grid: each target's wp only depends on its own multiplier,
    # so the best pair is just the argmax of each column independently.
    wp0_by_c = {c: weighted_pearson(y0, p0_raw * c) for c in GRID}
    wp1_by_c = {c: weighted_pearson(y1, p1_raw * c) for c in GRID}
    best_c0 = max(wp0_by_c, key=wp0_by_c.get)
    best_c1 = max(wp1_by_c, key=wp1_by_c.get)
    best_per_target = 0.5 * (wp0_by_c[best_c0] + wp1_by_c[best_c1])

    print(f"\n=== {name}: independent per-target ===")
    print(f"  t0 grid: " + "  ".join(f"c={c}:{wp0_by_c[c]:.6f}" for c in GRID))
    print(f"  t1 grid: " + "  ".join(f"c={c}:{wp1_by_c[c]:.6f}" for c in GRID))
    print(f"  best c_t0={best_c0} (wp={wp0_by_c[best_c0]:.6f}), best c_t1={best_c1} (wp={wp1_by_c[best_c1]:.6f})")
    print(f"  per-target wp_masked={best_per_target:.6f}  vs shared-c best={best_shared[1]:.6f}  "
          f"gain={best_per_target - best_shared[1]:+.6f}")

    # Flatness around the per-target optimum: neighbouring grid values for each target.
    def neighbourhood(wp_by_c, best_c):
        idx = GRID.index(best_c)
        lo = GRID[idx - 1] if idx > 0 else None
        hi = GRID[idx + 1] if idx < len(GRID) - 1 else None
        return {k: wp_by_c[k] for k in (lo, best_c, hi) if k is not None}

    print(f"  t0 neighbourhood: {neighbourhood(wp0_by_c, best_c0)}")
    print(f"  t1 neighbourhood: {neighbourhood(wp1_by_c, best_c1)}")

    spread0 = max(neighbourhood(wp0_by_c, best_c0).values()) - min(neighbourhood(wp0_by_c, best_c0).values())
    spread1 = max(neighbourhood(wp1_by_c, best_c1).values()) - min(neighbourhood(wp1_by_c, best_c1).values())
    gain = best_per_target - best_shared[1]
    verdict = "PREFER SHARED (gain smaller than local plateau variation)" if gain < max(spread0, spread1) else "per-target gain exceeds local plateau noise"
    print(f"  plateau spread (t0/t1): {spread0:.6f} / {spread1:.6f}  vs gain {gain:+.6f}  -> {verdict}")

    return {
        "best_shared_c": best_shared[0], "best_shared_wp": best_shared[1],
        "best_c0": best_c0, "best_c1": best_c1, "best_per_target_wp": best_per_target,
        "gain": gain, "spread0": spread0, "spread1": spread1, "verdict": verdict,
    }


if __name__ == "__main__":
    results = {}
    for name in CACHES:
        results[name] = sweep_shared_and_per_target(name)
    print("\n\n=== SUMMARY ===")
    for name, r in results.items():
        print(f"{name}: shared c={r['best_shared_c']} wp={r['best_shared_wp']:.6f} | "
              f"per-target c_t0={r['best_c0']} c_t1={r['best_c1']} wp={r['best_per_target_wp']:.6f} "
              f"gain={r['gain']:+.6f} -> {r['verdict']}")
