"""notes/TASK.md step 4: the measurement that decides the TCN direction, run entirely on
cached full-validation prediction arrays (nothing is re-run).

usage: python3 tools/tcn_analysis.py <tcn_cache.npz> <label>

Reports for the given TCN cache:
  - full-validation wp_masked / wp_all_rows / ratio (uncalibrated) and a shared-c sweep
  - Pearson correlation of TCN vs GRU (02_trunk_global_full) vs calibrated baseline (x0.35),
    per target, on scored rows (same definition as tools/ensemble_test.py: unweighted
    np.corrcoef of the calibrated prediction streams)
  - best 3-way blend (shared weights and per-target weights, simplex grid step 0.1), plus
    the 2-way blends, against the best single model, with the >0.003 threshold check and
    the top of the weight surface as an over-tuning guard
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import weighted_pearson

ROOT = Path(__file__).resolve().parents[1]
BASELINE_C = 0.35  # previously chosen, conservative, larger-c point on the plateau
GRID = [1.0, 0.8, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2]
THRESHOLD = 0.003


def load(path):
    d = np.load(path)
    return d["preds"], d["targets"], d["need"], d["scored"]


def masked(preds, targets, mask):
    return (targets[..., 0][mask], targets[..., 1][mask], preds[..., 0][mask], preds[..., 1][mask])


def wp_mean(y0, y1, p0, p1):
    return 0.5 * (weighted_pearson(y0, p0) + weighted_pearson(y1, p1))


def main(tcn_cache: str, label: str):
    tcn_cache = ROOT / tcn_cache
    paths = {
        "baseline": ROOT / "tools/_baseline_raw_preds_valid.npz",
        "gru": ROOT / "tools/_trunk_global_full_raw_preds_valid.npz",
        "tcn": tcn_cache,
    }
    scale = {"baseline": BASELINE_C, "gru": 1.0, "tcn": 1.0}

    streams, all_rows = {}, {}
    y0 = y1 = None
    for name, path in paths.items():
        preds, targets, need, scored = load(path)
        m = need & scored
        a0, a1, p0, p1 = masked(preds, targets, m)
        if y0 is None:
            y0, y1 = a0, a1
        else:
            assert np.array_equal(y0, a0) and np.array_equal(y1, a1), f"{name}: target/mask mismatch vs first cache"
        streams[name] = (p0, p1)
        if name == "tcn":
            r0, r1, q0, q1 = masked(preds, targets, need)
            all_rows_tcn = (r0, r1, q0, q1)
        del preds, targets

    # ---- 1. TCN solo, full validation, calibration sweep
    p0, p1 = streams["tcn"]
    solo_masked = wp_mean(y0, y1, p0, p1)
    r0, r1, q0, q1 = all_rows_tcn
    solo_all = wp_mean(r0, r1, q0, q1)
    print(f"=== {label}: full 1,873-sequence validation (uncalibrated) ===")
    print(f"wp_masked={solo_masked:.6f} (t0={weighted_pearson(y0, p0):.6f} t1={weighted_pearson(y1, p1):.6f})  "
          f"wp_all_rows={solo_all:.6f}  ratio={solo_all/solo_masked:.3f}  (baseline 0.713, GRU 0.735)")
    print("\n--- shared multiplier sweep (TCN) ---")
    sweep = []
    for c in GRID:
        w = wp_mean(y0, y1, p0 * c, p1 * c)
        sweep.append((c, w))
        print(f"  c={c:<5} wp_masked={w:.6f}")
    best_c, best_w = max(sweep, key=lambda r: r[1])
    print(f"  best c={best_c} ({best_w:.6f}); uncalibrated (c=1.0) = {dict(sweep)[1.0]:.6f}; "
          f"gain from calibration = {best_w - dict(sweep)[1.0]:+.6f}")

    # calibrated streams for the ensemble (TCN uses its own best-shared c only if it helps
    # by more than the plateau noise; otherwise 1.0 like the GRU)
    tcn_c = best_c if (best_w - dict(sweep)[1.0]) > 0.001 else 1.0
    print(f"  -> using c={tcn_c} for the TCN in the blends below")
    cal = {n: (streams[n][0] * (tcn_c if n == "tcn" else scale[n]),
               streams[n][1] * (tcn_c if n == "tcn" else scale[n])) for n in streams}

    solo = {n: wp_mean(y0, y1, *cal[n]) for n in cal}
    print("\n=== solo scores after each model's own calibration ===")
    for n, v in solo.items():
        print(f"  {n:<9} {v:.6f}")
    best_single_name = max(solo, key=solo.get)
    best_single = solo[best_single_name]

    # ---- 2. correlation matrix per target on scored rows
    print("\n=== prediction-stream Pearson correlation (calibrated, scored rows) ===")
    names = ["baseline", "gru", "tcn"]
    for k, tgt in enumerate(("t0", "t1")):
        M = np.corrcoef(np.stack([cal[n][k] for n in names]))
        print(f"  {tgt}:            " + "  ".join(f"{n:>9}" for n in names))
        for i, n in enumerate(names):
            print(f"      {n:<9} " + "  ".join(f"{M[i, j]:9.4f}" for j in range(3)))

    # ---- 3. blends
    def blend(w, k):
        return sum(w[i] * cal[n][k] for i, n in enumerate(names))

    simplex = [w for w in itertools.product(np.arange(0, 1.01, 0.1), repeat=3) if abs(sum(w) - 1.0) < 1e-9]
    print(f"\n=== blends (simplex grid step 0.1, {len(simplex)} points); best single = {best_single_name} {best_single:.6f} ===")

    shared = sorted(((wp_mean(y0, y1, blend(w, 0), blend(w, 1)), w) for w in simplex), reverse=True)
    print("shared weights (baseline, gru, tcn) — top 5 (over-tuning guard: how flat is the top?):")
    for s, w in shared[:5]:
        print(f"   {s:.6f}  w=({w[0]:.1f},{w[1]:.1f},{w[2]:.1f})")

    per0 = max(((weighted_pearson(y0, blend(w, 0)), w) for w in simplex), key=lambda r: r[0])
    per1 = max(((weighted_pearson(y1, blend(w, 1)), w) for w in simplex), key=lambda r: r[0])
    per_target = 0.5 * (per0[0] + per1[0])
    print(f"per-target weights: t0 w=({per0[1][0]:.1f},{per0[1][1]:.1f},{per0[1][2]:.1f}) {per0[0]:.6f} | "
          f"t1 w=({per1[1][0]:.1f},{per1[1][1]:.1f},{per1[1][2]:.1f}) {per1[0]:.6f} -> mean {per_target:.6f}")

    def two_way(a, b):
        i, j = names.index(a), names.index(b)
        best = max(
            ((wp_mean(y0, y1, x * cal[a][0] + (1 - x) * cal[b][0], x * cal[a][1] + (1 - x) * cal[b][1]), x)
             for x in np.arange(0, 1.01, 0.1)), key=lambda r: r[0])
        return best
    print("\n2-way blends (best weight on first-named model):")
    for a, b in [("gru", "tcn"), ("baseline", "tcn"), ("baseline", "gru")]:
        s, x = two_way(a, b)
        print(f"   {a}+{b}: {s:.6f} at w_{a}={x:.1f}  ({s - best_single:+.6f} vs best single)")

    best_blend = max(shared[0][0], per_target)
    gain = best_blend - best_single
    print(f"\nbest 3-way blend = {best_blend:.6f}; best single model = {best_single_name} {best_single:.6f}; "
          f"gain = {gain:+.6f}  (threshold {THRESHOLD}) -> {'CLEARS' if gain > THRESHOLD else 'does NOT clear'} the bar")
    print(f"vs. calibrated baseline 0.672580: best blend {best_blend - 0.672580:+.6f}, TCN solo {solo['tcn'] - 0.672580:+.6f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
