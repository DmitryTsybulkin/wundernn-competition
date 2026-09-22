"""notes/TASK_gpu_prep.md step 4: how much does the 500-sequence in-run validation subset
(datasets/valid_subset.parquet, tools/make_valid_subset.py) differ from the full 1,873?

Everything except --run-model is pure numpy over the cached full-validation prediction arrays
(tools/_*_raw_preds_valid.npz), using per-sequence weighted moments so that any subset's pooled
Global WP is exact (identical to utils.weighted_pearson on the pooled rows) and cheap.

Reports, per model: full-valid wp_masked, the seed-42 subset's wp_masked (from the cache, i.e.
fp32 features), the offset; the spread of that offset over 2,000 random 500-sequence subsets
(what a *typical* subset would give, so the seed-42 draw can be judged); and the spread of the
PAIRED differences between models (what matters when comparing two training runs).
With --run-model: also runs runs/02_trunk_global_full/best.pt on the actual float16
valid_subset.parquet, to separate the sampling offset from any float16 quantisation effect.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils import weighted_pearson

MODELS = {  # name -> (cache, output scale)
    "baseline x0.35": ("tools/_baseline_raw_preds_valid.npz", 0.35),
    "GRU 02_trunk_global_full": ("tools/_trunk_global_full_raw_preds_valid.npz", 1.0),
    "TCN rf63": ("tools/_tcn_rf63_raw_preds_valid.npz", 1.0),
    "TCN rf255": ("tools/_tcn_rf255_raw_preds_valid.npz", 1.0),
}


def seq_moments(preds, targets, need, scored, c):
    n = preds.shape[0]
    S = np.zeros((n, 2, 6))  # per sequence, per target: W, Swy, Swp, Swyy, Swpp, Swyp
    for i in range(n):
        m = need[i] & scored[i]
        y = np.clip(targets[i][m], -2, 2).astype(np.float64)
        p = np.clip(preds[i][m] * c, -2, 2).astype(np.float64)
        for k in range(2):
            w = np.abs(y[:, k])
            S[i, k] = [w.sum(), (w * y[:, k]).sum(), (w * p[:, k]).sum(),
                       (w * y[:, k] ** 2).sum(), (w * p[:, k] ** 2).sum(), (w * y[:, k] * p[:, k]).sum()]
    return S


def pooled(S):
    """S: (n_seq, 2, 6) -> mean over targets of the pooled weighted Pearson."""
    t = S.sum(axis=0)
    out = []
    for k in range(2):
        W, wy, wp, wyy, wpp, wyp = t[k]
        my, mp = wy / W, wp / W
        cov, vy, vp = wyp / W - my * mp, wyy / W - my * my, wpp / W - mp * mp
        out.append(cov / np.sqrt(vy * vp))
    return 0.5 * (out[0] + out[1])


def main(run_model: bool):
    idx = np.array(json.load(open(ROOT / "datasets/valid_subset_indices.json"))["row_groups"])
    moments = {}
    for name, (path, c) in MODELS.items():
        d = np.load(ROOT / path)
        moments[name] = seq_moments(d["preds"], d["targets"], d["need"], d["scored"], c)
        del d
    n_full = len(next(iter(moments.values())))
    # sanity: the moment formula must reproduce the official pooled number
    d = np.load(ROOT / MODELS["GRU 02_trunk_global_full"][0])
    m = d["need"] & d["scored"]
    ref = 0.5 * sum(weighted_pearson(d["targets"][..., k][m], d["preds"][..., k][m]) for k in range(2))
    print(f"sanity: moment-based full GRU {pooled(moments['GRU 02_trunk_global_full']):.6f} vs utils.weighted_pearson {ref:.6f}")
    del d, m

    rng = np.random.default_rng(0)
    draws = [rng.choice(n_full, size=len(idx), replace=False) for _ in range(2000)]
    print(f"\n{'model':<28}{'full 1873':>11}{'subset 500':>12}{'offset':>10}   random-500 offset: mean / sd")
    full, sub, rand = {}, {}, {}
    for name, S in moments.items():
        full[name], sub[name] = pooled(S), pooled(S[idx])
        rand[name] = np.array([pooled(S[d]) for d in draws])
        off = rand[name] - full[name]
        print(f"{name:<28}{full[name]:>11.6f}{sub[name]:>12.6f}{sub[name]-full[name]:>+10.4f}   {off.mean():+.4f} / {off.std():.4f}")
    print("\npaired differences (what a monitoring comparison between two models sees):")
    print(f"{'pair':<52}{'full':>9}{'subset':>9}{'diff':>9}   random-500 paired-diff sd")
    for a, b in [("GRU 02_trunk_global_full", "baseline x0.35"), ("GRU 02_trunk_global_full", "TCN rf63"),
                 ("TCN rf63", "TCN rf255")]:
        f, s = full[a] - full[b], sub[a] - sub[b]
        sd = (rand[a] - rand[b]).std()
        print(f"{a + ' - ' + b:<52}{f:>+9.4f}{s:>+9.4f}{s-f:>+9.4f}   {sd:.4f}")

    if run_model:
        import torch
        from train.eval import evaluate_fast
        from train.model import SequenceModel
        ckpt = torch.load(ROOT / "runs/02_trunk_global_full/best.pt", map_location="cpu", weights_only=False)
        model = SequenceModel(ckpt["model_cfg"])
        model.load_state_dict(ckpt["state_dict"])
        r = evaluate_fast(model, ROOT / "datasets/valid_subset.parquet", torch.device("cpu"), max_sequences=None, batch_size=32)
        print(f"\nrun on the actual float16 valid_subset.parquet: wp_masked={r['wp_masked']['weighted_pearson']:.6f} "
              f"(t0={r['wp_masked']['t0']:.6f} t1={r['wp_masked']['t1']:.6f}) wp_all_rows={r['wp_all_rows']['weighted_pearson']:.6f}"
              f"   [cache-derived subset value {sub['GRU 02_trunk_global_full']:.6f}]")


if __name__ == "__main__":
    main("--run-model" in sys.argv)
