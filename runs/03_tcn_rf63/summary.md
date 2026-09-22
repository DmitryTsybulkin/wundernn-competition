# 03_tcn_rf63

**Config**: `--arch tcn --trunk-hidden 256 128 --tcn-channels 128 --tcn-kernel-size 3
--tcn-layers 5 --loss-mode global --use-skip --skip-init ridge --batch-size 64
--chunk-len 2000 --early-stop-patience 5 --max-seconds 3600 --valid-every-windows 200
--valid-sequences 400`. Receptive field 63 rows, 309,988 params, dilations 1-16,
LayerNorm (per-timestep) + residual, state = ring buffer of the last 62 trunk outputs.

**Result (full 1,873-sequence validation)**: last.pt wp_masked=**0.655782**
(t0=0.650189, t1=0.661375), wp_all_rows=0.475998, ratio=0.726. best.pt 0.654504 /
0.475833 / 0.727. No calibration needed (multiplier=1.0; c<1 strictly worse).

**Peaked at**: not converged — step 1600, when the 60-min cap stopped it; it improved
at every one of the 8 evaluations (400-seq sample 0.6280 -> 0.6672).

**Wall clock**: 60 min (3492 s to the last eval). ~437 s per 200 steps incl. a
400-sequence eval, vs. ~480 s for the GRU (`02_trunk_global_full`): ~9% faster.

**What was learned**: a 63-row receptive field with the same trunk + ridge-init skip +
global-loss recipe lands 0.011 below the GRU on the full set (0.6558 vs 0.6672), and its
predictions correlate 0.97 with the GRU's — a weaker copy of it, not a different signal.
