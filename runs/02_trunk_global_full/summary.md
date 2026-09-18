# 02_trunk_global_full

**Config**: `--trunk-hidden 256 128 --loss-mode global --use-skip --skip-init ridge
--gru-hidden 128 --gru-layers 2 --batch-size 64 --chunk-len 2000 --epochs 4
--early-stop-patience 5 --max-seconds 5400 --valid-every-windows 200 --valid-sequences 400
--calibrate-sequences 400`. 260,452 params. Skip connection initialized from
`models/ridge_v1.npz`.

**Result (full 1,873-sequence validation)**: wp_masked=**0.667194**
(t0=0.663937, t1=0.670452), wp_all_rows=0.490706, ratio=0.735. `best.pt` and
`last.pt` are identical.

**Peaked at**: step 2200, when `--max-seconds 5400` (90 min) stopped it — the
run was still improving monotonically at that point (never went the 5
evaluations without improvement that would have triggered early stopping
instead). Not converged; more compute would likely improve it further.

**Wall clock**: ~90 minutes (5418s).

**What was learned**: initializing a linear skip connection from a
closed-form weighted-ridge fit, combined with correctly pooling the training
loss the same way Global WP pools its rows (no per-sequence renormalization),
fixes both problems that plagued every prior checkpoint (stages 2-7) at
once — mask-fitting (this run's ratio 0.735 vs. those runs' 0.24-0.50) and
output-scale calibration (multiplier=1.0 needed here vs. those runs never
having been checked, and the *provided baseline* needing a 0.35x correction).
This is the first checkpoint from this pipeline to beat the baseline
(0.617052) outright.
