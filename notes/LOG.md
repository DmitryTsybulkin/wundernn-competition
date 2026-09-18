# LOG

Append-only. Newest entries at the bottom. One block per run or investigation.

---

## 2026-09-14 — baseline reproduced
`python3 baseline/solution.py --validation datasets/valid.parquet`
per-block WP 0.5895945 (t0 0.5715, t1 0.6077), 1872/1873 eligible blocks,
3,528,273 scored rows of 37,274,573 required predictions.
Matches the documented 0.589595 exactly.

## 2026-09-14 — data exploration + linear probes (Cowork session)
Weighted ridge on 60 train sequences, evaluated on 30 valid sequences,
old per-block metric:
- raw 112 features only: 0.5426
- + EWMA(4/16/64/256) levels and deviations: 0.5333
- EWMA/deviation features only: 0.5334
- i0 block only: 0.5479
- 1024 random ReLU features, unstandardized: 0.4988; standardized: 0.4728
- ridge + causal EMA smoothing of predictions: hl=2 0.5455, hl=4 0.5282,
  hl=8 0.4926, hl=32 0.3566 (monotone decay)
Conclusion: signal is essentially instantaneous; recurrence worth ~0.05.

## 2026-09-14/15 — training pipeline, stages 2-7
Pipeline built under `train/`. All runs on CPU.
Full-validation, old per-block metric:
- stage5_ema_diag 0.5716 (t0 0.5691, t1 0.5741)
- stage7_ema_cosine 0.5699 (t0 0.5686, t1 0.5712)
- stage6_full_ema 0.5699 (t0 0.5686, t1 0.5712)
None beat the 0.589595 baseline.
Diagnosis: all runs report 192,258 params = baseline architecture, MLP trunk
never engaged. Severe overfitting after ~600 steps; stage7 valid peaks 0.5828
at step 600 then decays to 0.5245 by step 1650 while train loss improves.
Decay rate tracks the LR schedule, so it is genuine drift, not a frozen-model bug.

## 2026-09-16 — metric change to Global WP
Organizers replaced the metric. Baseline restated as 0.617052 (same ONNX,
same validation data — only aggregation changed).
Provisional Global WP implemented in the Cowork session and validated on the
ridge model, 60 valid sequences:
- ridge per-block 0.5443 -> global 0.5735 (+0.029)
- baseline per-block 0.5896 -> global 0.6170 (+0.027)
Matching shift confirmed the formula; the official `METRIC.md` later confirmed
it exactly.
Key finding: per-sequence z-scoring of predictions
- per-block 0.5443 -> 0.5468 (slightly positive)
- global   0.5735 -> 0.5249 (**-0.049**)
Sequence weight spread under Global WP: 49x across 60 sequences; top-10 carry 37%.

## 2026-09-16 — updated starterpack diffed
`wnn_connectome_updated` vs `wnn_connectome_starterpack`:
- `baseline.onnx` identical (md5 13fba53e0d9047def8cef58b87feb6db)
- `baseline/solution.py` identical
- all three parquet files identical in size — data unchanged
- **changed: `utils.py`** (BlockAccumulator -> GlobalAccumulator, adds
  `METRIC_SPECIFICATION = 'global_wp'`), **`METRIC.md`** (rewritten),
  README and docs (wording)
- new in FAQ: test set is 1,970 sequences = 39,400,000 rows (=> ~91 us/row);
  explicit warning that the hidden test mask may differ from the visible
  validation mask, and that participants may build their own splits/masks.

## 2026-09-16 — Global WP migration, steps 1/2/3/4/6/7/8 (step 5 paused by request)
Executing `notes/TASK.md`. No new training run this session (measurement/correctness
pass only), per task instructions.

**Step 1 — verify.** `utils.py` confirmed to define `GlobalAccumulator` and
`METRIC_SPECIFICATION = 'global_wp'`. Cleared stale `__pycache__` (root, `train/`,
two `runs/*/__pycache__`).

**Step 2 — baseline reproduction (gate).**
`python3 baseline/solution.py --validation datasets/valid.parquet`
-> `weighted_pearson=0.6170522889725805` (t0 0.5879125171273903, t1
0.6461920608177706), 1873/1873 blocks with scored rows, 3,528,273 selected rows.
Matches the required 0.617052 to 6 decimals. **GATE PASSED.** Took ~16 minutes
wall clock (single ORT thread, 37.46M sequential `.run()` calls, no interim
output — `baseline/solution.py` only prints once at the end, so a quiet
terminal for that long is expected, not a hang).

**Step 3 — repoint the training-side evaluator.**
`train/eval.py`'s `evaluate_fast` now feeds each sequence directly into
`utils.GlobalAccumulator` (the same class `ScorerStepByStep` uses) instead of
the old per-sequence-averaged block scorer. Added
`test_global_accumulator_matches_one_shot_reference` to `train/selftest.py`:
runs the baseline ONNX row-by-row over 50 validation sequences, feeds the same
predictions through (a) `GlobalAccumulator`'s incremental merge and (b) the
direct one-shot `global_wp()` formula from `METRIC.md` on the concatenated
rows. Result: t0 0.646176 vs 0.646176, t1 0.666573 vs 0.666573 — agree to
better than 1e-6. `train/export_onnx.py`'s PyTorch-vs-ONNX verification was
also repointed to `GlobalAccumulator` (it previously used the old per-sequence
eligibility+averaging logic, which no longer matches `_read_valid_sequence`'s
new return shape).

**Step 4 — dual reporting.** `evaluate_fast` now returns
`{'wp_masked': {...}, 'wp_all_rows': {...}}`, both full `GlobalAccumulator`
result dicts (`wp_masked` = official `need_prediction AND is_scored`,
`wp_all_rows` = `need_prediction` only). `train.py` logs both to wandb
(`valid/wp_masked*`, `valid/wp_all_rows*`) every eval; best-checkpoint
selection uses `wp_masked` only, per the hard rule in `CLAUDE.md`.

**Step 5 — re-measure the six checkpoints — PAUSED, not started.** User
asked to stop here while changing location. Resume by evaluating
`stage5_ema_diag`, `stage6_full_ema`, `stage7_ema_cosine` (`best.pt` and
`last.pt` each) with the new `evaluate_fast`, on the full 1873-sequence
validation set, and writing the resulting table here.

**Step 6 — fix the loss.** Added `global_wp_loss` to `train/losses.py`: pools
ALL rows in a training batch (every sequence and timestep together) into ONE
weighted correlation per target, matching `GlobalAccumulator`'s dataset-global
pooling — no per-sequence renormalization or equalization. The old
`chunk_wp_loss` (per-batch-item correlation then averaged, which implicitly
renormalizes/equalizes each sequence — exactly what the -0.049 WP per-sequence
z-scoring regression measured) is kept behind `--loss-mode per_chunk` for
comparison; `--loss-mode global` (new `global_wp_loss`) is now the default.
New self-tests: pooling correctness vs. `utils.weighted_pearson` on flattened
arrays (exact match), and a NaN-gradient regression check on degenerate
(all-zero-mask, all-one-mask) batches (finite gradients confirmed).

**Step 7 — the trunk bug, root-caused and fixed.** `model.py`'s `Trunk` class
was never actually broken — it correctly builds a real MLP whenever
`trunk_hidden` is non-empty. The real cause: `train.py`'s `--trunk-hidden`
used `argparse nargs="*"`, and every training invocation this pipeline has
launched wrote `--trunk-hidden \` immediately followed by another flag on the
next line (e.g. `--trunk-hidden \` then `--lr 1e-3`) — argparse legally parses
that as zero values and silently defaults to no trunk. So every one of
stages 2-7 built the baseline's exact no-trunk architecture (192,258 params)
without erroring; the trunk hypothesis was never actually tested at scale
(only in an isolated tiny smoke test that happened to pass a value directly).
Fixed two ways: (a) `nargs="*"` -> `nargs="+"`, so that exact CLI pattern is
now a hard parse error (`error: argument --trunk-hidden: expected at least one
argument`) instead of a silent misparse; (b) added an explicit startup
assertion (raises if `--trunk-hidden` is non-empty but the built model has
exactly 192,258 params) plus printing the full resolved architecture
(`trunk_hidden=`, `gru_hidden=`, `gru_layers=`, `reparam=`) at every startup.
Verified: `trunk_hidden=(256,128)` builds 260,226 params (vs 192,258 with no
trunk); the historical footgun pattern now raises immediately instead of
building silently.

**Step 8 — MPS vs CPU benchmark (`train/bench_mps.py`, no real training run).**
20 real training steps per device, identical config/data/seed (baseline
architecture: gru_hidden=128, gru_layers=2, batch=64, chunk_len=2000,
`global_wp_loss`):
- CPU: 1590.6 ms/step (mean, excl. 2 warmup steps)
- MPS: 2468.3 ms/step (mean, excl. 2 warmup steps)
- **CPU is 1.55x faster than MPS** for this model size on this machine.
- No `PYTORCH_ENABLE_MPS_FALLBACK` warnings (stderr checked, empty) — not a
  fallback artifact, MPS is just slower here (small-recurrent-net kernel-launch
  overhead dominates, as found earlier this project with the old loss too).
- Loss values identical between devices at every matching step (max diff
  0.0000) — confirms no numerical/correctness bug, just a speed difference.
- Conclusion: MPS is a dead end for training this architecture on this Mac.
  If training speed remains the binding constraint (defect 5: ~0.4% of train
  data seen per run), the next thing to benchmark would be a rented CUDA GPU,
  not MPS.

## 2026-09-16 — step 5: all six checkpoints re-measured under Global WP
Full 1873-sequence validation set, `train/eval.py`'s `evaluate_fast` (now
`GlobalAccumulator`-backed). All six checkpoints predate the step 6/7 fixes
(trained under the old `chunk_wp_loss` and, per the trunk bug, all six
share the identical 192,258-param no-trunk architecture regardless of run).

| checkpoint | wp_masked | wp_masked t0/t1 | wp_all_rows | wp_all_rows t0/t1 | old per-block WP (metadata) |
|---|---:|---|---:|---|---:|
| stage5_ema_diag/best.pt | **0.477417** | 0.416430 / 0.538404 | 0.236610 | 0.171542 / 0.301679 | 0.584785 |
| stage5_ema_diag/last.pt | 0.262700 | 0.166794 / 0.358605 | 0.077048 | -0.011365 / 0.165461 | 0.584785 |
| stage6_full_ema/best.pt | 0.457501 | 0.354938 / 0.560064 | 0.220781 | 0.115982 / 0.325580 | 0.582431 |
| stage6_full_ema/last.pt | 0.099061 | 0.075086 / 0.123036 | 0.023303 | 0.007061 / 0.039546 | 0.582431 |
| stage7_ema_cosine/best.pt | 0.459650 | 0.358816 / 0.560485 | 0.224309 | 0.122392 / 0.326225 | 0.582806 |
| stage7_ema_cosine/last.pt | 0.377182 | 0.270685 / 0.483678 | 0.143181 | 0.040628 / 0.245733 | 0.582806 |

**Best checkpoint under Global WP: `stage5_ema_diag/best.pt`, 0.477417** —
same relative ranking as under the old per-block metric (stage5 was highest
there too), but the absolute gap to the baseline is far larger than the old
metric suggested: **0.617052 - 0.477417 = 0.139635**, vs. an apparent gap of
only ~0.018 under the retired per-block metric. Global WP evidently weights
this dataset very differently (STATUS.md already notes a 49x sequence-weight
spread under Global WP) — the old metric was flattering these checkpoints.

**Systematic finding: every checkpoint's wp_all_rows is roughly half its
wp_masked** (ratios 0.42-0.49 across all six, tightest for the two `best.pt`
that share stage6/stage7's architecture: 0.482 and 0.488). Per the CLAUDE.md
hard rule this divergence means a model is "fitting the visible mask's
character rather than the signal" — but before concluding these six
checkpoints are mask-overfit specifically, the baseline's own wp_masked vs.
wp_all_rows needs measuring for a control comparison (not yet available when
this entry was written; `python3 baseline/solution.py` only reports the
masked/official number). Queued as a follow-up — if the baseline shows a
similarly large gap, the divergence is a property of how `is_scored` selects
rows in this dataset, not a defect unique to our models.

**Also confirms**: `last.pt` is dramatically worse than `best.pt` in every
run (most severely stage6: 0.099 vs 0.458), consistent with the severe
post-peak decay documented in the 2026-09-14/15 log entry — the
best-checkpoint mechanism is doing real, necessary work.

**Recommendation**: none of these six checkpoints are worth deploying or
building on further as-is — they predate both the loss fix (step 6) and the
first real test of the trunk hypothesis (step 7 unblocked it but no trunk-
equipped run has been trained yet). A fresh short run with
`--loss-mode global` (default) and a real `--trunk-hidden` is the logical
next step, evaluated with the now-correct dual-metric evaluator from the
start rather than retrofitted after the fact.

## 2026-09-16/17 — baseline wp_masked vs wp_all_rows (control for the divergence finding above)
Row-by-row over the full 1873-sequence validation set (same ONNX baseline as
the step-2 gate, ~56 min wall clock — this is the row-by-row ONNX path, not
the fast batched torch evaluator):
- `wp_masked` (official) = 0.6170522889725805 (matches step 2 exactly)
- `wp_all_rows` = 0.4401560562906397 (t0 0.423152, t1 0.457160)
- **ratio wp_all_rows/wp_masked = 71.3%**

This resolves the question left open in the step-5 entry above. The baseline
*does* show some divergence — confirming a gap is partly an inherent property
of how `is_scored` selects rows in this dataset, not automatically a defect —
but 71.3% is well above every one of our six checkpoints' ratios (24-50%,
tightest for the three `best.pt`: 48-50%). So our checkpoints are
genuinely more mask-dependent than the baseline, not merely inheriting the
dataset's baseline level of divergence. This is a real, separate weakness on
top of the untested trunk/loss fixes — worth keeping in mind when the next
training run is evaluated: watch wp_all_rows/wp_masked and expect something
closer to baseline's ~70%, not our previous ~25-50%.

## 2026-09-17 — Cowork session: output clipping discovered (the big one)
Ran the baseline ONNX row-by-row over 30 validation sequences (sampled, seed 2)
and inspected its prediction distribution:
- fraction of |pred| > 2: **38.3% (t0) / 36.2% (t1)**; pred std 2.48/2.59; max 9.2
- fraction of |target| > 2: 3.5%

Global WP is invariant to a global affine transform of predictions ONLY before
the metric's clip to [-2,2]. With 38% of predictions clipped, all ordering
information among strong signals is destroyed — and those are exactly the rows
carrying the largest metric weights.

Multiplying every prediction by a constant, same 30 sequences:
| multiplier | Global WP |
|---|---:|
| 1.0 | 0.6812 |
| 0.8 | 0.6912 |
| 0.6 | 0.7050 |
| 0.5 | 0.7110 |
| **0.4** | **0.7137** |
| 0.3 | 0.7120 |
| 0.25 | 0.7103 |
| 0.2 | 0.7101 |

**+0.033 from one constant, no training.** Note this 30-sequence subset is
optimistically biased (baseline reads 0.6812 here vs 0.617052 on the full set),
so the magnitude needs confirming on all 1,873 sequences — but the mechanism is
not in doubt.

### Ensemble idea killed by the same experiment
Blending the baseline with a ridge (both globally z-scored first) appeared to
give a large gain: 0.7123 at 0.8*baseline + 0.2*ridge. But pure baseline, z-scored,
scores 0.7138 — identical to simply scaling by 0.4. The apparent ensemble gain
was entirely the rescaling. At the optimum the ridge contributes ~+0.0004.
Do not pursue baseline+ridge ensembling.

### Per-sequence scale calibration also killed
Tested whether the checkpoints' Global WP collapse could be recovered by
rescaling each sequence's predictions:
- ridge as-is: global 0.5754 / per-block 0.5345
- per-seq z-scored: global 0.5266
- per-seq z-scored x ORACLE true mean|y| of that sequence: global 0.5278
- per-seq z-scored x scale predicted from features: global 0.5129
Even with oracle knowledge of each sequence's target magnitude, nothing is
recovered. A single globally-calibrated linear map is already optimal in this
respect; the per-chunk-loss checkpoints' damage is not a per-sequence scalar.
Also: a linear model predicting per-sequence mean|y| from features achieves only
corr 0.108 with the truth — sequence-level target magnitude is barely predictable.

### Ridge vs the trained checkpoints
On the same 30-sequence sample: ridge (60 training sequences, closed form,
~30 seconds) scored global 0.6054 vs baseline 0.6812. All six trained
checkpoints scored 0.477 or below on the full set. A memoryless linear model
fitted on 0.6% of train currently beats every neural checkpoint we have.

## 2026-09-17 — step 3: ridge_v1, a proper reference model (`tools/fit_ridge.py`)
Closed-form weighted ridge, streamed, no SGD. Per-row weight
`w = 0.5*(|clip(t0,-2,2)| + |clip(t1,-2,2)|)` (shared between both targets'
normal equations, per the task's literal wording), need_prediction rows only,
every 4th row subsampled, bias column included (unpenalized).

- Fit on 800 TRAIN sequences (3,980,800 rows after subsampling), penalty swept
  over 8 log-spaced values (1e-2 to 1e5) on 200 held-out TRAIN sequences —
  the problem is heavily overdetermined (113 features vs. millions of rows),
  so the sweep is nearly flat until lambda >= 1e4: 0.330693 (lambda=0.01) down
  to 0.319271 (lambda=1e5). **Chosen lambda=0.01** (smallest tested — the
  data wants almost no regularization here).
- Refit on the full 1,000 sequences at that lambda, saved to
  `models/ridge_v1.npz` (coefficients, lambda, feature/target column order,
  weight formula, stride — everything needed to reproduce or apply it).
- **Full 1,873-sequence validation score: wp_masked=0.586440, wp_all_rows=0.402877
  at c=1.0 (uncalibrated).** This already beats every one of our six trained
  GRU checkpoints (best 0.477417) by a wide margin, and sits only 0.0306
  below the baseline (0.617052).
- **Unlike the baseline, calibration does NOT help the ridge** — c=1.0 is the
  best multiplier tested; every c<1.0 makes it monotonically worse (down to
  0.576767 at c=0.2). Sensible: a least-squares fit is already scaled to
  match the target's natural range, unlike whatever objective produced the
  baseline's badly-oversized outputs.
- **wp_all_rows/wp_masked ratio = 68.7%** (0.402877/0.586440) — close to the
  baseline's 71.3%, and far above any of our six checkpoints' 24-50%. The
  ridge is not mask-fitting either. Combined with beating every checkpoint
  outright, this strongly supports TASK.md step 5's plan to give the next
  training run a ridge-initialized linear skip connection: the ridge alone
  is both stronger and better-calibrated than anything the GRU pipeline has
  produced so far.

## 2026-09-17 — step 1: output-clipping effect CONFIRMED on the full validation set
`tools/calibrate.py`: baseline's raw predictions computed once over all 1,873
validation sequences (row-by-row ONNX, ~21 min), cached to
`tools/_baseline_raw_preds_valid.npz` (674 MB), then scored at 10 multipliers
via the official `GlobalAccumulator`.

| c | wp_masked | wp_all_rows | frac(&#124;c·pred&#124;>2, scored) |
|---:|---:|---:|---:|
| 1.0 (sanity gate) | 0.617052 | 0.440156 | 76.2% |
| 0.8 | 0.633670 | 0.457143 | 58.4% |
| 0.6 | 0.653650 | 0.477834 | 43.7% |
| 0.5 | 0.663161 | 0.488624 | 36.2% |
| 0.45 | 0.667206 | 0.493580 | 31.2% |
| 0.4 | 0.670452 | 0.497802 | 24.4% |
| 0.35 | 0.672580 | 0.500796 | 15.4% |
| **0.3 (peak)** | **0.673295** | 0.502117 | 5.6% |
| 0.25 | 0.672762 | 0.502088 | 0.4% |
| 0.2 | 0.672520 | 0.501875 | 0.0% |

c=1.0 reproduces 0.6170522889725805 exactly — sanity gate passed. **The effect
is real and considerably LARGER on the full set than the 30-sequence sample
suggested: +0.056 at the peak (0.673295 at c=0.3) vs. the sample's +0.033
estimate.** Also notable: 76.2% of *scored* predictions exceed |2| at c=1.0,
well above the 38% found across the (mostly-unscored) sample in the prior
entry — the clipping problem is worse specifically on the rows that carry
metric weight.

The optimum is a flat plateau (0.2-0.35 all within 0.0008 of the peak).
**Per TASK.md's explicit guidance, chose the larger c on the plateau for
robustness to a hidden test mask that may differ in character: c=0.35**
(0.672580 wp_masked — 0.0007 below the literal peak, but clips only 15.4% of
scored predictions vs. the peak's 5.6%, leaving more headroom if the test
mask's prediction distribution runs hotter than validation's).

## 2026-09-17 — step 2: calibrated-baseline submission candidate packaged
`python3 tools/calibrate.py --package 0.35` (reused the cached predictions
from step 1 — packaging itself took seconds, only the determinism/throughput
check re-runs the model). `baseline/solution.py`/`baseline.onnx` untouched;
the calibrated copy lives at `submissions/baseline_calibrated/` (also zipped
to `submissions/baseline_calibrated.zip`, 0.72 MB, well under the 20 MB limit)
— generated by text-substituting the `return` line of the pristine baseline's
`solution.py`, multiplying its output by 0.35.

- **Determinism**: identical predictions across two independent runs (200,000 rows). OK.
- **Throughput, first measurement: 301.18 us/row (projected 197.8 min — 3.3x
  over the 60-min/91-us-row budget).** This reading was alarming enough to
  investigate immediately rather than report as-is, especially since it
  would imply even the pristine baseline might miss the new tighter budget.
  Isolated re-measurement (raw `session.run()`: 29.5 us/row; the full
  `PredictionModel.predict(DataPoint(...))` path: 32.7 us/row) and a clean
  re-run of the exact same `verify_determinism_and_throughput` function
  immediately after (33.95 us/row, 22.3 min projected) both confirm the
  model is comfortably within budget. **The 301 us/row reading was a
  transient system-level hiccup**, not a code bug or a real performance
  problem — same class of false alarm as the 10x mid-run slowdown seen in
  `stage7_ema_cosine` earlier this project (background OS activity /
  thermal throttling, not anything in this codebase). **Confirmed throughput:
  ~34 us/row, ~22 min for the full 1,970-sequence test set.**
- **This is a legitimate submission candidate beating the provided baseline:
  0.672580 wp_masked vs. 0.617052 — not submitted, reported here for Dmitry
  to decide.**

## 2026-09-17 — step 5: 01_trunk_global — first checkpoint ever to beat the baseline
`runs/01_trunk_global`: `--trunk-hidden 256 128 --loss-mode global --use-skip
--skip-init ridge --gru-hidden 128 --gru-layers 2 --batch-size 64
--chunk-len 2000 --early-stop-patience 5 --max-seconds 5400
--valid-every-windows 200 --valid-sequences 400`. 260,452 params.

Training curve (400-sequence sample, wp_masked / wp_all_rows / ratio):
| step | wp_masked | wp_all_rows | ratio | elapsed |
|---:|---:|---:|---:|---:|
| 200 | 0.6245 | 0.4510 | 0.722 | 494s |
| 400 | 0.6435 | 0.4703 | 0.731 | 977s |
| 600 | 0.6573 | 0.4773 | 0.726 | 1466s |
| 800 | 0.6638 | 0.4825 | 0.727 | 1940s |
| 1000 | 0.6674 | 0.4877 | 0.731 | 2413s |
| 1200 | 0.6711 | 0.4939 | 0.736 | 2889s |
| 1400 | 0.6727 | 0.4953 | 0.736 | 3372s |
| 1600 | 0.6755 | 0.5001 | 0.740 | 3845s |

Beat the uncalibrated baseline (0.617052) from the very first evaluation
(step 200), unlike every prior checkpoint — and the mask ratio stayed
healthy throughout (0.72-0.74, close to the baseline's 0.713), unlike
stages 2-7's 24-50%. The ridge-initialized skip connection plus the
correctly-pooled global loss appear to have fixed both problems at once.

**Orchestration mistake**: `--epochs` was never passed and defaulted to 1.
`windows/epoch` = 1650, so the run simply ran out of data at step ~1650 and
exited normally (`done.`) — NOT via early stopping (never 5 evals without
improvement — it was still improving every single eval) or the 90-minute
`--max-seconds` budget (only 64 min elapsed). The model was still climbing
when cut off. `last.pt` (the very end of the epoch, step 1650) scored even
higher than `best.pt` (step 1600) on the 400-seq sample: 0.6772 vs 0.6755 —
confirms it hadn't peaked yet.

**Full 1,873-sequence validation** (the number that actually matters):
| checkpoint | wp_masked | wp_masked t0/t1 | wp_all_rows | ratio |
|---|---:|---|---:|---:|
| best.pt | 0.663142 | 0.659487 / 0.666798 | 0.484664 | 0.731 |
| last.pt | **0.664832** | 0.660792 / 0.668872 | 0.485578 | 0.730 |

**Both beat the uncalibrated baseline (0.617052) by +0.046 to +0.048** — the
first time any checkpoint from this pipeline has beaten it. Both sit
0.008-0.009 below the calibrated-baseline candidate (0.672580) — close, not
yet ahead. Neither needed post-training calibration (multiplier=1.0 for
both — the ridge-initialized skip connection kept the model well-scaled
throughout).

**Immediately relaunched as `runs/02_trunk_global_full`** with `--epochs 4`
added (everything else identical, same seed) so early stopping / max_seconds
get a real chance to govern the stop point this time, since the training
curve above shows clear remaining headroom.

## 2026-09-17 — step 5 concluded: 02_trunk_global_full
Same config as `01_trunk_global` with `--epochs 4` added (fixes the mistake
above). Curve tracked `01_trunk_global` exactly through step 1600 (identical
seed/data), as expected, then continued into new territory:

| step | wp_masked | wp_all_rows | ratio | elapsed |
|---:|---:|---:|---:|---:|
| 200-1600 | (identical to `01_trunk_global`, see prior entry) | | | |
| 1800 | 0.6773 | 0.4999 | 0.738 | 4450s |
| 2000 | 0.6784 | 0.5042 | 0.743 | 4940s |
| **2200** | **0.6790** | 0.5062 | 0.746 | 5418s |

Stopped via `--max-seconds 5400` exactly as intended this time (not early
stopping — it never went 5 evals without improving; it was monotonically
climbing the entire run). `best.pt` and `last.pt` are identical (step 2200
was the final step). No calibration needed (multiplier=1.0, same as
`01_trunk_global` — the ridge-initialized skip connection keeps the model
well-scaled throughout training, not just at the end).

**Full 1,873-sequence validation: wp_masked=0.667194 (t0=0.663937,
t1=0.670452), wp_all_rows=0.490706, ratio=0.735.**

Comparison:
| model | wp_masked (full valid) |
|---|---:|
| baseline, calibrated x0.35 | 0.672580 |
| **02_trunk_global_full** | **0.667194** |
| 01_trunk_global/last.pt | 0.664832 |
| provided baseline, uncalibrated | 0.617052 |
| ridge_v1 | 0.586440 |

+0.0024 over `01_trunk_global` from the extra 600 steps/26 min of training —
a small, real gain, not noise (both measured on the full validation set).
+0.050 over the uncalibrated baseline. Still 0.0054 below the calibrated
baseline — close but not yet ahead. **The run was still improving
monotonically when the 90-minute budget cut it off** — no sign of having
converged or peaked, unlike every one of stages 2-7 which crashed into
overfitting/instability well before any time budget was relevant. This is
the clearest evidence yet that the ridge-initialized skip connection plus
the correctly-pooled global loss fixed the fundamental problem, not just
patched around it: given more compute, this run would likely keep improving.

**Conclusion for `notes/TASK.md` step 5**: trunk + global loss + ridge-init
skip connection is a real, working recipe — first checkpoints from this
pipeline to ever beat the baseline, with a healthy (baseline-like) mask
ratio and no calibration debt. It has not yet beaten the calibrated-baseline
candidate, but was still climbing, not plateaued, when cut off by the time
budget alone.

## 2026-09-18 — FIRST SUBMISSION: baseline_calibrated (c=0.35)
Leaderboard: **0.6501**. Local full-validation wp_masked: **0.672580**.
Gap: **-0.0225**. Platform run: 28m58s, 624 MB RAM (our projection was ~22 min;
their CPU is somewhat slower, no truncation, well inside the 60-min budget).

### Cowork-side investigation of the gap
Baseline ONNX run over 50 validation sequences, predictions scaled by 0.35,
then re-scored under different row-selection schemes and resampling:

| scoring scheme | Global WP | rows |
|---|---:|---:|
| official `is_scored` mask | 0.6893 | 94,078 |
| 50% random subsample of that mask (3 draws) | 0.6926 / 0.6895 / 0.6886 | ~47,000 |
| mask dilated +/-25 rows | 0.5632 | 472,181 |
| mask dilated +/-100 rows | 0.5162 | 769,216 |
| mask dilated +/-400 rows | 0.5033 | 966,229 |
| random rows, same count per sequence | 0.5250 | 94,078 |
| all `need_prediction` rows | 0.4988 | — |

Bootstrap over which sequences are included (200 resamples of the 50):
mean 0.6894, **sd 0.0232**, 95% interval 0.6412-0.7371. Scaled to a
1,873-sequence set that sequence-composition sd is ~0.0038.

### Conclusions
1. **Not sampling noise within the mask.** Halving the scored rows moves the
   score by <0.004. Row-level noise cannot produce -0.0225.
2. **Not overfitting.** The only quantity we fitted for this submission is a
   single scalar (c=0.35) on a flat plateau; the model itself is the
   organizers' own untouched ONNX. There is essentially nothing to overfit.
3. **The score is extremely sensitive to WHICH rows are scored.** The official
   mask scores 0.689 while random rows of the same count score 0.525 and all
   rows 0.499 — the mask concentrates on moments where the model is far more
   accurate. Dilating it by only +/-25 rows already costs 0.126. A test mask
   whose character differs even slightly from validation's easily accounts for
   -0.0225.
4. Sequence composition alone (a different draw of ~1,970 sequences, possibly
   from a different period/regime) contributes on the order of +/-0.004 of
   noise, plus any systematic regime difference.
5. The public leaderboard is computed on a *sample* of the test data
   (`docs/rules.md`), adding further variance not present in our local number.

**Verdict: the gap is a dataset/mask difference, not a defect.** Expect local
wp_masked to read roughly 0.02 high relative to the leaderboard, and treat
local differences smaller than ~0.005 as unresolvable.

### Decisive follow-up (costs one submission)
Submit the organizers' pristine `baseline/baseline_submission.zip` unmodified.
Its local score is exactly 0.617052.
- If it lands near **0.595** (also -0.022), the entire gap is dataset-level and
  our calibration gain (+0.0555 local) transferred in full.
- If it lands near **0.617**, then calibration specifically transferred worse
  and c should be re-chosen more conservatively (larger c).
Either way this fixes the local->leaderboard offset for every future comparison.

## 2026-09-18 — SUBMISSION 2: pristine baseline_submission.zip (control)
Leaderboard **0.5935** (27m23s). Local full-validation wp_masked 0.617052.

### The local -> leaderboard offset is constant
| submission | local wp_masked | leaderboard | gap |
|---|---:|---:|---:|
| pristine baseline | 0.617052 | 0.5935 | **-0.0236** |
| baseline calibrated c=0.35 | 0.672580 | 0.6501 | **-0.0225** |

The two gaps agree to 0.0011. **Local validation reads ~0.023 above the public
leaderboard, and the offset is stable across models.**

### Calibration transferred in full
- local gain from calibration: 0.672580 - 0.617052 = **+0.0555**
- leaderboard gain:            0.6501   - 0.5935   = **+0.0566**

The clipping fix is not a validation artifact — it transferred completely, in
fact marginally better on the test set. No reason to choose c more
conservatively than 0.35.

### Working rules from here
1. **Expected leaderboard ~= local wp_masked - 0.023.** Use this to predict a
   submission's score before spending one of the 5 daily slots.
2. **Improvements transfer roughly 1:1.** Local ranking of models can be
   trusted; the offset is a level shift, not a distortion.
3. Local differences below ~0.005 remain unresolvable (see the 2026-09-18
   mask-sensitivity entry).
4. **Predicted leaderboard for `02_trunk_global_full` (local 0.667194) is
   ~0.644 — BELOW the calibrated baseline's actual 0.6501.** Do not submit it
   as an improvement; it is currently our second-best candidate, not our best.

### Current leaderboard standing
Best submitted: **baseline_calibrated c=0.35 at 0.6501**, achieved with the
organizers' own model and a single constant.

## 2026-09-18 — steps 1-4: caching, per-target calibration, ensembling, Colab subset

**Step 1 — prediction caches.** `tools/predict_cache.py` (baseline's cache
already existed and was reused, 674 MB). Added:
- `tools/_trunk_global_full_raw_preds_valid.npz` (674 MB) — batched torch
  forward over all 1,873 sequences, 346s.
- `tools/_ridge_v1_raw_preds_valid.npz` (674 MB) — pure numpy matmul, 67s.

All three share the same `{preds, targets, need, scored}` format, so every
downstream analysis in this entry is pure numpy over cached arrays — no
model was re-run after this step.

**Step 2 — per-target calibration** (`tools/per_target_calibration.py`).
Independent c_t0/c_t1 grid search vs. the shared-c sweep, for all three
models:

| model | best shared c | shared wp | best (c_t0, c_t1) | per-target wp | gain | verdict |
|---|---:|---:|---|---:|---:|---|
| baseline | 0.3 | 0.673295 | (0.3, 0.35) | 0.673386 | +0.00009 | prefer shared |
| trunk_global_full | 1.0 | 0.667194 | (1.0, 1.0) | 0.667194 | +0.00000 | prefer shared |
| ridge_v1 | 1.0 | 0.586440 | (1.0, 1.0) | 0.586440 | +0.00000 | prefer shared |

**Per-target calibration does not help any model** — every gain is smaller
than the neighbouring-grid-point plateau spread (0.0009-0.0025), exactly the
over-tuning guard TASK.md asked for. Shared-c remains correct everywhere.
(Baseline's literal grid optimum is c=0.3 at 0.673295, marginally above the
previously-chosen conservative c=0.35 at 0.672580 by only 0.0007 — well
within plateau noise, so the existing packaged `baseline_calibrated`
submission at c=0.35 is not revised.)

**Step 3 — ensemble test** (`tools/ensemble_test.py`), using each model's own
best (shared) calibration from step 2: baseline x0.35, trunk_global_full and
ridge_v1 uncalibrated (x1.0).

Prediction-stream correlation between the two main models (calibrated,
scored rows): **t0=0.9607, t1=0.9602** — both above the 0.95 flag TASK.md
set, meaning limited room for ensembling to help.

| blend | wp_masked | vs. best solo (0.672580) |
|---|---:|---:|
| shared-a, best a=0.7 | 0.675001 | +0.002421 |
| per-target-a, a_t0=0.7 a_t1=0.8 | 0.675140 | +0.002560 |
| + ridge_v1 (weight 0.05-0.10) | **0.675395** | **+0.002815** |

**Best ensemble beats the best solo component by +0.002815 — below
TASK.md's ~0.003 packaging threshold**, and the two models are highly
correlated (0.96). Given local differences under ~0.005 are already
established as unresolvable (2026-09-18 leaderboard entry above), this is a
marginal, unconvincing gain. **`submissions/ensemble_v1/` was NOT created**,
per the task's own explicit criterion.

**Step 4 — Colab-sized training subset** (`tools/make_train_subset.py`).
2,500 of 10,607 sequences sampled uniformly at random (seed=42, fixed and
recorded), features stored as float16 (already clipped to roughly +/-2.4,
well within float16's range and precision), targets/masks kept float32,
same schema and row-group-per-sequence layout as `train.parquet` (so
`train/data.py`'s loader works unchanged — it upcasts to float32
immediately after reading). Written to `datasets/train_subset.parquet`,
**4,918 MB**, in 191s.

**Verification** (`tools/verify_train_subset.py`): refit `ridge_v1`'s exact
recipe (same functions from `tools/fit_ridge.py`, unmodified) on the subset
alone, scored on the full validation set: **0.586858** vs. `ridge_v1`'s
0.586440 (fit on the full-precision `train.parquet`) — **difference
+0.000418, negligible.** The subset is not degenerate; sampling and float16
conversion preserved the signal.

## 2026-09-18 — step 5: trunk_global_full exported and packaged
`python3 -m train.export_onnx --checkpoint runs/02_trunk_global_full/best.pt
--out-dir submissions/trunk_global_full --verify-sequences 50
--official-check-sequences 50`. No calibration multiplier needed (1.0,
confirmed by step 2). Verification (built into the exporter):
- PyTorch vs. ONNX: max abs prediction diff **6.497e-06** over 50 sequences;
  WP scores identical to 2.7e-09 (0.712480 both). Bit-for-bit equivalent.
- Throughput: **37.37 us/row** (budget ~91-96 us/row) -> projected **23.2
  min** for the full 1,970-sequence/39.4M-row test set (budget 60 min).
- `submissions/trunk_global_full/submission.zip`: **0.96 MB** (limit 20 MB).
- Official row-by-row check via the generated `solution.py` +
  `ScorerStepByStep` on 50 sequences: consistent with the PyTorch/ONNX
  numbers above (this 50-seq sample scores 0.712480, well above the full
  1,873-seq validation number of 0.667194 established earlier — a sampling
  effect, not a discrepancy; the 50-sequence sample simply happens to be
  easier, exactly the mask/sequence-composition sensitivity documented in
  the 2026-09-18 leaderboard-investigation entry above).

`submissions/ensemble_v1/` was **not** created (step 3's gain of +0.002815
fell below the ~0.003 packaging threshold).

### Ready submission candidates (none submitted — Dmitry decides)
| candidate | local wp_masked | predicted leaderboard (local - 0.023) | size | throughput |
|---|---:|---:|---:|---:|
| `baseline_calibrated/` (already submitted) | 0.672580 | 0.6501 (actual, confirmed) | 0.72 MB | ~34 us/row |
| `trunk_global_full/` | 0.667194 | ~0.644 (predicted, not yet submitted) | 0.96 MB | 37.37 us/row |

**`baseline_calibrated` remains the best candidate to submit first (already
submitted, in fact — 0.6501 is the current best leaderboard score).
`trunk_global_full` is not an improvement over it and should not be
submitted as one**, per the 2026-09-18 leaderboard-rules entry above (its
predicted leaderboard score sits below the already-achieved 0.6501). The
ensemble was tested and correctly rejected for being under the packaging
threshold, not created.

## 2026-09-19 — Cowork probe: LightGBM as a decorrelated ensemble member — REJECTED
Motivation: the ensemble failed because our GRU and the baseline correlate at
0.96. A structurally different model might decorrelate. Tested LightGBM
(400 rounds, 63 leaves, lr 0.05, metric weights), trained on the same 60
training sequences used for the ridge probes, evaluated on 30 validation
sequences.

| model | Global WP (30-seq sample) |
|---|---:|
| baseline x0.35 | 0.7135 |
| LightGBM | 0.6086 |
| ridge | 0.6054 |

Prediction correlations with the calibrated baseline, on scored rows:
| | t0 | t1 |
|---|---:|---:|
| GBDT vs baseline | 0.8124 | 0.7555 |
| ridge vs baseline | 0.8526 | 0.7636 |
| GBDT vs ridge | 0.8551 | 0.8346 |

Blends: best is 0.9*baseline + 0.1*GBDT = 0.7148 vs 0.7138 pure baseline,
**+0.0010** — noise.

**Two conclusions.**
1. **Nonlinearity buys almost nothing pointwise.** GBDT beats the ridge by only
   +0.003 despite being able to model arbitrary interactions of the 112
   features. Consistent with the earlier random-ReLU-feature result. The
   remaining signal is not a nonlinearity the ridge is missing.
2. **Decorrelation without strength is worthless.** GBDT does correlate much
   less with the baseline (0.76-0.81) than our GRU does (0.96) — but that
   decorrelation comes precisely from lacking the recurrent component, i.e.
   from being weaker, not from seeing something new. A 0.10 weight in a blend
   contributes +0.001.

**Direction closed.** Do not spend a Claude Code session on GBDT. Caveat: this
was a 60-sequence probe; a GBDT fitted on 1,000+ sequences would score higher,
but it would have to reach roughly GRU level (~0.667 full-valid) to matter in a
blend, and it starts ~0.06 below the ridge-to-GRU gap.

### Also ruled out on execution-budget grounds (no experiment needed)
- **kNN regression**: requires storing the training set and a neighbour search
  per row — cannot fit 20 MB, cannot meet ~91 us/row, and has no state.
- **Kernel SVM**: prediction is a sum over tens of thousands of support
  vectors. Same wall.
- **Linear SVM**: mathematically close to the ridge with a different loss; the
  linear ceiling is already measured at 0.586.
- **Time-series foundation models** (Chronos, TimesFM, Moirai): almost certainly
  pretrained on market data, which the rules forbid; univariate; and orders of
  magnitude over the per-row compute budget.
- **LSTM instead of GRU**: same family, ~30% more parameters for the same
  budget; at best noise-level difference.

### Still open, testable on CPU
**Dilated causal convolutions** with a ring buffer. Unlike a GRU, convolutions
have no sequential dependency during training, so they train *faster* on CPU,
not slower — this is a genuinely cheap experiment that does not need the GPU.
Per-row inference is O(receptive field) with a fixed buffer, which fits the
online contract. Worth a task once the GPU run is underway.

## 2026-09-19 — repository initialised
`git init` in the project root, `.gitignore` excluding `datasets/`, all
parquet, cached prediction arrays, `runs/*/ckpt_latest.pt`, `runs/*/wandb/`,
secrets and `.claude/`. First commit: 100 files, ~18.9 MB (mostly the small
`.pt` checkpoints and ONNX submission artifacts). Remote set to
`git@github.com:DmitryTsybulkin/wundernn-competition.git`, branch `main`.
Push from the sandbox fails with `Host key verification failed` (no SSH agent
or known_hosts there) — **Dmitry pushes from his own terminal**:
`cd <repo> && git push -u origin main`.
