# STATUS — Wundernn Alfa Connectome

Last updated: 2026-09-19

## Metric (CHANGED 2026-09-15/16)
**Global Weighted Pearson.** One weighted correlation per target over ALL rows
selected by `need_prediction AND is_scored` across the entire dataset, then the
two are averaged. Weights `|clip(y,-2,2)|`. No per-sequence averaging, no
sequence eligibility rule. Spec id `global_wp`.

Official implementation: `GlobalAccumulator` in the updated `utils.py`
(`/Users/main/Projects/Wundernn/wnn_connectome_updated/utils.py`).

## Leaderboard standing (2026-09-19)
| | score |
|---|---:|
| leader | 0.6622 |
| **us (`baseline_calibrated` c=0.35)** | **0.6501** |
| pristine baseline | 0.5935 |

The entire field spans 0.069 above the pristine baseline. Remaining gains are
small; the 0.012 to the leader is comparable to the valid->test shift we
measured (0.023), so late-stage submission choice should favour robustness over
a third decimal on the public board. To pass the leader we need ~0.685 local.

## Leaderboard vs local (measured, 2 submissions)
| submission | local wp_masked | leaderboard | gap |
|---|---:|---:|---:|
| baseline calibrated c=0.35 | 0.672580 | **0.6501** | -0.0225 |
| pristine baseline | 0.617052 | 0.5935 | -0.0236 |

**Rule: expected leaderboard ~= local wp_masked - 0.023**, stable across models.
Improvements transfer ~1:1 (calibration gained +0.0555 locally, +0.0566 on the
leaderboard). Local differences below ~0.005 are unresolvable.
Best submitted so far: **0.6501**. Predicted leaderboard for
`02_trunk_global_full` is ~0.644 — do not submit it, it is behind.

## Current scores (Global WP, full 1873-sequence validation)
| model | wp_masked (official) | wp_masked t0/t1 | wp_all_rows | old per-block WP |
|---|---:|---|---:|---:|
| **baseline, calibrated x0.35 (`submissions/baseline_calibrated/`)** | **0.672580** | — | 0.500796 | n/a |
| 02_trunk_global_full/best.pt (=last.pt, trunk+skip+global loss, epochs fixed) | 0.667194 | 0.663937 / 0.670452 | 0.490706 | n/a |
| 01_trunk_global/last.pt (same recipe, epochs bug, see log) | 0.664832 | 0.660792 / 0.668872 | 0.485578 | n/a |
| 01_trunk_global/best.pt | 0.663142 | 0.659487 / 0.666798 | 0.484664 | n/a |
| provided GRU baseline, uncalibrated | 0.617052 | 0.587913 / 0.646192 | 0.440156 | 0.589595 |
| ridge_v1 (`models/ridge_v1.npz`), c=1.0 | 0.586440 | — | 0.402877 | n/a |
| stage5_ema_diag/best.pt | 0.477417 | 0.416430 / 0.538404 | 0.236610 | 0.584785 |
| stage5_ema_diag/last.pt | 0.262700 | 0.166794 / 0.358605 | 0.077048 | 0.584785 |
| stage6_full_ema/best.pt | 0.457501 | 0.354938 / 0.560064 | 0.220781 | 0.582431 |
| stage6_full_ema/last.pt | 0.099061 | 0.075086 / 0.123036 | 0.023303 | 0.582431 |
| stage7_ema_cosine/best.pt | 0.459650 | 0.358816 / 0.560485 | 0.224309 | 0.582806 |
| stage7_ema_cosine/last.pt | 0.377182 | 0.270685 / 0.483678 | 0.143181 | 0.582806 |
| weighted ridge, 112 raw features, memoryless (60-seq sample) | ~0.5735 (OLD metric only) | — | — | 0.5443 |

**Calibrating the provided baseline's output by a constant 0.35x beats the
baseline itself: 0.672580 vs 0.617052 (+0.0555).** Packaged, verified
deterministic, ~34 us/row (~22 min for the full test set, well under budget)
— see `notes/LOG.md` 2026-09-17 steps 1-2. Not submitted; reported for Dmitry
to decide. `ridge_v1` (step 3) beats every stage2-7 checkpoint (best 0.477417)
while sitting 0.031 below the uncalibrated baseline.

**Best trained model: `02_trunk_global_full` at 0.667194** (trunk[256,128] +
ridge-init linear skip + correctly-pooled global loss) — the first
checkpoints from this pipeline to beat the baseline outright (+0.050 over
0.617052), needing no calibration (multiplier=1.0 throughout — already
well-scaled) and with a healthy mask ratio (0.735, in line with the
baseline's 0.713, unlike stages 2-7's 24-50%). `01_trunk_global` (same
recipe, but `--epochs` accidentally defaulted to 1) got 0.664832 before
running out of data; `02_trunk_global_full` (`--epochs 4`, otherwise
identical, same seed) picked up where it left off and reached 0.667194
before the 90-minute `--max-seconds` budget stopped it — **still improving
monotonically at cutoff, not converged.** Still 0.0054 below the
calibrated-baseline candidate (0.672580). See `notes/LOG.md` 2026-09-17 for
both runs' full curves and `runs/02_trunk_global_full/summary.md`.

**Mask-dependence, resolved (control measured):** baseline's
wp_all_rows/wp_masked ratio is **71.3%** (0.440156/0.617052); our six
checkpoints range 24-50% (best.pt's cluster at 48-50%). Some divergence is
apparently inherent to how `is_scored` selects rows in this dataset (even the
baseline doesn't hit 100%), but our checkpoints are well below the baseline's
level — genuinely more mask-dependent, a real weakness on top of the
untested trunk/loss fixes. Next training run should watch this ratio and
aim for something closer to ~70%, not our previous ~25-50%.

## Per-target calibration and ensembling (2026-09-18)
**Per-target calibration does not help.** Independent c_t0/c_t1 grid search
for all three models (baseline, `trunk_global_full`, `ridge_v1`) gained at
most +0.00009 over shared-c, smaller than the local plateau spread
(0.0009-0.0025) in every case — prefer shared-c everywhere. See
`notes/LOG.md` 2026-09-18 for the full table.

**Ensembling gives a small, sub-threshold gain.** Calibrated baseline +
`trunk_global_full` (per-target blend weights, plus a small `ridge_v1`
contribution): best **0.675395**, beating the best solo component
(calibrated baseline, 0.672580) by **+0.002815** — below the ~0.003
packaging bar, and the two models' predictions correlate at 0.96 on both
targets (little independent signal to combine). `submissions/ensemble_v1/`
was not created.

## Ready submission candidates (none submitted by this session)
| candidate | local wp_masked | leaderboard | size | throughput |
|---|---:|---:|---:|---:|
| `submissions/baseline_calibrated/` | 0.672580 | **0.6501 (submitted, confirmed)** | 0.72 MB | ~34 us/row |
| `submissions/trunk_global_full/` | 0.667194 | ~0.644 (predicted, not submitted) | 0.96 MB | 37.37 us/row |

`baseline_calibrated` remains the best submission — already achieved
0.6501 on the leaderboard. `trunk_global_full` is not an improvement over
it (predicted ~0.644 < 0.6501) and should not be submitted as one. Do NOT
submit anything without Dmitry's decision; daily limit is 5.

## Pipeline correctness fixes this session (2026-09-16)
- **Trunk bug root-caused and fixed.** Not a `model.py` bug — `train.py`'s
  `--trunk-hidden` used `nargs="*"`, and every prior invocation wrote
  `--trunk-hidden \` immediately followed by another flag on the next line, so
  argparse silently parsed zero values. Every run through stage7 built the
  baseline's exact 192,258-param no-trunk architecture; the trunk hypothesis
  was never tested at scale. Fixed: `nargs="+"` (that pattern is now a hard
  parse error) plus a startup assertion and full architecture printout. See
  `notes/LOG.md` 2026-09-16 for verification.
- **Loss fixed for Global WP.** New `global_wp_loss` (`train/losses.py`) pools
  every row in a training batch into one weighted correlation per target,
  matching `GlobalAccumulator`'s dataset-global pooling (no per-sequence
  renormalization/equalization). Old `chunk_wp_loss` — which implicitly did
  exactly the per-sequence equalization Global WP penalizes — kept behind
  `--loss-mode per_chunk` for comparison; `global` is now the default. All six
  existing checkpoints (stage5/6/7) were trained under the OLD `chunk_wp_loss`
  and are stale relative to this fix — worth a retrain once step 5's numbers
  are in.
- **Evaluator repointed.** `train/eval.py`'s `evaluate_fast` now uses
  `utils.GlobalAccumulator` directly (bit-identical to `ScorerStepByStep`'s
  aggregation) and returns both `wp_masked` (official) and `wp_all_rows`
  (mask-fitting diagnostic). Checkpoint selection uses `wp_masked` only.
- **MPS benchmarked, not worth it.** 20 real training steps, baseline
  architecture: CPU 1590.6 ms/step vs MPS 2468.3 ms/step — CPU is 1.55x
  faster on this Mac for this model size, no fallback warnings, identical
  loss values between devices. If training throughput remains the binding
  constraint, benchmark a rented CUDA GPU next, not MPS.

## Known facts about the data
- Sequence = 20,000 rows; steps 0-98 warm-up; 112 float32 features.
- Features already normalized/clipped to roughly +/-2.4; `a5`-`a7` wider, floor -5.199.
- Column indices are scrambled: `p0` is NOT the best bid. Classic microstructure
  feature engineering (spread, imbalance, depth weighting) is impossible.
- corr(t0,t1) = -0.71. ~20% of target values are exactly 0 (zero metric weight).
- Target autocorrelation 0.96 @lag1, 0.36 @lag100, ~0 @lag1000.
- `is_scored` covers 1-36% of rows per sequence, in contiguous clusters.
- Only ~9.5% of required predictions are scored on validation.

## Measured modelling facts
- A memoryless weighted ridge on the raw 112 features reaches ~92% of the GRU's
  score. Recurrence is worth only ~0.05 WP.
- EWMA history features (halflife 4/16/64/256) add nothing linearly.
- Causal EMA smoothing of predictions does not help (hl=2 neutral, hl>=4 degrades).
- Random nonlinear features (1024 random ReLU) hurt.
- i1 block adds nothing linearly.
- **Per-sequence z-scoring of predictions now COSTS 0.049 Global WP.** Under the
  old metric it was slightly positive. Never normalize per sequence again.
- Under Global WP sequences are weighted by their summed `|clip(y,-2,2)|`:
  measured spread 49x, top-10 of 60 sequences carry 37% of total weight.
- **OUTPUT SCALE MATTERS because of the clip.** 38% of the baseline's predictions
  exceed |2| and are clipped by the metric, destroying ordering exactly on the
  highest-weight rows. Scaling all predictions by ~0.4 gained +0.033 Global WP on
  a 30-sequence sample (0.6812 -> 0.7137) with no training. Global WP is
  scale-invariant only BEFORE clipping. Calibrate the output scale of every model
  before scoring or submitting. Needs full-validation confirmation (TASK step 1).
- Ensembling the baseline with a ridge adds ~nothing (+0.0004) once both are
  correctly scaled — the apparent gain was the rescaling. Dead end.
- Per-sequence rescaling of predictions does not recover the checkpoints' losses,
  even with oracle knowledge of each sequence's target magnitude. Sequence-level
  mean|y| is barely predictable from features (corr 0.108). Dead end.
- **ridge_v1 (`models/ridge_v1.npz`, `tools/fit_ridge.py`), full 1,873-sequence
  validation: wp_masked=0.586440, wp_all_rows=0.402877 (ratio 68.7%), c=1.0
  uncalibrated — c=1.0 is already optimal, unlike the baseline.** Beats all six
  trained GRU checkpoints (best 0.477417) by a wide margin, sits 0.0306 below
  the baseline (0.617052), and its wp_all_rows ratio (68.7%) is close to the
  baseline's (71.3%) rather than our checkpoints' (24-50%) — the ridge is not
  mask-fitting. Strong candidate for the step-5 linear skip connection.
- `global_wp_loss`/`chunk_wp_loss` already clip predictions to [-2,2] exactly
  like the metric (`weighted_pearson_torch`'s `torch.clamp`) — verified, no fix
  needed. All prior training runs already optimized the correctly-clipped
  objective; the checkpoints' problems are not a training/eval loss mismatch
  on this axis.
- Calibration is now part of the pipeline: `train/eval.py` has
  `sweep_output_multiplier` (runs the model once, sweeps cheaply against
  cached predictions); `train.py` auto-calibrates `best.pt`/`last.pt` after
  training and stores the chosen multiplier in the checkpoint; `export_onnx.py`
  bakes it into the ONNX graph by scaling `model.head.weight`/`bias` (verified
  mathematically exact, including through the reparam transform — see
  `train/selftest.py`).

## Execution limits
- 1 vCPU, 16 GB RAM, no internet, Python 3.11, zip <= 20 MB.
- Test set: 1,970 sequences = 39.4M rows, 60 min total => **~91 us/row**.
- Test mask is hidden and may differ in character from the visible validation mask.
  Organizers explicitly warn against selecting on the visible mask alone.

## Pipeline state
`train/` — data.py, model.py, losses.py, eval.py, train.py, export_onnx.py, selftest.py.
Runs in `runs/`. wandb offline.

### Known defects
1. **RESOLVED 2026-09-16.** MLP trunk never ran — root cause was an argparse
   footgun in `train.py` (`nargs="*"` swallowing the value when
   `--trunk-hidden` was immediately followed by another flag), not a
   `model.py` bug. Fixed (`nargs="+"` + startup assertion + verified). The
   trunk hypothesis itself is still untested at scale — that's separate,
   future training work, not a code defect.
2. **RESOLVED 2026-09-16.** `train/eval.py` now uses `utils.GlobalAccumulator`
   directly.
3. **RESOLVED 2026-09-16 (code).** `global_wp_loss` added and made the
   default (`--loss-mode global`); old per-chunk loss kept behind
   `--loss-mode per_chunk` for comparison. Existing checkpoints (stage5/6/7)
   were still trained under the old loss and would need a retrain to benefit.
4. **RESOLVED 2026-09-16.** Checkpoint selection now uses `wp_masked` going
   forward, and all six existing checkpoints have been re-measured under
   Global WP (see scores table above / `notes/LOG.md` step-5 entry). In every
   run `best.pt` beats `last.pt` by a wide margin under the new metric too
   (e.g. stage6: 0.458 vs 0.099), confirming the old best-checkpoint picks
   still hold up qualitatively even though the absolute numbers are far lower
   than the retired metric implied. None beat the baseline.
5. Data starvation: `windows/epoch: 1650` against 10,607 sequences ≈ 0.4% of train.
   CPU throughput (~2.6 s/step incl. eval) is the binding constraint. Not
   addressed this session — MPS benchmarked and ruled out as a fix (see log);
   a rented CUDA GPU is the remaining untested option.
6. Valid-subset bias: 200-sequence subset read ~0.011 high under the old metric.
   Not re-measured under Global WP yet.
