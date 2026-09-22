# TASK — dilated causal convolutions (TCN) as a second, decorrelated model

Read `notes/STATUS.md` first. Follow `CLAUDE.md`. Use `python3`. Append to
`notes/LOG.md`, update `notes/STATUS.md` when done.

**This is the active task.** `notes/TASK_gpu_prep.md` is queued and must NOT be
started in this session — it will run later, once Colab is set up. This task is
CPU-only by design and does not compete with it.

## Why
Everything cheap is exhausted: calibration is taken (+0.0555, transferred to
the leaderboard in full), per-target calibration does not help, the ridge and
LightGBM are far behind, and the GRU+baseline ensemble failed because the two
correlate at 0.96. LightGBM decorrelated (0.76-0.81) but only by being weaker.

What is needed is a model that is **both strong and structurally different**.
A temporal convolutional network is the best remaining candidate:
- It has a genuinely different inductive bias from a GRU — a fixed finite
  receptive field with explicit per-lag weights, rather than a learned
  gated state. The signal here is known to be fast (target autocorrelation
  0.96@lag1 but the *predictable* part does not respond to smoothing), which
  suits a bounded receptive field.
- **It trains faster than a GRU on CPU**, not slower: convolutions have no
  sequential dependency across the chunk, so a training step parallelises over
  time. This is the rare experiment the GPU is not needed for.
- Inference fits the online contract: keep a ring buffer of the last R rows and
  compute one forward pass per row, O(R) with a fixed R.

Target to beat: `02_trunk_global_full` at 0.667194 full-validation wp_masked.
Even landing somewhat below that is a useful result **if** its predictions
correlate meaningfully less than 0.96 with the GRU's — that is what makes an
ensemble worth revisiting.

## 1. Implement the model — DONE (2026-09-21)
Add to `train/model.py` (do not disturb the GRU path — the queued GPU task
depends on it):
- causal 1-D convolution stack over the feature dimension, dilations
  1, 2, 4, 8, ... up to a configurable depth; residual connections and
  (weight or layer) normalisation between blocks
- the same **MLP trunk** on the 112 features before the stack, and the same
  **ridge-initialised linear skip** from raw features to the two outputs that
  made the GRU work — those two fixes are what got the GRU above the baseline
  and there is no reason to re-discover them
- `--arch {gru,tcn}` selects the path; everything else (loss, evaluator,
  checkpointing, early stopping) is shared and unchanged
- print the receptive field in rows and the parameter count at startup

Warm-up is 99 rows, so a receptive field beyond ~99 cannot be fully populated
at the first required prediction. Handle the beginning of a sequence by
zero-padding the buffer, and state plainly in the log how you handled it.

## 2. Correctness before performance — DONE, both gates passed
- **Causality test**: assert that changing row t's features cannot alter the
  prediction at any row < t. Do it numerically, by perturbing one row and
  checking which outputs move. A causality bug here would silently leak the
  future and produce a beautiful validation score that collapses on the
  leaderboard.
- **Streaming equivalence**: the ring-buffer per-row inference path must
  reproduce the batched training-time forward pass to ~1e-5 on a full
  sequence. Report the max absolute difference.

Both tests must pass before any training run. If either fails, stop and report.

## 3. Short training runs — DONE (03_tcn_rf63, 04_tcn_rf255; wider RF~511 skipped, reason in LOG)
Two or three configurations, each capped at **60 minutes** wall clock:
- receptive field ~64 (small), ~256 (medium), and one wider if time allows
- `--loss-mode global`, ridge-initialised skip, trunk 256,128
- early stopping patience 5, evaluate every 200 steps on 400 validation
  sequences, report `wp_masked`, `wp_all_rows` and their ratio (the baseline's
  is 0.713; the GRU's is 0.735 — anything much lower means mask-fitting)

Compare s/step and rows/second against the GRU's ~2.5 s/step at batch 64. If
the TCN is substantially faster, say so and note what batch size it allows.

## 4. The measurement that decides the direction — DONE
For the best TCN checkpoint, on the **full 1,873-sequence** validation set:
- `wp_masked`, `wp_all_rows`, and the ratio
- **the correlation of its predictions with `02_trunk_global_full`'s and with
  the calibrated baseline's, per target, on scored rows**
- the best blend of TCN + GRU + calibrated baseline, and whether it beats the
  best single model by more than 0.003 (the same threshold as before)
- the calibration sweep — does the TCN need a multiplier, like the baseline
  did, or is it self-calibrated like the GRU?

Reuse the cached prediction arrays from the earlier ensemble work rather than
recomputing them.

## 5. Verdict — DONE: TCN CLOSED (see LOG.md 2026-09-21)
End with a clear recommendation in `notes/LOG.md`: is the TCN worth pursuing on
GPU alongside the GRU, worth keeping only as an ensemble member, or closed like
LightGBM. State the numbers that justify it.

## Deliverables
1. `notes/LOG.md`: causality and streaming test results, the training table,
   the full-validation numbers, the correlation matrix, the blend result.
2. `notes/STATUS.md`: scores table updated; TCN recorded as open or closed.
3. `runs/<NN>_tcn_*/summary.md` per `CLAUDE.md` for each run.
4. Do not export a submission unless the TCN or a blend containing it beats
   0.672580 locally.
