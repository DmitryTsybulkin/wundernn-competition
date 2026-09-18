# TASK — make training GPU-ready, resumable, and throughput-scaled

Read `notes/STATUS.md` first. Follow `CLAUDE.md`. Use `python3`.
Append to `notes/LOG.md`, update `notes/STATUS.md` when done.

**Do not start a long training run in this task.** Everything here is
preparation, benchmarking and dry-running on CPU. The long run happens next, on
a rented T4 in Google Colab, and every Colab minute costs compute units — so
nothing may be debugged there.

## Why this task exists
`02_trunk_global_full` was still improving monotonically when a 90-minute wall
clock stopped it. It is the only untested direction left: calibration is taken,
per-target calibration does not help, the ensemble fell below threshold, the
ridge is weaker. Current leaderboard: leader 0.6622, us 0.6501, pristine
baseline 0.5935 — the whole field spans 0.069, so remaining gains are small and
must not be wasted on avoidable failures.

## 1. Resumable training — the hard requirement
Colab sessions are interrupted (time limits, disconnects, idle timeouts). A run
that cannot resume turns an interruption into hours of wasted compute units.

Checkpoint every N steps (`--checkpoint-every`, default ~200) into
`runs/<name>/ckpt_latest.pt`, written atomically (temp file + rename, so an
interrupted write cannot corrupt it). It must contain everything needed to
continue bit-identically:
- model state dict
- optimizer and LR-scheduler state
- step counter, epoch, wall-clock elapsed so far
- the data iterator's position (which sequences/windows are consumed) and the
  RNG states (python, numpy, torch) needed to reproduce the shuffle
- the best-so-far score and the config

Add `--resume <path>` (and `--resume auto` to pick up `ckpt_latest.pt` if it
exists). On resume, print what it restored: step, epoch, elapsed, best score.

**Test it properly**: run 300 steps straight; separately run 150 steps, kill the
process, resume, run 150 more. The two final checkpoints must agree closely —
report the max absolute parameter difference and the two validation scores. A
large divergence means the RNG or data position is not actually being restored.

## 2. Batch-size scaling benchmark
The model is small and the GRU is sequential over the chunk, so at batch 64 a
GPU will sit mostly idle. Batch size — not the GPU itself — is the main lever on
how much data a run sees.

On CPU, measure s/step and rows/second at batch 64, 128, 256 (larger if RAM
allows) with chunk_len 2000, everything else fixed. Report a table.

The purpose is not CPU speed; it is to establish the shape of the curve and to
confirm larger batches work correctly (no OOM, no silent shape bugs, loss values
sane). Note in the log which batch sizes are worth trying first on the GPU.

Also add `--lr` scaling guidance to the log: if batch grows 8x, the LR that was
right at batch 64 is probably not right at 512. Suggest a starting rule (e.g.
square-root scaling) but do not tune it here.

## 3. Colab notebook
`colab/train_wnn.ipynb`, written to run top to bottom with no editing except two
constants at the top (repo URL, Drive folder):
1. print the assigned GPU (`nvidia-smi`) and abort loudly if none
2. mount Google Drive; expect `train_subset.parquet` and a validation file there
3. clone the private repo via a token read from Colab secrets — **never
   hard-code or print a token**
4. `pip install` whatever is missing versus Colab's preinstalled stack
5. run training with `--resume auto` so re-running the cell after a disconnect
   continues rather than restarts
6. on completion (or on resume-after-interrupt), commit and push back to the
   repo: `runs/<name>/summary.md`, `log.txt`, `config.json`, `best.pt`,
   `ckpt_latest.pt` (all small), with `datasets/` and any large artifacts
   gitignored
7. print the elapsed time and an estimate of compute units consumed

Include a "smoke mode" flag that runs ~20 steps end to end, so the first Colab
session verifies the whole pipeline for a couple of minutes of units instead of
discovering a bug three hours in.

## 4. Validation data for Colab
Full `valid.parquet` is 5.6 GB. For in-run monitoring that is wasteful.
Produce `datasets/valid_subset.parquet`: 500 sequences sampled with a fixed,
recorded seed, float16 features like the training subset, same schema.

Then measure the offset: score `02_trunk_global_full/best.pt` on this 500-
sequence subset and on the full 1,873, and record the difference. In-run numbers
will come from the subset; final numbers always from the full set, locally.

## 5. Full CPU dry run
Run the complete Colab path locally in smoke mode: the same entry point, same
flags, `--resume auto`, 20 steps, checkpoint, kill, resume, finish, produce
`summary.md`. Confirm it works end to end with zero manual intervention.

Report the exact command line the Colab notebook will execute, so it can be
reviewed before any units are spent.

## Deliverables
1. LOG.md: the resume-correctness test numbers, the batch-scaling table, the
   valid-subset offset, and the exact GPU command line.
2. STATUS.md: note that training is now resumable and Colab-ready; record the
   leaderboard context (leader 0.6622, us 0.6501, baseline 0.5935).
3. A short summary: is the pipeline safe to run on paid GPU time, and which
   batch size / LR to start with.
