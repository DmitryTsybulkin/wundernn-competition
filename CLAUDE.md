# Working protocol for this repository

Competition: Wunder Fund "Alfa Connectome". Metric: **Global Weighted Pearson**
(`global_wp`). Run everything with `python3`.

## The notes/ protocol
This repo is worked on from two places: a Cowork chat (analysis, hypotheses,
task authoring) and Claude Code sessions (execution). `notes/` is the only
channel between them. Keep it accurate — the other side has no other context.

- `notes/STATUS.md` — current state of the world: metric, best scores, known
  facts, known defects. **Overwrite** the parts that changed. Keep it short
  enough to read in full at the start of a session.
- `notes/TASK.md` — what to do right now. Written by the Cowork side.
  **Do not edit it** except to tick items off; it is replaced wholesale for the
  next task.
- `notes/LOG.md` — **append-only**, newest at the bottom. One block per run or
  investigation: what was run, the exact command, the numbers, the conclusion.
  Never rewrite history here.

## Required at the end of every session
1. Append a `notes/LOG.md` entry for everything you ran, including failures and
   dead ends. A negative result that is not written down will be repeated.
2. Update `notes/STATUS.md`: scores table, defect list.
3. Print a short summary in the chat: what changed, what the numbers are, what
   is now the best model, what you would do next.

## Required for every training run
Each run writes `runs/<name>/` containing at minimum:
- `log.txt` — stdout
- `config.json` — full resolved config, including parameter count
- `best.pt`, `last.pt`
- `summary.md` — 5-10 lines: config, best Global WP (both masked and all-rows),
  step at which it peaked, wall clock, and one sentence on what was learned.

Name runs `<NN>_<short_slug>`, numbered sequentially.

## Hard rules
- Never compare a number produced by a different metric version. If in doubt,
  re-measure.
- Report two numbers for every evaluation: `wp_masked` (official, on
  `need_prediction AND is_scored`) and `wp_all_rows` (on all `need_prediction`
  rows). Divergence means the model is fitting the visible mask.
- Never normalize predictions per sequence. Under Global WP this costs ~0.05 WP.
- Print the model parameter count at startup of every run.
- Do not delete anything under `runs/` or `datasets/`.
