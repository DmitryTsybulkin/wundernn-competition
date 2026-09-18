# 01_trunk_global

**Config**: same recipe as `02_trunk_global_full` but `--epochs` was left at
its default of 1 by mistake (not explicitly passed) — see notes/LOG.md
2026-09-17. 260,452 params. Skip connection initialized from
`models/ridge_v1.npz`.

**Result (full 1,873-sequence validation)**: best.pt wp_masked=0.663142
(wp_all_rows=0.484664, ratio=0.731); last.pt wp_masked=**0.664832**
(wp_all_rows=0.485578, ratio=0.730).

**Peaked at**: never — the run simply ran out of data at the end of its
single epoch (step 1650) while still improving every single evaluation
(never triggered early stopping) and well before the 90-minute
`--max-seconds` budget (only 64 min elapsed). `last.pt` (the very last step)
scored higher than `best.pt` (the last periodic-eval checkpoint), confirming
it hadn't peaked.

**Wall clock**: ~65 minutes (cut short by the epochs mistake, not by design).

**What was learned**: superseded by `02_trunk_global_full`, which is the
same recipe with the epochs bug fixed. Kept for the record since it was the
first checkpoint from this pipeline to ever beat the baseline.
