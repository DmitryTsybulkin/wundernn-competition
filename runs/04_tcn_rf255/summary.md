# 04_tcn_rf255

**Config**: identical to `03_tcn_rf63` except `--tcn-layers 7` (dilations 1-64): receptive
field 255 rows, 409,060 params. trunk 256,128 + ridge-init skip + global loss, batch 64,
chunk 2000, early-stop patience 5, `--max-seconds 3600`, eval every 200 steps on 400 seqs.

**Result (full 1,873-sequence validation)**: best.pt (= last.pt) wp_masked=**0.651498**
(t0=0.645643, t1=0.657352), wp_all_rows=0.472617, ratio=0.725. No calibration needed
(multiplier=1.0; every c<1 strictly worse).

**Peaked at**: not converged — step 1200, when the 60-min cap stopped it (3700 s); improved
at every evaluation (400-seq sample 0.6275 -> 0.6635).

**Wall clock**: 62 min. ~560-590 s per 200 steps incl. eval in steady state (the first two
intervals, 784 s / 640 s, overlapped a full-validation eval job), vs. ~437 s for RF63 and
~480 s for the GRU — so only 1200 steps fit in the cap vs. 1600 for RF63.

**What was learned**: at *matched steps* the longer receptive field is slightly better than
RF63 (+0.002 on the 400-seq sample at steps 1000/1200), but that does not pay for its ~30%
higher cost under a fixed time cap, and it is still 0.008 behind the GRU at the same step;
its predictions correlate 0.97 with the GRU's, so it adds nothing to a blend.
