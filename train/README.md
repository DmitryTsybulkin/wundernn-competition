# Training pipeline

```
train/
├── data.py         streaming TBPTT loader over train.parquet (1 row group = 1 sequence)
├── model.py        pointwise MLP trunk -> GRU -> linear head(2), optional t0/t1 reparam
├── losses.py        differentiable weighted-Pearson loss (matches METRIC.md exactly)
├── eval.py          fast batched validator (torch) + official row-by-row validator (utils.ScorerStepByStep)
├── train.py          training loop, wandb logging, checkpointing
├── export_onnx.py    checkpoint -> ONNX (stateful single-step) -> solution.py -> submission.zip, + verification
└── selftest.py       correctness checks — run this after touching model.py/losses.py
```

## Quickstart

```bash
python3 -m train.selftest   # correctness checks (metric math, batched==stepwise, NaN-gradient regression)

python3 -m train.train \
  --max-row-groups 3000 --batch-size 64 --chunk-len 2000 \
  --gru-hidden 128 --gru-layers 2 --epochs 4 \
  --valid-sequences 200 --valid-every-windows 200 \
  --out-dir runs/my_run --wandb-mode offline

python3 -m train.export_onnx \
  --checkpoint runs/my_run/best.pt --out-dir runs/my_run_export \
  --verify-sequences 50 --official-check-sequences 200
```

`export_onnx.py` writes `model.onnx`, `solution.py`, and `submission.zip` (checked against
the 20MB limit), then verifies the exported ONNX reproduces the PyTorch model's predictions
(max abs diff, expect ~1e-7) and WP score, runs an official `utils.ScorerStepByStep` spot
check through the generated `solution.py`, and reports throughput in us/row against the
96us/row budget.

## Notes

- CPU is faster than MPS for this workload on this machine (small recurrent net, batch<=64
  — MPS kernel-launch overhead dominates). `train.py` defaults to CPU; pass `--device mps`
  to override.
- `--reparam` trains the head to predict `(t0+t1)/2, (t0-t1)/2` instead of `t0, t1` directly;
  the inverse transform is baked into the model (including the ONNX export), so it's
  invisible to everything downstream.
- wandb logs to project `Wundernn`, entity `my-lore`, mode `offline` by default (syncable
  later with `wandb sync <run dir>` once an API key is available). Never imported outside
  `train.py`.
