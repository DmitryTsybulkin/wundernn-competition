"""notes/TASK_gpu_prep.md step 2: batch-size scaling benchmark. NOT a training run.

For each batch size: the real loader + model + loss + optimizer path used by train.py
(trunk[256,128] + ridge-init skip + GRU 128x2, global loss, EMA, chunk_len 2000), N timed steps
after warm-up, on the requested device. Reports compute s/step (forward+backward+optimizer+EMA,
excluding data loading), loader time per batch of sequences (amortised over the windows it
feeds), rows/s (= batch*chunk_len / s), peak resident memory, and the loss values (sanity:
must be finite and in the same range at every batch size).

usage: python3 -m train.bench_batch [--device cpu] [--batch-sizes 64 128 256 512] [--steps 4]
"""
from __future__ import annotations

import argparse
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train.data import TBPTTLoader
from train.losses import global_wp_loss
from train.model import ModelConfig, SequenceModel

ROOT = Path(__file__).resolve().parents[1]


def rss_gb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e9 if sys.platform == "darwin" else r / 1e6  # macOS: bytes, Linux: KiB


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def bench(batch_size, device, steps, chunk_len, train_path, lr):
    torch.manual_seed(0)
    cfg = ModelConfig(trunk_hidden=(256, 128), gru_hidden=128, gru_layers=2, use_skip=True)
    model = SequenceModel(cfg)
    model.init_skip_from_ridge(str(ROOT / "models/ridge_v1.npz"))
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    decay = 0.98
    ema = torch.optim.swa_utils.AveragedModel(model, avg_fn=lambda a, c, _n: decay * a + (1 - decay) * c)
    loader = TBPTTLoader(train_path, batch_size=batch_size, chunk_len=chunk_len, seed=0)

    t_load0 = time.time()
    it = loader.epoch()
    first = next(it)  # includes reading one batch of sequences
    load_s = time.time() - t_load0
    n_windows = 20000 // chunk_len

    times, losses = [], []
    hidden = None
    window = first
    for step in range(steps + 1):  # step 0 = warm-up (allocator, cudnn autotune, page faults)
        if window.is_first:
            hidden = model.init_hidden(window.features.shape[0], device=device)
        else:
            hidden = hidden.detach()
        x = torch.from_numpy(window.features).to(device)
        t0 = torch.from_numpy(window.t0).to(device)
        t1 = torch.from_numpy(window.t1).to(device)
        mask = torch.from_numpy(window.mask).to(device)
        sync(device)
        t = time.time()
        model.train()
        pred, hidden = model(x, hidden)
        loss, wp0, wp1 = global_wp_loss(pred[..., 0], pred[..., 1], t0, t1, mask)
        opt.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema.update_parameters(model)
        sync(device)
        dt = time.time() - t
        if step > 0:
            times.append(dt)
            losses.append(loss.item())
        assert torch.isfinite(loss) and torch.isfinite(gn), f"non-finite loss/grad at batch {batch_size} step {step}"
        if step < steps:
            window = next(it)
    med = float(np.median(times))
    return {
        "batch": batch_size, "s_step": med, "s_step_mean": float(np.mean(times)),
        "rows_s": batch_size * chunk_len / med, "load_s_per_batch": load_s,
        "load_s_per_step": load_s / n_windows, "loss_first": losses[0], "loss_last": losses[-1],
        "peak_rss_gb": rss_gb(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[64, 128, 256, 512])
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--chunk-len", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--train-path", default=str(ROOT / "datasets/train_subset.parquet"))
    args = ap.parse_args()
    device = torch.device(args.device)
    print(f"device={device} threads={torch.get_num_threads()} chunk_len={args.chunk_len} steps/size={args.steps} "
          f"(+1 warm-up) train={Path(args.train_path).name}", flush=True)
    rows = []
    for b in args.batch_sizes:
        r = bench(b, device, args.steps, args.chunk_len, args.train_path, args.lr)
        rows.append(r)
        print(f"batch {r['batch']:>4}: {r['s_step']:.2f} s/step (median; mean {r['s_step_mean']:.2f})  "
              f"{r['rows_s']:,.0f} rows/s  load {r['load_s_per_batch']:.1f}s/batch = {r['load_s_per_step']:.2f}s/step  "
              f"loss {r['loss_first']:.4f}->{r['loss_last']:.4f}  peak RSS {r['peak_rss_gb']:.1f} GB", flush=True)
    base = rows[0]
    print("\n| batch | s/step | rows/s | vs batch %d | ms per 1000 rows | load s/step | peak RSS GB | loss (first->last) |" % base["batch"])
    print("|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in rows:
        print(f"| {r['batch']} | {r['s_step']:.2f} | {r['rows_s']:,.0f} | {r['rows_s']/base['rows_s']:.2f}x | "
              f"{1000/r['rows_s']*1000:.2f} | {r['load_s_per_step']:.2f} | {r['peak_rss_gb']:.1f} | {r['loss_first']:.3f} -> {r['loss_last']:.3f} |")


if __name__ == "__main__":
    main()
