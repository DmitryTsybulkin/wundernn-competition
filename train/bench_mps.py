"""Step 8 of notes/TASK.md: short CPU-vs-MPS training-step benchmark. NOT a training run —
20 real steps per device, identical config and data, to decide whether moving training
(not inference — that stays 1 vCPU/no GPU per the competition rules) to MPS is worth it."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train.data import TBPTTLoader
from train.losses import global_wp_loss
from train.model import ModelConfig, SequenceModel

N_STEPS = 20
BATCH_SIZE = 64
CHUNK_LEN = 2000
MAX_ROW_GROUPS = 256  # >= BATCH_SIZE * ceil(N_STEPS / (20000//CHUNK_LEN)) worth of distinct sequences
ROOT_TRAIN = str(Path(__file__).resolve().parents[1] / "datasets/train.parquet")


def run(device_name: str, seed: int = 0):
    device = torch.device(device_name)
    torch.manual_seed(seed)
    cfg = ModelConfig(gru_hidden=128, gru_layers=2)  # baseline-matching architecture
    model = SequenceModel(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    loader = TBPTTLoader(
        ROOT_TRAIN, batch_size=BATCH_SIZE, chunk_len=CHUNK_LEN, max_row_groups=MAX_ROW_GROUPS, seed=seed
    )

    losses = []
    step_times = []
    hidden = None
    steps_done = 0
    for window in loader.epoch():
        if steps_done >= N_STEPS:
            break
        if window.is_first:
            hidden = model.init_hidden(window.features.shape[0], device=device)
        else:
            hidden = hidden.detach()

        features = torch.from_numpy(window.features).to(device)
        t0 = torch.from_numpy(window.t0).to(device)
        t1 = torch.from_numpy(window.t1).to(device)
        mask = torch.from_numpy(window.mask).to(device)

        if device.type == "mps":
            torch.mps.synchronize()
        t_start = time.perf_counter()

        pred, hidden = model(features, hidden)
        loss, _, _ = global_wp_loss(pred[..., 0], pred[..., 1], t0, t1, mask)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if device.type == "mps":
            torch.mps.synchronize()
        step_times.append(time.perf_counter() - t_start)
        losses.append(loss.item())
        steps_done += 1

    return step_times, losses


if __name__ == "__main__":
    print(f"mps available: {torch.backends.mps.is_available()}")
    print(f"config: batch={BATCH_SIZE} chunk_len={CHUNK_LEN} gru_hidden=128 gru_layers=2 steps={N_STEPS}")

    cpu_times, cpu_losses = run("cpu")
    cpu_mean = sum(cpu_times[2:]) / len(cpu_times[2:])  # drop 2 warmup steps
    print(f"\nCPU: mean {cpu_mean*1000:.1f} ms/step (excl. 2 warmup steps), "
          f"first loss {cpu_losses[0]:.4f}, last loss {cpu_losses[-1]:.4f}")

    if torch.backends.mps.is_available():
        mps_times, mps_losses = run("mps")
        mps_mean = sum(mps_times[2:]) / len(mps_times[2:])
        print(f"MPS: mean {mps_mean*1000:.1f} ms/step (excl. 2 warmup steps), "
              f"first loss {mps_losses[0]:.4f}, last loss {mps_losses[-1]:.4f}")
        speedup = cpu_mean / mps_mean
        print(f"\nspeedup (CPU/MPS): {speedup:.2f}x {'(MPS faster)' if speedup > 1 else '(CPU faster)'}")
        loss_diff = max(abs(a - b) for a, b in zip(cpu_losses, mps_losses))
        print(f"max |cpu_loss - mps_loss| across matching steps (same seed/data, expect small "
              f"numeric drift, not a match): {loss_diff:.4f}")
    else:
        print("MPS not available on this machine.")
