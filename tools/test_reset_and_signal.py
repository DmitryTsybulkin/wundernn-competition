"""notes/TASK_gpu_prep.md step 1/5 extras: the two interruptions a Colab session really has.
  (a) graceful stop: SIGTERM mid-run -> train.py checkpoints at the next step boundary and exits 143
  (b) runtime reset: the LOCAL run dir is wiped (a fresh /content) while the mirror (Google
      Drive) survives -> `--resume auto` restores from the mirror and finishes.
Small slices of the real subsets, CPU, batch 16, ~2 minutes. Everything happens in a temp dir.
usage: python3 tools/test_reset_and_signal.py
"""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
W = Path("/private/tmp/claude-501/reset_test")


def small(src, dst, n):
    pf = pq.ParquetFile(src)
    w = None
    for i in range(n):
        t = pf.read_row_group(i)
        w = w or pq.ParquetWriter(dst, t.schema)
        w.write_table(t, row_group_size=20_000)
    w.close()


def cmd(extra=()):
    return [sys.executable, "-u", "-m", "train.train", "--train-path", str(W / "train.parquet"), "--valid-path", str(W / "valid.parquet"),
            "--trunk-hidden", "256", "128", "--use-skip", "--batch-size", "16", "--max-steps", "40", "--valid-every-windows", "10",
            "--checkpoint-every", "10", "--valid-sequences", "16", "--calibrate-sequences", "16", "--wandb-mode", "disabled",
            "--tee-log", "--resume", "auto", "--out-dir", str(W / "local"), "--mirror-dir", str(W / "mirror"), *extra]


def main():
    shutil.rmtree(W, ignore_errors=True)
    W.mkdir(parents=True)
    small(ROOT / "datasets/train_subset.parquet", W / "train.parquet", 48)
    small(ROOT / "datasets/valid_subset.parquet", W / "valid.parquet", 16)

    print("=== (a) SIGTERM mid-run ===", flush=True)
    p = subprocess.Popen(cmd(), cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = []
    for line in p.stdout:
        lines.append(line)
        if line.startswith("[ckpt] step 20"):       # a checkpoint exists; terminate a bit later, mid-window
            time.sleep(3)
            p.send_signal(signal.SIGTERM)
            break
    rest = p.stdout.read()
    rc = p.wait()
    text = "".join(lines) + rest
    print("\n".join(l for l in text.splitlines() if "signal" in l or "[ckpt]" in l or "summary" in l))
    print(f"exit code {rc} (expected 143)")
    assert rc == 143 and "checkpointing at step" in text
    ck = torch.load(W / "local" / "ckpt_latest.pt", map_location="cpu", weights_only=False)
    step_a = ck["global_step"]
    print(f"ckpt_latest.pt after SIGTERM: step {step_a}, finished={ck['finished']}")
    assert not ck["finished"] and step_a >= 20

    print("\n=== (b) local disk wiped, mirror survives ===", flush=True)
    shutil.rmtree(W / "local")
    assert not (W / "local").exists() and (W / "mirror" / "ckpt_latest.pt").exists()
    out = subprocess.run(cmd(), cwd=ROOT, capture_output=True, text=True).stdout
    print("\n".join(l for l in out.splitlines() if "restored" in l or "RESUMED" in l or "stopping" in l or "summary" in l or "calibrated" in l))
    assert "restored run files from mirror" in out and "RESUMED from" in out and "max_steps=40 reached" in out
    ck = torch.load(W / "local" / "ckpt_latest.pt", map_location="cpu", weights_only=False)
    assert ck["finished"] and ck["global_step"] == 40
    for n in ("summary.md", "best.pt", "last.pt", "log.txt", "config.json"):
        assert (W / "local" / n).exists() and (W / "mirror" / n).exists(), n
    log = (W / "local" / "log.txt").read_text()
    assert log.count("===== launch") == 2, "log.txt should hold both launches (restored from the mirror, then appended)"
    print(f"resumed from step {step_a} after the wipe -> finished at step 40; log.txt holds both launches\nRESET+SIGNAL OK")


if __name__ == "__main__":
    main()
