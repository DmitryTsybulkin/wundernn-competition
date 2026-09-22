"""Single source of truth for the GPU training command (notes/TASK_gpu_prep.md).

Used by colab/train_wnn.ipynb AND by the local CPU dry run, so what is rehearsed locally is
exactly what runs on the paid GPU (only --device / paths differ).

    python3 colab/launch.py --run-name 05_gru_gpu --train-path P --valid-path P \
        --out-dir runs/05_gru_gpu --mirror-dir /content/drive/MyDrive/wnn/runs/05_gru_gpu \
        --device cuda [--batch-size 256] [--lr 6e-4] [--max-hours 3] [--smoke] [--print-only]

Normal mode: one `train.train ... --resume auto` process. Re-running the identical command after
a crash / disconnect / runtime reset continues from ckpt_latest.pt (local or mirror).

--smoke: ~20 steps end to end through the SAME entry point with the same model/batch/lr flags,
under a fresh timestamped run name: phase 1 is killed on purpose (os._exit(137), like SIGKILL) at
step 10 right after a checkpoint, phase 2 is the identical command with --resume auto and must
resume, finish, calibrate and write summary.md. Exits non-zero, loudly, if any of that fails.
"""
from __future__ import annotations

import argparse
import math
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Recipe of runs/02_trunk_global_full (the model that was still improving when its wall clock
# stopped it) — model, loss, EMA, chunk length unchanged. Only batch/lr/budget differ.
BASE_BATCH, BASE_LR = 64, 3e-4


def scaled_lr(batch_size: int) -> float:
    """Square-root LR scaling from the batch-64 / 3e-4 that 02_trunk_global_full used. A
    starting rule only — NOT tuned (notes/LOG.md 2026-09-21)."""
    return float(f"{BASE_LR * math.sqrt(batch_size / BASE_BATCH):.2g}")


def build_command(
    *, run_name, train_path, valid_path, out_dir, mirror_dir, device, batch_size=256, lr=None,
    max_hours=3.0, smoke=False, crash_after_step=None, extra=(), python=sys.executable,
):
    lr = scaled_lr(batch_size) if lr is None else lr
    cmd = [
        python, "-u", "-m", "train.train",
        "--arch", "gru", "--trunk-hidden", "256", "128", "--gru-hidden", "128", "--gru-layers", "2",
        "--use-skip", "--skip-init", "ridge", "--loss-mode", "global",
        "--batch-size", str(batch_size), "--chunk-len", "2000", "--lr", str(lr),
        "--ema-decay", "0.98", "--seed", "0",
        # --epochs defaults to 1 (a footgun: run 01 stopped after one pass). The run is bounded by
        # --max-seconds (training time, summed across resumes) and early stopping instead.
        "--epochs", "100000",
        "--max-seconds", str(int(max_hours * 3600)), "--early-stop-patience", "8",
        "--valid-every-windows", "200", "--valid-sequences", "500", "--valid-batch-size", "32",
        "--calibrate-sequences", "500",
        "--checkpoint-every", "200", "--resume", "auto", "--tee-log", "--prefetch",
        "--train-path", str(train_path), "--valid-path", str(valid_path),
        "--out-dir", str(out_dir), "--mirror-dir", str(mirror_dir), "--run-name", run_name,
        "--device", device, "--wandb-mode", "disabled",
    ]
    if smoke:
        cmd.append("--smoke")
    if crash_after_step is not None:
        cmd += ["--test-crash-after-step", str(crash_after_step)]
    cmd += list(extra)  # advanced: appended last, so a repeated flag overrides the defaults above
    return cmd


def run(cmd) -> int:
    print("$ " + " ".join(cmd), flush=True)
    p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        for line in p.stdout:
            print(line, end="", flush=True)
    except KeyboardInterrupt:  # notebook "interrupt": let train.py checkpoint before exiting
        p.terminate()
        for line in p.stdout:
            print(line, end="", flush=True)
    return p.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--train-path", required=True)
    ap.add_argument("--valid-path", required=True)
    ap.add_argument("--out-dir", required=True, help="run directory (runs/<name>); smoke mode appends a suffix to the name")
    ap.add_argument("--mirror-dir", required=True, help="persistent copy (Google Drive) so a wiped runtime can resume")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=None, help="default: sqrt-scaled from 3e-4 at batch 64")
    ap.add_argument("--max-hours", type=float, default=3.0, help="training-time budget across all resumes")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--print-only", action="store_true", help="print the command(s) and exit")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                    help="everything after --extra is appended verbatim to the train.train command")
    a = ap.parse_args()

    if a.device == "cuda":
        import torch
        if not torch.cuda.is_available():
            sys.exit("ABORT: --device cuda but torch.cuda.is_available() is False — this is not a GPU runtime.")

    run_name, out_dir, mirror_dir = a.run_name, Path(a.out_dir), Path(a.mirror_dir)
    if a.smoke:
        suffix = time.strftime("smoke_%H%M%S")
        run_name = f"{suffix}_{a.run_name}"
        out_dir, mirror_dir = out_dir.with_name(run_name), mirror_dir.with_name(run_name)
    common = dict(run_name=run_name, train_path=a.train_path, valid_path=a.valid_path, out_dir=out_dir,
                  mirror_dir=mirror_dir, device=a.device, batch_size=a.batch_size, lr=a.lr,
                  max_hours=a.max_hours, smoke=a.smoke, extra=a.extra)

    if not a.smoke:
        cmd = build_command(**common)
        if a.print_only:
            print(" ".join(cmd))
            return
        sys.exit(run(cmd))

    phase1 = build_command(**common, crash_after_step=10)
    phase2 = build_command(**common)
    if a.print_only:
        print("phase 1 (killed on purpose):\n" + " ".join(phase1) + "\nphase 2 (resume):\n" + " ".join(phase2))
        return
    print(f"=== SMOKE: {run_name}: phase 1, killed at step 10 on purpose ===", flush=True)
    rc1 = run(phase1)
    print(f"=== SMOKE: phase 1 exited with {rc1} (137 expected) ===", flush=True)
    print("=== SMOKE: phase 2, identical command, --resume auto ===", flush=True)
    rc2 = run(phase2)
    log = (out_dir / "log.txt").read_text() if (out_dir / "log.txt").exists() else ""
    checks = {
        "phase 1 died at the kill hook (137)": rc1 == 137,
        "phase 2 exited 0": rc2 == 0,
        "phase 2 printed RESUMED from step 10": "RESUMED from" in log and "step=10" in log,
        "finished at step 20": "stopping: max_steps=20 reached" in log,
        **{f"{n} exists locally": (out_dir / n).exists() for n in
           ("summary.md", "log.txt", "config.json", "best.pt", "last.pt", "ckpt_latest.pt")},
        "mirror has ckpt_latest.pt + summary.md": (mirror_dir / "ckpt_latest.pt").exists() and (mirror_dir / "summary.md").exists(),
    }
    for k, v in checks.items():
        print(f"  [{'ok' if v else 'FAIL'}] {k}")
    if not all(checks.values()):
        sys.exit("SMOKE FAILED — do not start the long run.")
    print("SMOKE OK — the full path (train -> checkpoint -> kill -> resume -> finish -> calibrate -> summary) works.")


if __name__ == "__main__":
    main()
