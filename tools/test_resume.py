"""notes/TASK_gpu_prep.md step 1: is resumable training actually bit-identical?

Runs the SAME 300-step training four ways (identical seed/data/config, CPU, fixed thread
count) and compares the final checkpoints:
  S  straight 300 steps
  P  straight 300 steps with --prefetch (must not change anything)
  A  the task's test: crash after step 150 (checkpoint-every 150, no final save), then
     `--resume auto` for the remaining 150
  B  the harder test: checkpoint-every 37, crash after step 190 => the last checkpoint is
     step 185 = MID-BATCH (window 5 of 10, hidden state carried) and MID-EPOCH; the resume
     redoes steps 186-190 and crosses the epoch boundaries at steps 200 and 300.
Data: 160 training sequences at batch 16 = 100 windows/epoch, so 300 steps = 3 epochs (the
epoch reshuffle is exercised). In-run validation every 50 steps on 40 sequences of
datasets/valid_subset.parquet.

usage: python3 tools/test_resume.py [work_dir]      (default: /private/tmp/.../resume_test)
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # older ckpt_latest.pt files pickle the ModelConfig dataclass
WORK = Path([a for a in sys.argv[1:] if not a.startswith('--')][0]) if [a for a in sys.argv[1:] if not a.startswith('--')] else Path("/private/tmp/claude-501/resume_test" + ("_quick" if "--quick" in sys.argv else ""))

QUICK = "--quick" in sys.argv   # 60-step regression version (used after late edits to train.py)
STEPS, EVAL_EVERY = (60, 20) if QUICK else (300, 50)
A_CKPT, A_CRASH, B_CKPT, B_CRASH = (30, 30, 13, 45) if QUICK else (150, 150, 37, 190)

BASE = [
    sys.executable, "-u", "-m", "train.train",
    "--train-path", "datasets/train_subset.parquet", "--valid-path", "datasets/valid_subset.parquet",
    "--trunk-hidden", "256", "128", "--use-skip", "--skip-init", "ridge", "--loss-mode", "global",
    "--batch-size", "16", "--chunk-len", "2000", "--max-row-groups", "160", "--lr", "3e-4",
    "--max-steps", str(STEPS), "--valid-every-windows", str(EVAL_EVERY), "--valid-sequences", "40",
    "--calibrate-sequences", "40", "--wandb-mode", "disabled", "--tee-log", "--seed", "0",
]
ENV = dict(os.environ, OMP_NUM_THREADS="3", MKL_NUM_THREADS="3")


def launch(name, extra):
    out = WORK / name
    cmd = BASE + ["--out-dir", str(out), "--run-name", name] + extra
    return subprocess.Popen(cmd, cwd=ROOT, env=ENV, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def max_abs_diff(a: dict, b: dict):
    return max((a[k].float() - b[k].float()).abs().max().item() for k in a)


def load(p):
    return torch.load(p, map_location="cpu", weights_only=False)


def scores(name):
    out = []
    for line in (WORK / name / "log.txt").read_text().splitlines():
        if line.startswith("[step") and "wp_masked" in line:
            out.append(line.split("loss=")[0] + "wp_masked=" + line.split("wp_masked=")[1].split(" ")[0])
    return out


def main():
    import shutil
    if "--compare-only" not in sys.argv:
        shutil.rmtree(WORK, ignore_errors=True)
        WORK.mkdir(parents=True)

    if "--compare-only" in sys.argv:
        return compare()
    # phase 1: everything that runs to a crash or to completion, in parallel
    procs = {
        "S": launch("S", ["--checkpoint-every", str(EVAL_EVERY)]),
        "P": launch("P", ["--checkpoint-every", str(EVAL_EVERY), "--prefetch"]),
        "A": launch("A", ["--checkpoint-every", str(A_CKPT), "--test-crash-after-step", str(A_CRASH)]),
        "B": launch("B", ["--checkpoint-every", str(B_CKPT), "--test-crash-after-step", str(B_CRASH)]),
    }
    for k, p in procs.items():
        rc = p.wait()
        print(f"phase 1: {k} exited with code {rc} ({'crash hook' if rc == 137 else 'completed' if rc == 0 else 'UNEXPECTED'})", flush=True)

    # phase 2: resume the two crashed runs with the identical command (+ --resume auto)
    assert not (WORK / "A" / "last.pt").exists() and not (WORK / "B" / "last.pt").exists(), "crashed runs must not have finished"
    resumed = {
        "A": launch("A", ["--checkpoint-every", str(A_CKPT), "--resume", "auto"]),
        "B": launch("B", ["--checkpoint-every", str(B_CKPT), "--resume", "auto"]),
    }
    for k, p in resumed.items():
        print(f"phase 2: {k} resumed run exited with code {p.wait()}", flush=True)

    compare()


def compare():
    ref = load(WORK / "S" / "ckpt_latest.pt")
    print(f"\nreference S: step {ref['global_step']} epoch {ref['epoch']} finished={ref['finished']}")
    for k in ("P", "A", "B"):
        c = load(WORK / k / "ckpt_latest.pt")
        opt_diff = max(
            (ref["opt"]["state"][i][n] - c["opt"]["state"][i][n]).abs().max().item()
            for i in ref["opt"]["state"] for n in ("exp_avg", "exp_avg_sq")
        )
        print(
            f"{k} vs S: step {c['global_step']} epoch {c['epoch']} finished={c['finished']} | "
            f"max|dparam| raw={max_abs_diff(ref['state_dict'], c['state_dict']):.3e} "
            f"ema={max_abs_diff(ref['ema_state_dict'], c['ema_state_dict']):.3e} "
            f"adam-moments={opt_diff:.3e} | loader pos equal={ref['loader']['next_batch'] == c['loader']['next_batch'] and ref['loader']['next_window'] == c['loader']['next_window']}"
        )
    print("\nvalidation scores at each eval (wp_masked on 40 seqs):")
    for k in ("S", "P", "A", "B"):
        print(f"  {k}: " + " | ".join(s.split("] ")[0][1:] + " " + s.split("wp_masked=")[1] for s in scores(k)))
    for k in ("S", "P", "A", "B"):
        print(f"  {k} best.pt valid_wp={load(WORK / k / 'best.pt')['valid_wp']:.6f}  last.pt vs S max|dparam|="
              f"{max_abs_diff(load(WORK / 'S' / 'last.pt')['state_dict'], load(WORK / k / 'last.pt')['state_dict']):.3e}")


if __name__ == "__main__":
    main()
