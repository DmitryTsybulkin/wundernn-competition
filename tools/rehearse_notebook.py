"""notes/TASK_gpu_prep.md step 5: rehearse colab/train_wnn.ipynb locally, with zero manual steps.

Executes the notebook's REAL code cells top to bottom (not a re-implementation), with only the
environment swapped: the "private GitHub repo" is a local bare repo holding a snapshot of the
working tree, "Drive" is a temp folder with small train/valid parquet slices, DEVICE='cpu'
(so the GPU check and google.colab imports are skipped), batch 16. Two passes:
  1. SMOKE=True  -> kill at step 10 on purpose, resume, finish, calibrate, summary  (SMOKE OK)
  2. SMOKE=False -> a short real-mode run (--max-steps 20 via EXTRA), then the commit+push cell,
     then verifies the files really arrived in the bare "remote" and that nothing large was pushed.
  3. Re-executes the train cell with the finished run to show that re-running is idempotent
     (resumes, sees finished, does not retrain).

usage: python3 tools/rehearse_notebook.py [work_dir]
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
WORK = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/private/tmp/claude-501/nb_rehearsal")


def sh(*a, cwd=None):
    r = subprocess.run(a, cwd=cwd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"{a}: {r.stderr}")
    return r.stdout.strip()


def small_parquet(src, dst, n):
    pf = pq.ParquetFile(src)
    w = None
    for i in range(n):
        t = pf.read_row_group(i)
        w = w or pq.ParquetWriter(dst, t.schema)
        w.write_table(t, row_group_size=20_000)
    w.close()


def run_cells(nb, patches, skip=()):
    ns = {"__name__": "__main__"}
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code" or i in skip:
            continue
        src = "".join(c["source"])
        for old, new in patches.items():
            src = src.replace(old, new)
        print(f"\n######## notebook cell {i} ########", flush=True)
        exec(compile(src, f"cell{i}", "exec"), ns)
    return ns


def main():
    shutil.rmtree(WORK, ignore_errors=True)
    (WORK / "drive").mkdir(parents=True)
    # snapshot of the working tree as the "pushed" repo (only what the clone would need)
    src = WORK / "src"
    for rel in ("train", "colab", "utils.py", "models", "requirements.txt", ".gitignore"):
        s = ROOT / rel
        (shutil.copytree(s, src / rel, ignore=shutil.ignore_patterns("__pycache__")) if s.is_dir()
         else (src.mkdir(exist_ok=True), shutil.copy(s, src / rel)))
    sh("git", "init", "-q", "-b", "main", cwd=src)
    sh("git", "add", ".", cwd=src)
    sh("git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "snapshot for rehearsal", cwd=src)
    sh("git", "clone", "-q", "--bare", str(src), str(WORK / "remote.git"))
    small_parquet(ROOT / "datasets/train_subset.parquet", WORK / "drive/train_subset.parquet", 48)
    small_parquet(ROOT / "datasets/valid_subset.parquet", WORK / "drive/valid_subset.parquet", 40)

    nb = json.load(open(ROOT / "colab/train_wnn.ipynb"))
    base = {
        'REPO_URL  = "https://github.com/DmitryTsybulkin/wundernn-competition.git"': f'REPO_URL  = "{WORK / "remote.git"}"',
        'DRIVE_DIR = "/content/drive/MyDrive/wnn"': f'DRIVE_DIR = "{WORK / "drive"}"',
        'WORK_DIR = "/content"': f'WORK_DIR = "{WORK / "content"}"',
        'DEVICE     = "cuda"': 'DEVICE     = "cpu"',
        "BATCH_SIZE = 256": "BATCH_SIZE = 16",
    }
    (WORK / "content").mkdir()
    print("=================== PASS 1: SMOKE=True ===================")
    ns = run_cells(nb, base)
    print("\n=================== PASS 2: SMOKE=False (short real-mode run) ===================")
    real = dict(base, **{
        "SMOKE      = True": "SMOKE      = False",
        "EXTRA      = []": 'EXTRA      = ["--max-steps", "20", "--valid-every-windows", "10", "--checkpoint-every", "10", "--valid-sequences", "16", "--calibrate-sequences", "16"]',
    })
    ns = run_cells(nb, real)
    print("\n=================== VERIFY: what reached the 'remote' ===================")
    files = sh("git", "ls-tree", "-r", "--long", "HEAD", "--", "runs", cwd=WORK / "remote.git")
    print(files)
    need = {"summary.md", "log.txt", "config.json", "best.pt", "last.pt", "ckpt_latest.pt"}
    got = {l.split()[-1].split("/")[-1] for l in files.splitlines()}
    assert need <= got, f"missing from remote: {need - got}"
    sizes = [int(l.split()[3]) for l in files.splitlines()]
    assert max(sizes) < 10_000_000, "something large was pushed"
    assert "askpass" not in files
    print("push verified: all six files present, largest", max(sizes), "bytes")
    print("\n=================== PASS 3: re-run the train cell on the finished run ===================")
    train_cell = [i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code" and "launcher exit code" in "".join(c["source"])][0]
    src_cell = "".join(nb["cells"][train_cell]["source"])
    for old, new in real.items():
        src_cell = src_cell.replace(old, new)
    exec(compile(src_cell, "train_cell", "exec"), ns)
    log = (Path(ns["REPO"]) / "runs" / ns["RUN_NAME"] / "log.txt").read_text()
    assert "already finished" in log, "re-run did not detect the finished run"
    print("\nREHEARSAL OK")


if __name__ == "__main__":
    main()
