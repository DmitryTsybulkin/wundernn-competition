"""notes/TASK.md step 4: a Colab-sized training subset. 2,500 sequences sampled
uniformly at random from the 10,607 in train.parquet (fixed seed), features stored as
float16 (already clipped to roughly +/-2.4, so float16 — ~3 decimal digits, range up to
65504 — is lossless enough for this range), targets/masks kept float32. Same schema and
row-group-per-sequence layout as the original, so train/data.py's loader works unchanged
(it reads columns then .astype(np.float32) immediately, transparently upcasting).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "datasets/train.parquet"
OUT_PATH = ROOT / "datasets/train_subset.parquet"

SEED = 42
N_SUBSET = 2500


def main():
    pf = pq.ParquetFile(TRAIN_PATH)
    rng = np.random.default_rng(SEED)
    chosen = np.sort(rng.choice(pf.num_row_groups, size=N_SUBSET, replace=False))
    print(f"sampling {N_SUBSET} of {pf.num_row_groups} sequences, seed={SEED}")

    feature_set = set(FEATURE_COLUMNS)
    writer = None
    t0 = time.time()
    for i, rg in enumerate(chosen):
        table = pf.read_row_group(int(rg), use_threads=False)
        arrays, names = [], []
        for name in table.column_names:
            col = table.column(name)
            if name in feature_set:
                arr = col.to_numpy().astype(np.float16)
                arrays.append(pa.array(arr, type=pa.float16()))
            else:
                arrays.append(col)
            names.append(name)
        new_table = pa.table(arrays, names=names)
        if writer is None:
            writer = pq.ParquetWriter(OUT_PATH, new_table.schema)
        writer.write_table(new_table, row_group_size=20_000)
        if (i + 1) % 500 == 0:
            print(f"{i+1}/{N_SUBSET} sequences written, {time.time()-t0:.0f}s elapsed", flush=True)
    writer.close()

    size_mb = OUT_PATH.stat().st_size / 1e6
    print(f"wrote {OUT_PATH} ({size_mb:.0f} MB) in {time.time()-t0:.0f}s")
    print(f"seed={SEED}, row groups (sequence indices in original file): saved implicitly via seq_ix column")


if __name__ == "__main__":
    main()
