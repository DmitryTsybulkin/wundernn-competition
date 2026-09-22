"""notes/TASK_gpu_prep.md step 4: a Colab-sized validation subset for in-run monitoring.
500 sequences sampled uniformly at random from the 1,873 in valid.parquet (fixed, recorded
seed), features stored as float16 like datasets/train_subset.parquet (features are clipped to
roughly +/-2.4, so float16 is lossless enough), targets and the need_prediction / is_scored
columns kept as-is. Same schema and row-group-per-sequence layout as valid.parquet, so
train/eval.py reads it unchanged. The chosen original row-group indices are written to
datasets/valid_subset_indices.json (and printed) so the subset can be reproduced or scored
against the cached full-validation prediction arrays.

Final numbers are ALWAYS measured on the full valid.parquet locally; this subset is only for
in-run monitoring (see tools/valid_subset_offset.py for the measured subset-vs-full offset).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
VALID_PATH = ROOT / "datasets/valid.parquet"
OUT_PATH = ROOT / "datasets/valid_subset.parquet"
INDEX_PATH = ROOT / "datasets/valid_subset_indices.json"

SEED = 42
N_SUBSET = 500


def main():
    pf = pq.ParquetFile(VALID_PATH)
    rng = np.random.default_rng(SEED)
    chosen = np.sort(rng.choice(pf.num_row_groups, size=N_SUBSET, replace=False))
    print(f"sampling {N_SUBSET} of {pf.num_row_groups} validation sequences, seed={SEED}", flush=True)

    feature_set = set(FEATURE_COLUMNS)
    writer = None
    t0 = time.time()
    for i, rg in enumerate(chosen):
        table = pf.read_row_group(int(rg), use_threads=False)
        arrays, names = [], []
        for name in table.column_names:
            col = table.column(name)
            if name in feature_set:
                arrays.append(pa.array(col.to_numpy().astype(np.float16), type=pa.float16()))
            else:
                arrays.append(col)
            names.append(name)
        new_table = pa.table(arrays, names=names)
        if writer is None:
            writer = pq.ParquetWriter(OUT_PATH, new_table.schema)
        writer.write_table(new_table, row_group_size=20_000)
        if (i + 1) % 100 == 0:
            print(f"{i+1}/{N_SUBSET} sequences written, {time.time()-t0:.0f}s elapsed", flush=True)
    writer.close()

    with open(INDEX_PATH, "w") as f:
        json.dump({"seed": SEED, "n": N_SUBSET, "source": "datasets/valid.parquet",
                   "row_groups": [int(x) for x in chosen]}, f)
    print(f"wrote {OUT_PATH} ({OUT_PATH.stat().st_size / 1e6:.0f} MB) in {time.time()-t0:.0f}s; "
          f"indices -> {INDEX_PATH}")


if __name__ == "__main__":
    main()
