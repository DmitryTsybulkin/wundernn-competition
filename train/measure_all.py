"""Step 5 of notes/TASK.md: re-measure all six checkpoints under Global WP on the full
1873-sequence validation set, reporting both wp_masked and wp_all_rows for each."""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train.eval import evaluate_fast
from train.model import SequenceModel

ROOT = Path(__file__).resolve().parents[1]
VALID_PATH = ROOT / "datasets/valid.parquet"

CHECKPOINTS = [
    ("stage5_ema_diag", "best.pt"),
    ("stage5_ema_diag", "last.pt"),
    ("stage6_full_ema", "best.pt"),
    ("stage6_full_ema", "last.pt"),
    ("stage7_ema_cosine", "best.pt"),
    ("stage7_ema_cosine", "last.pt"),
]

device = torch.device("cpu")
results = {}

for run_name, ckpt_name in CHECKPOINTS:
    path = ROOT / "runs" / run_name / ckpt_name
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = SequenceModel(ckpt["model_cfg"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    t0 = time.time()
    result = evaluate_fast(model, VALID_PATH, device, max_sequences=None, batch_size=32)
    elapsed = time.time() - t0

    masked, all_rows = result["wp_masked"], result["wp_all_rows"]
    key = f"{run_name}/{ckpt_name}"
    results[key] = {
        "wp_masked": masked["weighted_pearson"],
        "wp_masked_t0": masked["t0"],
        "wp_masked_t1": masked["t1"],
        "wp_all_rows": all_rows["weighted_pearson"],
        "wp_all_rows_t0": all_rows["t0"],
        "wp_all_rows_t1": all_rows["t1"],
        "old_valid_wp_metadata": ckpt.get("valid_wp"),
        "elapsed_s": elapsed,
    }
    print(
        f"{key}: wp_masked={masked['weighted_pearson']:.6f} "
        f"(t0={masked['t0']:.6f} t1={masked['t1']:.6f})  "
        f"wp_all_rows={all_rows['weighted_pearson']:.6f} "
        f"(t0={all_rows['t0']:.6f} t1={all_rows['t1']:.6f})  "
        f"[{elapsed:.0f}s]",
        flush=True,
    )

print("\n" + json.dumps(results, indent=2))
