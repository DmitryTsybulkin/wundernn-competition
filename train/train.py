"""Main training loop: TBPTT over train.parquet, weighted-Pearson loss, wandb logging,
periodic fast validation, checkpointing. wandb is only ever imported here (never in
solution.py / the submission)."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train.data import TBPTTLoader
from train.eval import evaluate_fast, sweep_output_multiplier
from train.losses import chunk_wp_loss, global_wp_loss
from train.model import ModelConfig, SequenceModel

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser()
    # data
    p.add_argument("--train-path", default=str(ROOT / "datasets/train.parquet"))
    p.add_argument("--valid-path", default=str(ROOT / "datasets/valid.parquet"))
    p.add_argument("--max-row-groups", type=int, default=None, help="cap on train sequences used per epoch")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--chunk-len", type=int, default=2000)
    p.add_argument(
        "--prefetch", action="store_true",
        help="read the next batch of sequences on a background thread while training on the "
             "current one (doubles resident batch memory; does not change what is yielded)",
    )
    # model
    # nargs="+" (not "*"): omit the flag entirely for no trunk. "*" previously let
    # "--trunk-hidden \" immediately followed by another flag on the next line silently
    # parse as an empty list instead of erroring — every run before 2026-09-16 hit exactly
    # this and built the baseline's no-trunk architecture without noticing.
    p.add_argument("--trunk-hidden", type=int, nargs="+", default=[])
    p.add_argument("--arch", default="gru", choices=["gru", "tcn"])
    p.add_argument("--gru-hidden", type=int, default=128)
    p.add_argument("--gru-layers", type=int, default=2)
    p.add_argument("--tcn-channels", type=int, default=128)
    p.add_argument("--tcn-kernel-size", type=int, default=3)
    p.add_argument(
        "--tcn-layers", type=int, default=5,
        help="dilations 1,2,4,...,2^(n-1); receptive_field = 1 + (kernel_size-1)*(2^n - 1)",
    )
    p.add_argument("--reparam", action="store_true")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--use-skip", action="store_true", help="linear skip: raw features -> (t0,t1)")
    p.add_argument(
        "--skip-init", default="ridge", choices=["ridge", "zero"],
        help="'ridge' loads --skip-init-path (models/ridge_v1.npz) into the skip connection "
             "so the GRU only has to learn the residual; 'zero' lets it learn from scratch",
    )
    p.add_argument("--skip-init-path", default=str(ROOT / "models/ridge_v1.npz"))
    p.add_argument(
        "--loss-mode", default="global", choices=["global", "per_chunk"],
        help="'global' pools all rows in the batch into one weighted correlation per target, "
             "matching Global WP (utils.GlobalAccumulator). 'per_chunk' is the OLD per-sequence "
             "loss kept only for comparison — it equalizes/renormalizes per sequence, which "
             "Global WP does not reward (measured: costs -0.049 WP).",
    )
    # optim
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-schedule", default="none", choices=["none", "cosine"])
    p.add_argument("--lr-min-factor", type=float, default=0.05, help="cosine decays lr down to lr*this factor")
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument(
        "--max-steps", type=int, default=None,
        help="stop after this many optimizer steps (windows) in total, counted across resumes. "
             "When given, --epochs is raised as needed so it never ends the run early, and the "
             "cosine schedule (if any) is sized to this many steps.",
    )
    p.add_argument(
        "--ema-decay", type=float, default=0.98,
        help="EMA of weights used for eval/checkpointing, smooths noisy per-chunk-correlation "
             "gradient steps (set 0 to disable and eval the raw model)",
    )
    # eval
    p.add_argument("--valid-sequences", type=int, default=200)
    p.add_argument("--valid-every-windows", type=int, default=200)
    p.add_argument("--valid-batch-size", type=int, default=32)
    p.add_argument(
        "--calibrate-sequences", type=int, default=400,
        help="validation sequences used to pick the final output multiplier after training "
             "(Global WP is only scale-invariant before the metric's [-2,2] clip)",
    )
    p.add_argument(
        "--early-stop-patience", type=int, default=None,
        help="stop after this many consecutive evaluations with no wp_masked improvement "
             "(None disables early stopping)",
    )
    p.add_argument(
        "--max-seconds", type=float, default=None,
        help="stop training after this many seconds (checked at each eval), whichever of "
             "this or --early-stop-patience triggers first",
    )
    # checkpointing / resume (notes/TASK_gpu_prep.md)
    p.add_argument(
        "--checkpoint-every", type=int, default=200,
        help="write <out-dir>/ckpt_latest.pt (atomically) every N steps: model, EMA, optimizer, "
             "scheduler, step/epoch/elapsed, data position, RNG states, best-so-far. 0 disables.",
    )
    p.add_argument(
        "--resume", default=None,
        help="path to a ckpt_latest.pt to continue from, or 'auto' to continue from "
             "<out-dir>/ckpt_latest.pt (or <mirror-dir>/ckpt_latest.pt, whichever is further "
             "along) if one exists and start fresh otherwise",
    )
    p.add_argument(
        "--mirror-dir", default=None,
        help="second directory (e.g. on Google Drive) that ckpt_latest.pt, best.pt, last.pt, "
             "log.txt, config.json and summary.md are copied to after every checkpoint, so a "
             "wiped local disk does not lose the run. Failures to copy only warn.",
    )
    p.add_argument(
        "--tee-log", action="store_true",
        help="also append stdout to <out-dir>/log.txt (resume-safe; use instead of `| tee`)",
    )
    p.add_argument(
        "--smoke", action="store_true",
        help="~20-step end-to-end pipeline check: max-steps 20, eval + checkpoint every 10 steps, "
             "32 validation/calibration sequences",
    )
    p.add_argument("--test-crash-after-step", type=int, default=None, help=argparse.SUPPRESS)
    # infra
    p.add_argument("--device", default=None, choices=[None, "cpu", "mps", "cuda"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default=str(ROOT / "runs/dev"))
    p.add_argument("--run-name", default=None)
    # wandb
    p.add_argument("--wandb-mode", default="offline", choices=["online", "offline", "disabled"])
    p.add_argument("--wandb-project", default="Wundernn")
    p.add_argument("--wandb-entity", default="my-lore")
    return p.parse_args()


def pick_device(requested):
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    # MPS measured slower than CPU for this small-recurrent-net workload (kernel-launch
    # overhead dominates); default to CPU, opt into MPS explicitly with --device mps.
    return torch.device("cpu")


# --------------------------------------------------------------------------------------------
# infrastructure: atomic writes, log tee, mirror, wandb stub
# --------------------------------------------------------------------------------------------
def atomic_save(obj, path: Path):
    """torch.save to a temp file in the same directory, fsync, then rename over `path`, so an
    interrupted write can never leave a truncated/corrupt file at `path`."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)


def cpu_state(module):
    """State dict on the CPU, so best.pt/last.pt load anywhere (a GPU run's checkpoints are
    consumed on a Mac)."""
    return {k: v.detach().cpu() for k, v in module.state_dict().items()}


def atomic_write_text(path: Path, text: str):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class _Tee:
    def __init__(self, stream, path):
        self.stream = stream
        self.file = open(path, "a", buffering=1)

    def write(self, data):
        self.stream.write(data)
        self.file.write(data)

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def isatty(self):
        return False


class _NullWandb:
    """Stand-in when --wandb-mode disabled and wandb is not installed (e.g. a fresh Colab)."""

    class _Config:
        def update(self, *a, **k):
            pass

    config = _Config()

    def init(self, *a, **k):
        return None

    def log(self, *a, **k):
        pass

    def finish(self, *a, **k):
        pass


MIRROR_FILES = ("ckpt_latest.pt", "best.pt", "last.pt", "log.txt", "config.json", "summary.md")


def sync_to_mirror(out_dir: Path, mirror_dir: Path | None):
    if mirror_dir is None:
        return
    try:
        mirror_dir.mkdir(parents=True, exist_ok=True)
        for name in MIRROR_FILES:
            src = out_dir / name
            if src.exists():
                tmp = mirror_dir / (name + ".tmp")
                shutil.copyfile(src, tmp)
                os.replace(tmp, mirror_dir / name)
    except Exception as e:  # a flaky Drive mount must never kill the training run
        print(f"WARNING: mirror sync to {mirror_dir} failed: {e!r}", flush=True)


def _load_ckpt_if_valid(path: Path):
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        assert ckpt.get("format") == CKPT_FORMAT and "global_step" in ckpt
        return ckpt
    except Exception as e:
        print(f"WARNING: could not read checkpoint {path}: {e!r}", flush=True)
        return None


CKPT_FORMAT = 1
# Arguments that determine which data is consumed in which order / how the model is shaped:
# changing any of them across a resume would make "continue" meaningless.
RESUME_LOCKED_ARGS = (
    "batch_size", "chunk_len", "max_row_groups", "seed", "trunk_hidden", "arch", "gru_hidden",
    "gru_layers", "tcn_channels", "tcn_kernel_size", "tcn_layers", "reparam", "use_skip",
    "loss_mode", "lr", "lr_schedule", "ema_decay", "weight_decay", "grad_clip",
)


def find_resume_checkpoint(spec: str | None, out_dir: Path, mirror_dir: Path | None):
    """Returns (ckpt_dict, path) or (None, None). 'auto' = the furthest-along valid
    ckpt_latest.pt among out_dir and mirror_dir; anything else is an explicit path."""
    if spec is None:
        return None, None
    if spec != "auto":
        path = Path(spec)
        if not path.exists():
            raise FileNotFoundError(f"--resume {spec}: no such file")
        ckpt = _load_ckpt_if_valid(path)
        if ckpt is None:
            raise RuntimeError(f"--resume {spec}: not a readable checkpoint")
        return ckpt, path
    cands = []
    for d in [out_dir, mirror_dir]:
        if d is not None and (d / "ckpt_latest.pt").exists():
            ckpt = _load_ckpt_if_valid(d / "ckpt_latest.pt")
            if ckpt is not None:
                cands.append((ckpt["global_step"], ckpt, d / "ckpt_latest.pt"))
    if not cands:
        return None, None
    _, ckpt, path = max(cands, key=lambda c: c[0])
    return ckpt, path


def restore_rng(rng_state):
    random.setstate(rng_state["python"])
    np.random.set_state(rng_state["numpy"])
    torch.set_rng_state(rng_state["torch"].cpu())
    if rng_state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in rng_state["cuda"]])


def capture_rng():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def write_summary(out_dir: Path, args, n_params, n_valid, history, best, elapsed, finished, multiplier, resumes):
    """runs/<name>/summary.md as required by CLAUDE.md (5-10 lines)."""
    arch = (
        f"arch={args.arch} trunk={list(args.trunk_hidden)} "
        + (f"gru {args.gru_hidden}x{args.gru_layers}" if args.arch == "gru"
           else f"tcn ch={args.tcn_channels} k={args.tcn_kernel_size} layers={args.tcn_layers}")
        + f" skip={'ridge-init' if args.use_skip and args.skip_init == 'ridge' else args.use_skip}"
        + f" loss={args.loss_mode}"
    )
    last = history[-1] if history else None
    lines = [f"# {out_dir.name}", ""]
    lines.append(
        f"**Config**: {arch}; batch {args.batch_size}, chunk {args.chunk_len}, lr {args.lr:g} "
        f"({args.lr_schedule}), ema {args.ema_decay}, seed {args.seed}; {n_params:,} params; "
        f"train `{Path(args.train_path).name}`, in-run valid `{Path(args.valid_path).name}` "
        f"({n_valid} sequences). Full config in config.json."
    )
    if best["step"] is not None:
        b = best
        lines.append(
            f"**Best (in-run, {n_valid}-sequence subset — NOT the full-valid number)**: "
            f"wp_masked={b['wp_masked']:.6f} (t0={b['t0']:.4f} t1={b['t1']:.4f}), "
            f"wp_all_rows={b['wp_all_rows']:.6f}, ratio={b['ratio']:.3f}."
        )
        lines.append(f"**Peaked at**: step {b['step']}; last eval at step {last['step']} "
                     f"(wp_masked={last['wp_masked']:.6f}), {len(history)} evaluations.")
    else:
        lines.append("**Best**: no evaluation ran.")
    lines.append(
        f"**Wall clock**: {elapsed/60:.1f} min of training across {resumes + 1} launch(es) "
        f"({resumes} resume(s)); {'finished' if finished else 'INTERRUPTED — resume with --resume auto'}."
        + (f" Output multiplier (calibrated on the in-run subset): {multiplier}." if multiplier is not None else "")
    )
    if best["step"] is not None and history:
        recent = history[-min(3, len(history)):]
        trend = "still improving at the end" if best["step"] == last["step"] else (
            f"peaked {last['step'] - best['step']} steps before the end")
        lines.append(
            f"**What was learned**: (auto-generated) {trend}; last {len(recent)} evals "
            + ", ".join(f"{h['wp_masked']:.4f}" for h in recent)
            + ". Add the real takeaway by hand after the FULL-validation measurement."
        )
    atomic_write_text(out_dir / "summary.md", "\n".join(lines) + "\n")


def main():
    args = parse_args()
    if args.smoke:
        args.max_steps = args.max_steps or 20
        args.valid_every_windows = 10
        args.checkpoint_every = 10
        args.valid_sequences = min(args.valid_sequences, 32)
        args.calibrate_sequences = min(args.calibrate_sequences, 32)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mirror_dir = Path(args.mirror_dir) if args.mirror_dir else None

    # Fresh runtime after a disconnect: the local out_dir may be empty while the mirror (Drive)
    # holds the run. Pull its files back BEFORE the log is opened for append.
    if args.resume == "auto" and mirror_dir is not None and (mirror_dir / "ckpt_latest.pt").exists():
        mirror_ckpt = _load_ckpt_if_valid(mirror_dir / "ckpt_latest.pt")
        local_ckpt = _load_ckpt_if_valid(out_dir / "ckpt_latest.pt") if (out_dir / "ckpt_latest.pt").exists() else None
        if mirror_ckpt is not None and (local_ckpt is None or mirror_ckpt["global_step"] > local_ckpt["global_step"]):
            for name in MIRROR_FILES:
                if (mirror_dir / name).exists():
                    shutil.copyfile(mirror_dir / name, out_dir / name)
            print(f"restored run files from mirror {mirror_dir} (step {mirror_ckpt['global_step']})")
        del mirror_ckpt, local_ckpt
    if args.tee_log:
        sys.stdout = _Tee(sys.stdout, out_dir / "log.txt")
        print(f"\n===== launch {time.strftime('%Y-%m-%d %H:%M:%S')}  argv: {' '.join(sys.argv)} =====")

    torch.manual_seed(args.seed)
    device = pick_device(args.device)

    if args.wandb_mode == "disabled":
        try:
            import wandb
        except ImportError:
            wandb = _NullWandb()
    else:
        import wandb
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        name=args.run_name,
        config=vars(args),
        dir=str(out_dir),
    )

    model_cfg = ModelConfig(
        trunk_hidden=tuple(args.trunk_hidden),
        arch=args.arch,
        gru_hidden=args.gru_hidden,
        gru_layers=args.gru_layers,
        tcn_channels=args.tcn_channels,
        tcn_kernel_size=args.tcn_kernel_size,
        tcn_layers=args.tcn_layers,
        reparam=args.reparam,
        dropout=args.dropout,
        use_skip=args.use_skip,
    )
    model = SequenceModel(model_cfg)
    if args.use_skip and args.skip_init == "ridge":
        model.init_skip_from_ridge(args.skip_init_path)
        print(f"skip connection initialized from {args.skip_init_path}")
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    BASELINE_NO_TRUNK_PARAMS = 192_258
    if args.arch == "gru" and args.trunk_hidden and n_params == BASELINE_NO_TRUNK_PARAMS:
        raise RuntimeError(
            f"--trunk-hidden={args.trunk_hidden} was passed but the built model has exactly "
            f"{BASELINE_NO_TRUNK_PARAMS:,} params — the baseline's no-trunk architecture. The "
            f"trunk did not take effect (e.g. nargs='*' silently consumes zero values when "
            f"--trunk-hidden is immediately followed by another flag, such as a line "
            f"continuation like '--trunk-hidden \\' then '--lr ...' on the next line)."
        )
    arch_desc = (
        f"gru_hidden={model_cfg.gru_hidden} gru_layers={model_cfg.gru_layers}"
        if args.arch == "gru"
        else f"tcn_channels={model_cfg.tcn_channels} tcn_kernel_size={model_cfg.tcn_kernel_size} "
             f"tcn_layers={model_cfg.tcn_layers} receptive_field={model.temporal.receptive_field} rows"
    )
    print(
        f"model params: {n_params:,}  arch={args.arch}  trunk_hidden={model_cfg.trunk_hidden} "
        f"{arch_desc}  reparam={model_cfg.reparam}  device: {device}"
    )
    wandb.config.update({"n_params": n_params})

    loss_fn = global_wp_loss if args.loss_mode == "global" else chunk_wp_loss
    print(f"loss_mode: {args.loss_mode}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = TBPTTLoader(
        args.train_path,
        batch_size=args.batch_size,
        chunk_len=args.chunk_len,
        max_row_groups=args.max_row_groups,
        seed=args.seed,
        prefetch=args.prefetch,
    )
    print(f"train sequences/epoch: {loader.n_row_groups}  windows/epoch: {len(loader)}")
    if len(loader) == 0:
        raise RuntimeError(f"batch size {args.batch_size} > {loader.n_row_groups} training sequences: no batches")

    if args.max_steps is not None:
        # --epochs must never silently end a --max-steps run early (its default is 1).
        args.epochs = max(args.epochs, math.ceil(args.max_steps / len(loader)) + 1)
    total_steps = args.epochs * len(loader)
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)
    total_steps = max(1, total_steps)
    scheduler = None
    if args.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=total_steps, eta_min=args.lr * args.lr_min_factor
        )

    ema_model = None
    if args.ema_decay > 0:
        decay = args.ema_decay
        ema_avg_fn = lambda avg_p, cur_p, _n: decay * avg_p + (1 - decay) * cur_p
        ema_model = torch.optim.swa_utils.AveragedModel(model, avg_fn=ema_avg_fn)

    def eval_model():
        return ema_model.module if ema_model is not None else model

    # ---- training state (everything below is what a checkpoint must carry) -------------------
    best = {"wp": -1.0, "step": None, "wp_masked": None, "t0": None, "t1": None,
            "wp_all_rows": None, "ratio": None}
    history = []
    global_step = 0
    start_epoch = 0
    evals_since_improvement = 0
    elapsed_before = 0.0
    hidden = None
    finished = False
    resumes = 0
    resume_log = []

    ckpt, ckpt_path = find_resume_checkpoint(args.resume, out_dir, mirror_dir)
    if args.resume and ckpt is None:
        print(f"--resume {args.resume}: no checkpoint found, starting fresh")
    if ckpt is not None:
        saved_args = ckpt["args"]
        bad = [k for k in RESUME_LOCKED_ARGS if saved_args.get(k) != vars(args).get(k)]
        if bad:
            raise RuntimeError(
                "cannot resume: these arguments differ from the checkpointed run: "
                + ", ".join(f"{k} ({saved_args.get(k)!r} -> {vars(args).get(k)!r})" for k in bad)
            )
        model.load_state_dict(ckpt["state_dict"])
        opt.load_state_dict(ckpt["opt"])
        if scheduler is not None:
            scheduler.load_state_dict(ckpt["sched"])
        if ema_model is not None:
            ema_model.load_state_dict(ckpt["ema_state_dict"])
        loader.load_state_dict(ckpt["loader"])
        restore_rng(ckpt["rng"])
        global_step, start_epoch = ckpt["global_step"], ckpt["epoch"]
        elapsed_before = ckpt["elapsed"]
        best, history = ckpt["best"], ckpt["history"]
        evals_since_improvement = ckpt["evals_since_improvement"]
        finished = ckpt["finished"]
        resume_log = ckpt.get("resume_log", [])
        resumes = len(resume_log) + 1
        resume_log = resume_log + [{"step": global_step, "at": time.strftime("%Y-%m-%d %H:%M:%S")}]
        hidden = ckpt["hidden"].to(device) if ckpt["hidden"] is not None else None
        print(
            f"RESUMED from {ckpt_path}: step={global_step} epoch={start_epoch} "
            f"elapsed={elapsed_before:.0f}s best wp_masked={best['wp']:.6f} (step {best['step']}) "
            f"data position: batch {ckpt['loader']['next_batch']} window {ckpt['loader']['next_window']}"
            + ("  [run was already finished]" if finished else "")
        )
        del ckpt

    # notes/CLAUDE.md requires every run to write config.json with the full resolved config,
    # including parameter count.
    model_cfg_dict = {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(model_cfg).items()}
    atomic_write_text(
        out_dir / "config.json",
        json.dumps({"model_config": model_cfg_dict, "n_params": n_params, "train_args": vars(args),
                    "resume_log": resume_log}, indent=2),
    )

    t_launch = time.time()

    def elapsed_now():
        return elapsed_before + (time.time() - t_launch)

    def save_latest(epoch, is_finished):
        payload = {
            "format": CKPT_FORMAT,
            # plain dict, not the ModelConfig dataclass: ckpt_latest.pt must be loadable by any
            # process (e.g. the notebook's summary cell) without the repo on sys.path.
            "model_cfg": model_cfg_dict,
            "state_dict": model.state_dict(),
            "ema_state_dict": ema_model.state_dict() if ema_model is not None else None,
            "opt": opt.state_dict(),
            "sched": scheduler.state_dict() if scheduler is not None else None,
            "global_step": global_step,
            "epoch": epoch,
            "elapsed": elapsed_now(),
            "loader": loader.state_dict(),
            "hidden": hidden.detach().cpu() if hidden is not None else None,
            "rng": capture_rng(),
            "best": best,
            "history": history,
            "evals_since_improvement": evals_since_improvement,
            "finished": is_finished,
            "resume_log": resume_log,
            "args": vars(args),
        }
        t = time.time()
        atomic_save(payload, out_dir / "ckpt_latest.pt")
        size_mb = (out_dir / "ckpt_latest.pt").stat().st_size / 1e6
        sync_to_mirror(out_dir, mirror_dir)
        print(f"[ckpt] step {global_step} epoch {epoch} -> {out_dir / 'ckpt_latest.pt'} "
              f"({size_mb:.1f} MB, {time.time()-t:.1f}s)", flush=True)

    stop_signal = {"sig": None}

    def _on_signal(signum, _frame):
        stop_signal["sig"] = signum

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    def run_eval():
        nonlocal evals_since_improvement
        result = evaluate_fast(
            eval_model(), args.valid_path, device,
            max_sequences=args.valid_sequences, batch_size=args.valid_batch_size,
        )
        masked, all_rows = result["wp_masked"], result["wp_all_rows"]
        ratio = all_rows["weighted_pearson"] / masked["weighted_pearson"] if masked["weighted_pearson"] > 1e-8 else float("nan")
        elapsed = elapsed_now()
        print(
            f"[step {global_step}] loss={loss.item():.4f} lr={opt.param_groups[0]['lr']:.2e} "
            f"wp_masked={masked['weighted_pearson']:.4f} (t0={masked['t0']:.4f} t1={masked['t1']:.4f}) "
            f"wp_all_rows={all_rows['weighted_pearson']:.4f} ratio={ratio:.3f} (baseline: 0.713) "
            f"({elapsed:.0f}s elapsed)", flush=True
        )
        wandb.log(
            {
                "valid/wp_masked": masked["weighted_pearson"],
                "valid/wp_masked_t0": masked["t0"],
                "valid/wp_masked_t1": masked["t1"],
                "valid/wp_all_rows": all_rows["weighted_pearson"],
                "valid/wp_all_rows_t0": all_rows["t0"],
                "valid/wp_all_rows_t1": all_rows["t1"],
                "valid/mask_ratio": ratio,
            },
            step=global_step,
        )
        rec = {"step": global_step, "elapsed": elapsed, "wp_masked": masked["weighted_pearson"],
               "t0": masked["t0"], "t1": masked["t1"], "wp_all_rows": all_rows["weighted_pearson"], "ratio": ratio}
        history.append(rec)
        # Selection criterion is always the official masked metric (hard rule in
        # notes/CLAUDE.md); wp_all_rows is diagnostic only (mask-fitting check).
        if masked["weighted_pearson"] > best["wp"]:
            best.update({"wp": masked["weighted_pearson"], "step": global_step,
                         "wp_masked": masked["weighted_pearson"], "t0": masked["t0"], "t1": masked["t1"],
                         "wp_all_rows": all_rows["weighted_pearson"], "ratio": ratio})
            evals_since_improvement = 0
            atomic_save(
                {"model_cfg": model_cfg, "state_dict": cpu_state(eval_model()), "valid_wp": best["wp"]},
                out_dir / "best.pt",
            )
            print(f"  -> new best, saved to {out_dir / 'best.pt'}")
        else:
            evals_since_improvement += 1
        return elapsed

    loss = torch.zeros(())
    last_eval_step = history[-1]["step"] if history else None
    stop_training = finished
    cur_epoch = start_epoch
    speed_mark = {"step": global_step, "t": time.time()}

    if finished:
        print("checkpoint says the run already finished; skipping straight to the final stage")

    for epoch in range(start_epoch, args.epochs):
        if stop_training:
            break
        cur_epoch = epoch
        for window in loader.epoch():
            if window.is_first:
                hidden = model.init_hidden(window.features.shape[0], device=device)
            else:
                if hidden is None:
                    raise RuntimeError("resumed mid-batch but the checkpoint carries no hidden state")
                hidden = hidden.detach()

            features = torch.from_numpy(window.features).to(device)
            t0 = torch.from_numpy(window.t0).to(device)
            t1 = torch.from_numpy(window.t1).to(device)
            mask = torch.from_numpy(window.mask).to(device)

            model.train()
            pred, hidden = model(features, hidden)
            loss, wp0, wp1 = loss_fn(pred[..., 0], pred[..., 1], t0, t1, mask)

            opt.zero_grad()
            loss.backward()
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            if scheduler is not None:
                scheduler.step()
            if ema_model is not None:
                ema_model.update_parameters(model)

            global_step += 1
            wandb.log(
                {
                    "train/loss": loss.item(),
                    "train/wp_t0": wp0.item(),
                    "train/wp_t1": wp1.item(),
                    "train/wp_mean": -loss.item(),
                    "train/lr": opt.param_groups[0]["lr"],
                    "epoch": epoch,
                },
                step=global_step,
            )

            elapsed = elapsed_now()
            if global_step % args.valid_every_windows == 0:
                n_steps = global_step - speed_mark["step"]
                dt = time.time() - speed_mark["t"]
                print(f"[speed] steps {speed_mark['step']+1}-{global_step}: {dt/n_steps:.2f} s/step training only "
                      f"(data wait + fwd/bwd/opt/EMA), {n_steps*args.batch_size*args.chunk_len/dt:,.0f} rows/s", flush=True)
                elapsed = run_eval()
                last_eval_step = global_step
                if args.early_stop_patience is not None and evals_since_improvement >= args.early_stop_patience:
                    print(f"early stopping: {evals_since_improvement} evals without improvement")
                    stop_training = True
                if args.max_seconds is not None and elapsed >= args.max_seconds:
                    print(f"stopping: max_seconds={args.max_seconds} reached ({elapsed:.0f}s elapsed)")
                    stop_training = True
            if args.max_steps is not None and global_step >= args.max_steps:
                print(f"stopping: max_steps={args.max_steps} reached")
                stop_training = True
            if stop_training and last_eval_step != global_step:
                elapsed = run_eval()  # always end on an evaluated model
                last_eval_step = global_step

            if args.checkpoint_every and global_step % args.checkpoint_every == 0 and not stop_training:
                save_latest(cur_epoch, False)
            if global_step % args.valid_every_windows == 0:
                speed_mark.update(step=global_step, t=time.time())  # exclude eval/ckpt time from s/step
            if args.test_crash_after_step is not None and global_step >= args.test_crash_after_step:
                print(f"TEST HOOK: simulating a kill at step {global_step}", flush=True)
                os._exit(137)  # like SIGKILL: no cleanup, no final save
            if stop_signal["sig"] is not None and not stop_training:
                print(f"signal {stop_signal['sig']} received: checkpointing at step {global_step} and exiting", flush=True)
                save_latest(cur_epoch, False)
                write_summary(out_dir, args, n_params, args.valid_sequences, history, best, elapsed_now(), False, None, resumes)
                sync_to_mirror(out_dir, mirror_dir)
                sys.exit(128 + stop_signal["sig"])
            if stop_training:
                break

    # ---- final stage --------------------------------------------------------------------
    if not finished and global_step > 0 and last_eval_step != global_step:
        run_eval()  # data ran out between evaluations: still end on an evaluated model
    if not finished:
        atomic_save(
            {"model_cfg": model_cfg, "state_dict": cpu_state(eval_model()), "valid_wp": best["wp"]},
            out_dir / "last.pt",
        )
        save_latest(cur_epoch, True)  # finished=True: a re-run of the same command will not retrain
    print(f"done. best valid wp_masked (on {args.valid_sequences} sequences): {best['wp']:.4f}")

    # Calibrate the output scale: Global WP is only scale-invariant before the metric's
    # [-2,2] clip (notes/LOG.md 2026-09-17). Applies to both checkpoints; multiplier is
    # stored in the .pt file so export_onnx.py can bake it into the exported graph.
    multiplier = None
    for ckpt_name in ("best.pt", "last.pt"):
        ckpt_file = out_dir / ckpt_name
        if not ckpt_file.exists():
            continue
        ck = torch.load(ckpt_file, map_location="cpu", weights_only=False)
        if "output_multiplier" in ck:
            multiplier = ck["output_multiplier"] if ckpt_name == "best.pt" else multiplier
            continue
        calib_model = SequenceModel(ck["model_cfg"]).to(device)
        calib_model.load_state_dict(ck["state_dict"])
        best_c, table = sweep_output_multiplier(
            calib_model, args.valid_path, device,
            max_sequences=args.calibrate_sequences, batch_size=args.valid_batch_size,
        )
        uncalibrated = next(r["wp_masked"] for r in table if r["c"] == 1.0)
        calibrated = next(r["wp_masked"] for r in table if r["c"] == best_c)
        ck["output_multiplier"] = best_c
        atomic_save(ck, ckpt_file)
        print(
            f"calibrated {ckpt_name}: multiplier={best_c} "
            f"(wp_masked {uncalibrated:.4f} -> {calibrated:.4f}, "
            f"on {args.calibrate_sequences} sequences)"
        )
        if ckpt_name == "best.pt":
            multiplier = best_c
            wandb.log({
                "calib/best_multiplier": best_c,
                "calib/wp_masked_uncalibrated": uncalibrated,
                "calib/wp_masked_calibrated": calibrated,
            })

    write_summary(out_dir, args, n_params, args.valid_sequences, history, best, elapsed_now(), True, multiplier, resumes)
    sync_to_mirror(out_dir, mirror_dir)
    print(f"summary written to {out_dir / 'summary.md'}")
    wandb.finish()


if __name__ == "__main__":
    main()
