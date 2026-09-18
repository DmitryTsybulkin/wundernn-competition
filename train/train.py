"""Main training loop: TBPTT over train.parquet, weighted-Pearson loss, wandb logging,
periodic fast validation, checkpointing. wandb is only ever imported here (never in
solution.py / the submission)."""
from __future__ import annotations

import argparse
import json
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
    # model
    # nargs="+" (not "*"): omit the flag entirely for no trunk. "*" previously let
    # "--trunk-hidden \" immediately followed by another flag on the next line silently
    # parse as an empty list instead of erroring — every run before 2026-09-16 hit exactly
    # this and built the baseline's no-trunk architecture without noticing.
    p.add_argument("--trunk-hidden", type=int, nargs="+", default=[])
    p.add_argument("--gru-hidden", type=int, default=128)
    p.add_argument("--gru-layers", type=int, default=2)
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


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
        gru_hidden=args.gru_hidden,
        gru_layers=args.gru_layers,
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
    if args.trunk_hidden and n_params == BASELINE_NO_TRUNK_PARAMS:
        raise RuntimeError(
            f"--trunk-hidden={args.trunk_hidden} was passed but the built model has exactly "
            f"{BASELINE_NO_TRUNK_PARAMS:,} params — the baseline's no-trunk architecture. The "
            f"trunk did not take effect (e.g. nargs='*' silently consumes zero values when "
            f"--trunk-hidden is immediately followed by another flag, such as a line "
            f"continuation like '--trunk-hidden \\' then '--lr ...' on the next line)."
        )
    print(
        f"model params: {n_params:,}  trunk_hidden={model_cfg.trunk_hidden} "
        f"gru_hidden={model_cfg.gru_hidden} gru_layers={model_cfg.gru_layers} "
        f"reparam={model_cfg.reparam}  device: {device}"
    )
    wandb.config.update({"n_params": n_params})

    # notes/CLAUDE.md requires every run to write config.json with the full resolved config,
    # including parameter count.
    model_cfg_dict = {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(model_cfg).items()}
    with open(out_dir / "config.json", "w") as f:
        json.dump({"model_config": model_cfg_dict, "n_params": n_params, "train_args": vars(args)}, f, indent=2)

    loss_fn = global_wp_loss if args.loss_mode == "global" else chunk_wp_loss
    print(f"loss_mode: {args.loss_mode}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = TBPTTLoader(
        args.train_path,
        batch_size=args.batch_size,
        chunk_len=args.chunk_len,
        max_row_groups=args.max_row_groups,
        seed=args.seed,
    )
    print(f"train sequences/epoch: {loader.n_row_groups}  windows/epoch: {len(loader)}")

    total_steps = max(1, args.epochs * len(loader))
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

    best_wp = -1.0
    global_step = 0
    evals_since_improvement = 0
    stop_training = False
    t_start = time.time()

    for epoch in range(args.epochs):
        if stop_training:
            break
        hidden = None
        for window in loader.epoch():
            if window.is_first:
                hidden = model.init_hidden(window.features.shape[0], device=device)
            else:
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

            if global_step % args.valid_every_windows == 0:
                result = evaluate_fast(
                    eval_model(), args.valid_path, device,
                    max_sequences=args.valid_sequences, batch_size=args.valid_batch_size,
                )
                masked, all_rows = result["wp_masked"], result["wp_all_rows"]
                ratio = all_rows["weighted_pearson"] / masked["weighted_pearson"] if masked["weighted_pearson"] > 1e-8 else float("nan")
                elapsed = time.time() - t_start
                print(
                    f"[step {global_step}] loss={loss.item():.4f} lr={opt.param_groups[0]['lr']:.2e} "
                    f"wp_masked={masked['weighted_pearson']:.4f} (t0={masked['t0']:.4f} t1={masked['t1']:.4f}) "
                    f"wp_all_rows={all_rows['weighted_pearson']:.4f} ratio={ratio:.3f} (baseline: 0.713) "
                    f"({elapsed:.0f}s elapsed)"
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
                # Selection criterion is always the official masked metric (hard rule in
                # notes/CLAUDE.md); wp_all_rows is diagnostic only (mask-fitting check).
                if masked["weighted_pearson"] > best_wp:
                    best_wp = masked["weighted_pearson"]
                    evals_since_improvement = 0
                    torch.save(
                        {"model_cfg": model_cfg, "state_dict": eval_model().state_dict(), "valid_wp": best_wp},
                        out_dir / "best.pt",
                    )
                    print(f"  -> new best, saved to {out_dir / 'best.pt'}")
                else:
                    evals_since_improvement += 1

                if args.early_stop_patience is not None and evals_since_improvement >= args.early_stop_patience:
                    print(f"early stopping: {evals_since_improvement} evals without improvement")
                    stop_training = True
                if args.max_seconds is not None and elapsed >= args.max_seconds:
                    print(f"stopping: max_seconds={args.max_seconds} reached ({elapsed:.0f}s elapsed)")
                    stop_training = True
                if stop_training:
                    break

    torch.save(
        {"model_cfg": model_cfg, "state_dict": eval_model().state_dict(), "valid_wp": best_wp},
        out_dir / "last.pt",
    )
    print(f"done. best valid wp_masked (on {args.valid_sequences} sequences): {best_wp:.4f}")

    # Calibrate the output scale: Global WP is only scale-invariant before the metric's
    # [-2,2] clip (notes/LOG.md 2026-09-17). Applies to both checkpoints; multiplier is
    # stored in the .pt file so export_onnx.py can bake it into the exported graph.
    for ckpt_name in ("best.pt", "last.pt"):
        ckpt_path = out_dir / ckpt_name
        if not ckpt_path.exists():
            continue
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        calib_model = SequenceModel(ckpt["model_cfg"]).to(device)
        calib_model.load_state_dict(ckpt["state_dict"])
        best_c, table = sweep_output_multiplier(
            calib_model, args.valid_path, device,
            max_sequences=args.calibrate_sequences, batch_size=args.valid_batch_size,
        )
        uncalibrated = next(r["wp_masked"] for r in table if r["c"] == 1.0)
        calibrated = next(r["wp_masked"] for r in table if r["c"] == best_c)
        ckpt["output_multiplier"] = best_c
        torch.save(ckpt, ckpt_path)
        print(
            f"calibrated {ckpt_name}: multiplier={best_c} "
            f"(wp_masked {uncalibrated:.4f} -> {calibrated:.4f}, "
            f"on {args.calibrate_sequences} sequences)"
        )
        if ckpt_name == "best.pt":
            wandb.log({
                "calib/best_multiplier": best_c,
                "calib/wp_masked_uncalibrated": uncalibrated,
                "calib/wp_masked_calibrated": calibrated,
            })

    wandb.finish()


if __name__ == "__main__":
    main()
