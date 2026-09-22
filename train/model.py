"""Pointwise MLP trunk -> temporal backbone (GRU or TCN) -> heads, with optional t0/t1
reparameterization and an optional linear skip connection from the raw features straight
to the two outputs.

`--arch gru` is the original, untouched path (SequenceModel's GRU branch is exactly the
code that shipped before the TCN was added — see notes/LOG.md 2026-09-21).
`--arch tcn` is a dilated causal convolution stack, added as a structurally different,
CPU-fast alternative. Both backbones share the same init_state/forward(x, state) shape
contract so train.py's loop, losses, evaluator and checkpointing are unchanged either way.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ModelConfig:
    n_features: int = 112
    trunk_hidden: tuple[int, ...] = ()      # e.g. (64,) ; () = no trunk, backbone sees raw features
    trunk_activation: str = "relu"
    arch: str = "gru"                        # "gru" or "tcn"
    gru_hidden: int = 128
    gru_layers: int = 2
    tcn_channels: int = 128
    tcn_kernel_size: int = 3
    tcn_layers: int = 5                      # dilations 1,2,4,...,2^(tcn_layers-1)
    reparam: bool = False                    # predict (t0+t1)/2, (t0-t1)/2 instead of t0, t1 directly
    dropout: float = 0.0
    use_skip: bool = False                   # linear skip: raw features -> (t0, t1), added post-reparam


def _activation(name: str) -> nn.Module:
    return {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}[name]()


class Trunk(nn.Module):
    """Pointwise (shared-weight, per-timestep) MLP. Identity if trunk_hidden is empty."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        dims = [cfg.n_features, *cfg.trunk_hidden]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(_activation(cfg.trunk_activation))
            if cfg.dropout > 0:
                layers.append(nn.Dropout(cfg.dropout))
        self.net = nn.Sequential(*layers) if layers else nn.Identity()
        self.out_dim = dims[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, n_features) -> (B, L, out_dim); Linear broadcasts over leading dims already.
        return self.net(x)


class GRUBackbone(nn.Module):
    """Unchanged GRU path, wrapped behind the shared init_state/forward(x, state) contract."""

    def __init__(self, in_dim: int, hidden: int, layers: int):
        super().__init__()
        self.gru = nn.GRU(input_size=in_dim, hidden_size=hidden, num_layers=layers, batch_first=True)
        self.layers = layers
        self.hidden = hidden
        self.out_channels = hidden
        self.receptive_field = None  # unbounded (recurrent state carries the whole sequence)

    def init_state(self, batch_size: int, device=None, dtype=None) -> torch.Tensor:
        return torch.zeros(self.layers, batch_size, self.hidden, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, state: torch.Tensor):
        return self.gru(x, state)


class CausalConvBlock(nn.Module):
    """One dilated causal conv layer: left-pad by (kernel-1)*dilation only, so output at
    position t depends solely on inputs at positions <= t (never t+1 or later).

    Normalization MUST be per-timestep (channels only) — GroupNorm(1, C) on a (B,C,T)
    tensor pools its mean/var over channels AND all T positions jointly, which means an
    output at an early position would depend on later (future, within the same forward
    call) raw values through the shared statistics: a real causality leak, caught by a
    quick manual check before the causality self-test even ran (perturbing the last
    timestep changed the first timestep's normalized output by >1). nn.LayerNorm(channels)
    applied per-(batch,time) position has no such cross-time pooling, so it's used instead
    — this needs a (B,T,C) layout, hence the transposes around the conv (which wants
    (B,C,T)). Residual connection added in (B,T,C).
    """

    def __init__(self, channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.norm = nn.LayerNorm(channels)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        out = x.transpose(1, 2)                 # (B, C, T)
        out = F.pad(out, (self.left_pad, 0))
        out = self.conv(out)
        out = out.transpose(1, 2)                # (B, T, C)
        out = self.norm(out)
        out = self.act(out)
        return out + x


class TCNBackbone(nn.Module):
    """Dilated causal convolution stack. State is a ring buffer of the last
    (receptive_field - 1) TRUNK-OUTPUT feature vectors (trunk is pointwise/stateless, so
    storing its output instead of raw features is exactly equivalent and avoids
    recomputing the trunk on old buffer rows every step).

    Per notes/TASK.md's explicit instruction, a single forward step reprocesses the whole
    buffer window (O(receptive_field) per row, not an incrementally-cached O(1)) — this is
    the literal "ring buffer + one forward pass per row" design, not an optimization.
    """

    def __init__(self, in_dim: int, channels: int, kernel_size: int, n_layers: int):
        super().__init__()
        self.input_proj = nn.Conv1d(in_dim, channels, kernel_size=1) if in_dim != channels else nn.Identity()
        self.blocks = nn.ModuleList(
            [CausalConvBlock(channels, kernel_size, dilation=2**i) for i in range(n_layers)]
        )
        self.receptive_field = 1 + sum((kernel_size - 1) * (2**i) for i in range(n_layers))
        self.buffer_len = self.receptive_field - 1
        self.in_dim = in_dim
        self.out_channels = channels

    def init_state(self, batch_size: int, device=None, dtype=None) -> torch.Tensor:
        return torch.zeros(batch_size, self.buffer_len, self.in_dim, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, state: torch.Tensor):
        # x: (B, L, in_dim), state: (B, buffer_len, in_dim) — the trunk-output history from
        # immediately before this chunk (zeros at the start of a sequence: zero-padded
        # left-context, since a receptive field beyond 99 rows cannot be fully populated at
        # the first required prediction — see notes/LOG.md 2026-09-21).
        L = x.shape[1]
        x_ctx = torch.cat([state, x], dim=1)               # (B, buffer_len + L, in_dim)
        h = x_ctx.transpose(1, 2)                            # (B, in_dim, buffer_len+L)
        h = self.input_proj(h)
        h = h.transpose(1, 2)                                 # (B, buffer_len+L, channels) — blocks are (B,T,C)
        for block in self.blocks:
            h = block(h)
        out = h[:, -L:, :]
        new_state = x_ctx[:, x_ctx.shape[1] - self.buffer_len :, :] if self.buffer_len > 0 else x_ctx[:, :0, :]
        return out, new_state


class SequenceModel(nn.Module):
    """trunk(pointwise) -> temporal backbone(stateful) -> linear head(2). Reparam is a pure
    output transform. `cfg.arch` selects GRUBackbone (unchanged) or TCNBackbone."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.trunk = Trunk(cfg)
        if cfg.arch == "gru":
            self.temporal = GRUBackbone(self.trunk.out_dim, cfg.gru_hidden, cfg.gru_layers)
        elif cfg.arch == "tcn":
            self.temporal = TCNBackbone(self.trunk.out_dim, cfg.tcn_channels, cfg.tcn_kernel_size, cfg.tcn_layers)
        else:
            raise ValueError(f"unknown arch: {cfg.arch!r}")
        self.head = nn.Linear(self.temporal.out_channels, 2)
        # Skip connection targets (t0, t1) directly (ridge_v1 was fit on raw t0/t1), so it's
        # added AFTER to_targets, not mixed into the pre-reparam `raw` head output.
        self.skip = nn.Linear(cfg.n_features, 2) if cfg.use_skip else None

    def init_skip_from_ridge(self, ridge_npz_path: str | Path):
        """Loads models/ridge_v1.npz's closed-form weighted-ridge coefficients into the skip
        connection, so the temporal backbone only has to learn the residual beyond a model
        that already beats every trained checkpoint we have (see notes/LOG.md 2026-09-17)."""
        assert self.skip is not None, "use_skip=False; no skip connection to initialize"
        data = np.load(ridge_npz_path)
        beta = data["beta"]  # (n_features + 1, 2); last row is the bias term
        bias_idx = int(data["bias_column_index"])
        assert beta.shape == (self.cfg.n_features + 1, 2) and bias_idx == self.cfg.n_features
        with torch.no_grad():
            self.skip.weight.copy_(torch.from_numpy(beta[:bias_idx].T).float())
            self.skip.bias.copy_(torch.from_numpy(beta[bias_idx]).float())

    def init_hidden(self, batch_size: int, device=None, dtype=None) -> torch.Tensor:
        return self.temporal.init_state(batch_size, device=device, dtype=dtype)

    def load_state_dict(self, state_dict, strict=True):
        # Backward compat: every checkpoint saved before the TCN refactor (2026-09-21,
        # notes/LOG.md) stored the GRU directly as `gru.*` — it has since moved to
        # `temporal.gru.*` inside GRUBackbone. Remap transparently so old checkpoints
        # (including our best model, runs/02_trunk_global_full) still load.
        if any(k.startswith("gru.") for k in state_dict) and not any(k.startswith("temporal.") for k in state_dict):
            state_dict = {("temporal." + k if k.startswith("gru.") else k): v for k, v in state_dict.items()}
        return super().load_state_dict(state_dict, strict=strict)

    def raw_forward(self, features: torch.Tensor, hidden: torch.Tensor):
        """features: (B, L, n_features). Returns (raw_head_out (B, L, 2), next_hidden)."""
        x = self.trunk(features)
        out, next_hidden = self.temporal(x, hidden)
        return self.head(out), next_hidden

    def to_targets(self, raw: torch.Tensor) -> torch.Tensor:
        """Map raw head output to (t0_hat, t1_hat), undoing the reparam if configured."""
        if not self.cfg.reparam:
            return raw
        m, d = raw[..., 0], raw[..., 1]
        return torch.stack([m + d, m - d], dim=-1)

    def forward(self, features: torch.Tensor, hidden: torch.Tensor):
        """Returns (t0_hat, t1_hat) stacked as (B, L, 2), and next_hidden."""
        raw, next_hidden = self.raw_forward(features, hidden)
        pred = self.to_targets(raw)
        if self.skip is not None:
            pred = pred + self.skip(features)
        return pred, next_hidden


class StatefulStep(nn.Module):
    """Single-timestep wrapper for ONNX export: features[1,1,F] + hidden[layers,1,H] -> prediction[1,1,2] + next_hidden."""

    def __init__(self, model: SequenceModel):
        super().__init__()
        self.model = model

    def forward(self, features: torch.Tensor, hidden: torch.Tensor):
        pred, next_hidden = self.model(features, hidden)
        return pred, next_hidden
