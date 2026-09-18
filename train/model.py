"""Pointwise MLP trunk -> GRU -> heads, with optional t0/t1 reparameterization and an
optional linear skip connection from the raw features straight to the two outputs."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn


@dataclass
class ModelConfig:
    n_features: int = 112
    trunk_hidden: tuple[int, ...] = ()      # e.g. (64,) ; () = no trunk, GRU sees raw features
    trunk_activation: str = "relu"
    gru_hidden: int = 128
    gru_layers: int = 2
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


class SequenceModel(nn.Module):
    """trunk(pointwise) -> GRU(stateful) -> linear head(2). Reparam is a pure output transform."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.trunk = Trunk(cfg)
        self.gru = nn.GRU(
            input_size=self.trunk.out_dim,
            hidden_size=cfg.gru_hidden,
            num_layers=cfg.gru_layers,
            batch_first=True,
        )
        self.head = nn.Linear(cfg.gru_hidden, 2)
        # Skip connection targets (t0, t1) directly (ridge_v1 was fit on raw t0/t1), so it's
        # added AFTER to_targets, not mixed into the pre-reparam `raw` head output.
        self.skip = nn.Linear(cfg.n_features, 2) if cfg.use_skip else None

    def init_skip_from_ridge(self, ridge_npz_path: str | Path):
        """Loads models/ridge_v1.npz's closed-form weighted-ridge coefficients into the skip
        connection, so the recurrent part only has to learn the residual beyond a model that
        already beats every trained checkpoint we have (see notes/LOG.md 2026-09-17)."""
        assert self.skip is not None, "use_skip=False; no skip connection to initialize"
        data = np.load(ridge_npz_path)
        beta = data["beta"]  # (n_features + 1, 2); last row is the bias term
        bias_idx = int(data["bias_column_index"])
        assert beta.shape == (self.cfg.n_features + 1, 2) and bias_idx == self.cfg.n_features
        with torch.no_grad():
            self.skip.weight.copy_(torch.from_numpy(beta[:bias_idx].T).float())
            self.skip.bias.copy_(torch.from_numpy(beta[bias_idx]).float())

    def init_hidden(self, batch_size: int, device=None, dtype=None) -> torch.Tensor:
        return torch.zeros(
            self.cfg.gru_layers, batch_size, self.cfg.gru_hidden, device=device, dtype=dtype
        )

    def raw_forward(self, features: torch.Tensor, hidden: torch.Tensor):
        """features: (B, L, n_features). Returns (raw_head_out (B, L, 2), next_hidden)."""
        x = self.trunk(features)
        out, next_hidden = self.gru(x, hidden)
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
