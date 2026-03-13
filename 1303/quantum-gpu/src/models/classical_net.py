"""
Pure PyTorch classical baseline — fully connected network.

Serves as the performance benchmark against which hybrid
quantum-classical architectures are compared.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class ClassicalNet(nn.Module):
    """Configurable MLP for binary / multi-class classification.

    Parameters
    ----------
    input_dim : int
        Number of input features.
    hidden_dims : list[int]
        Width of each hidden layer.
    output_dim : int
        Number of output logits (2 for binary CE, 1 for BCE).
    activation : str
        ``"relu"`` | ``"tanh"`` | ``"gelu"``.
    dropout : float
        Dropout probability applied after each hidden layer.
    batch_norm : bool
        Whether to insert BatchNorm1d before activation.
    """

    _ACT_MAP = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
    }

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (64, 32),
        output_dim: int = 2,
        activation: str = "relu",
        dropout: float = 0.2,
        batch_norm: bool = True,
    ) -> None:
        super().__init__()

        act_cls = self._ACT_MAP.get(activation)
        if act_cls is None:
            raise ValueError(
                f"Unknown activation '{activation}'. "
                f"Choose from {list(self._ACT_MAP)}."
            )

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(act_cls())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

        self._init_weights()

    # ── Weight initialisation ────────────────────────────
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── Forward  ─────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits of shape ``(batch, output_dim)``."""
        return self.net(x)

    # ── Convenience ──────────────────────────────────────
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def from_config(cls, cfg: dict) -> "ClassicalNet":
        """Instantiate from a YAML ``classical`` block."""
        return cls(
            input_dim=cfg["input_dim"],
            hidden_dims=cfg.get("hidden_dims", [64, 32]),
            output_dim=cfg.get("output_dim", 2),
            activation=cfg.get("activation", "relu"),
            dropout=cfg.get("dropout", 0.2),
            batch_norm=cfg.get("batch_norm", True),
        )
