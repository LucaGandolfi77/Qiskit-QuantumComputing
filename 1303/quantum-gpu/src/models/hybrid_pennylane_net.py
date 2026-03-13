"""
Hybrid Quantum-Classical Neural Network — PennyLane backend.

Architecture::

    Classical Pre-Layers → PennyLane Quantum Layer → Classical Post-Layers → Logits

The quantum layer replaces one hidden layer in a standard MLP, forming
an end-to-end differentiable pipeline where gradients flow through the
quantum circuit via PennyLane's PyTorch interface.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from .quantum_layers import PennyLaneQuantumLayer


class HybridPennyLaneNet(nn.Module):
    """Hybrid quantum-classical network using PennyLane.

    Parameters
    ----------
    input_dim : int
        Number of input features.
    output_dim : int
        Number of output logits.
    classical_pre_layers : list[int]
        Hidden dims *before* the quantum layer.
    classical_post_layers : list[int]
        Hidden dims *after* the quantum layer.
    quantum_config : dict
        Configuration for ``PennyLaneQuantumLayer``.
    activation : str
        Classical activation function.
    dropout : float
        Dropout rate for classical layers.
    """

    _ACT_MAP = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU, "silu": nn.SiLU}

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 2,
        classical_pre_layers: Sequence[int] = (32,),
        classical_post_layers: Sequence[int] = (16,),
        quantum_config: dict | None = None,
        activation: str = "relu",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        act_cls = self._ACT_MAP.get(activation, nn.ReLU)
        qcfg = quantum_config or {}

        # ── Classical pre-processing ─────────────────────
        pre_layers: list[nn.Module] = []
        prev = input_dim
        for h in classical_pre_layers:
            pre_layers.extend([nn.Linear(prev, h), act_cls(), nn.Dropout(dropout)])
            prev = h
        # Map to n_qubits
        n_qubits = qcfg.get("n_qubits", 4)
        pre_layers.append(nn.Linear(prev, n_qubits))
        pre_layers.append(nn.Tanh())  # bound inputs to [-1, 1] for encoding
        self.pre_net = nn.Sequential(*pre_layers)

        # ── Quantum layer ────────────────────────────────
        self.quantum_layer = PennyLaneQuantumLayer.from_config(qcfg) if qcfg else PennyLaneQuantumLayer(n_qubits=n_qubits)
        q_out = qcfg.get("measurement", {}).get("n_outputs", n_qubits)

        # ── Classical post-processing ────────────────────
        post_layers: list[nn.Module] = []
        prev = q_out
        for h in classical_post_layers:
            post_layers.extend([nn.Linear(prev, h), act_cls(), nn.Dropout(dropout)])
            prev = h
        post_layers.append(nn.Linear(prev, output_dim))
        self.post_net = nn.Sequential(*post_layers)

    # ── Forward ──────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` shape ``(batch, input_dim)`` → logits ``(batch, output_dim)``."""
        x = self.pre_net(x)           # → (batch, n_qubits)
        x = self.quantum_layer(x)     # → (batch, n_outputs)  — may be float64
        x = x.float()                 # ensure float32 for post-net
        x = self.post_net(x)          # → (batch, output_dim)
        return x

    # ── Helpers ──────────────────────────────────────────
    def count_parameters(self) -> dict[str, int]:
        """Parameter count broken down by component."""
        pre = sum(p.numel() for p in self.pre_net.parameters() if p.requires_grad)
        q = sum(p.numel() for p in self.quantum_layer.parameters() if p.requires_grad)
        post = sum(p.numel() for p in self.post_net.parameters() if p.requires_grad)
        return {"classical_pre": pre, "quantum": q, "classical_post": post, "total": pre + q + post}

    def draw_circuit(self, sample_input: torch.Tensor | None = None) -> str:
        """Return a string diagram of the quantum sub-circuit."""
        return self.quantum_layer.draw(sample_input)

    @classmethod
    def from_config(cls, cfg: dict) -> "HybridPennyLaneNet":
        """Build from the full YAML config."""
        classical = cfg.get("classical", {})
        hybrid = cfg.get("hybrid", {})
        quantum = cfg.get("quantum", {})
        # Inject gradient method from training config
        training = cfg.get("training", {})
        quantum["gradient_method"] = training.get("gradient_method", "backprop")
        return cls(
            input_dim=classical.get("input_dim", 4),
            output_dim=classical.get("output_dim", 2),
            classical_pre_layers=hybrid.get("classical_pre_layers", [32]),
            classical_post_layers=hybrid.get("classical_post_layers", [16]),
            quantum_config=quantum,
            activation=classical.get("activation", "relu"),
            dropout=classical.get("dropout", 0.1),
        )
