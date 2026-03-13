"""
Hybrid Quantum-Classical Neural Network — Qiskit backend.

Uses ``qiskit-machine-learning``'s **EstimatorQNN** wrapped in a
``TorchConnector`` so the entire pipeline is end-to-end differentiable
within PyTorch.

Architecture::

    Classical Pre-Layers → Qiskit VQC (EstimatorQNN) → Classical Post-Layers → Logits
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

try:
    from qiskit.circuit import QuantumCircuit, ParameterVector
    from qiskit.circuit.library import RealAmplitudes, EfficientSU2, TwoLocal, ZZFeatureMap
    from qiskit_machine_learning.neural_networks import EstimatorQNN
    from qiskit_machine_learning.connectors import TorchConnector

    _QISKIT_AVAILABLE = True
except ImportError:
    _QISKIT_AVAILABLE = False


# ═══════════════════════════════════════════════════════════
#  Qiskit circuit builder
# ═══════════════════════════════════════════════════════════

def _build_qiskit_circuit(
    n_qubits: int,
    n_layers: int,
    encoding_type: str = "angle",
    ansatz_type: str = "real_amplitudes",
    entanglement: str = "full",
    reps: int = 3,
) -> tuple["QuantumCircuit", list, list]:
    """Return ``(circuit, input_params, weight_params)``.

    The circuit is composed of:
        1. Feature map (data encoding)
        2. Variational ansatz (trainable)
    """
    # ── Feature map ──────────────────────────────────────
    input_params = ParameterVector("x", n_qubits)
    feature_map = QuantumCircuit(n_qubits)

    if encoding_type == "angle":
        for i in range(n_qubits):
            feature_map.ry(input_params[i], i)
    elif encoding_type == "iqp":
        fm = ZZFeatureMap(feature_dimension=n_qubits, reps=2)
        feature_map = fm
        input_params = fm.parameters
    else:
        for i in range(n_qubits):
            feature_map.ry(input_params[i], i)

    # ── Ansatz ───────────────────────────────────────────
    ansatz_map = {
        "real_amplitudes": RealAmplitudes,
        "efficient_su2": EfficientSU2,
        "two_local": TwoLocal,
    }
    ansatz_cls = ansatz_map.get(ansatz_type, RealAmplitudes)
    ansatz = ansatz_cls(
        num_qubits=n_qubits,
        reps=reps,
        entanglement=entanglement,
    )
    weight_params = list(ansatz.parameters)

    # ── Compose ──────────────────────────────────────────
    qc = QuantumCircuit(n_qubits)
    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz, inplace=True)

    return qc, list(input_params), weight_params


# ═══════════════════════════════════════════════════════════
#  Qiskit Quantum Layer (nn.Module)
# ═══════════════════════════════════════════════════════════

class QiskitQuantumLayer(nn.Module):
    """Qiskit EstimatorQNN wrapped as a PyTorch layer.

    Requires ``qiskit>=1.0`` and ``qiskit-machine-learning>=0.7``.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 3,
        encoding_type: str = "angle",
        ansatz_type: str = "real_amplitudes",
        entanglement: str = "full",
        n_outputs: int = 2,
        reps: int = 3,
    ) -> None:
        super().__init__()
        if not _QISKIT_AVAILABLE:
            raise ImportError(
                "Qiskit ML not installed. Run: pip install qiskit qiskit-aer qiskit-machine-learning"
            )

        self.n_qubits = n_qubits
        self.n_outputs = n_outputs

        qc, input_params, weight_params = _build_qiskit_circuit(
            n_qubits, n_layers, encoding_type, ansatz_type, entanglement, reps
        )

        # Build observables for EstimatorQNN
        from qiskit.quantum_info import SparsePauliOp

        observables = []
        for i in range(min(n_outputs, n_qubits)):
            label = ["I"] * n_qubits
            label[i] = "Z"
            observables.append(SparsePauliOp("".join(label)))

        self._estimator_qnn = EstimatorQNN(
            circuit=qc,
            input_params=input_params,
            weight_params=weight_params,
            observables=observables,
        )
        self._torch_layer = TorchConnector(self._estimator_qnn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` shape ``(batch, n_qubits)`` → ``(batch, n_outputs)``."""
        return self._torch_layer(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def from_config(cls, cfg: dict) -> "QiskitQuantumLayer":
        return cls(
            n_qubits=cfg["n_qubits"],
            n_layers=cfg["n_layers"],
            encoding_type=cfg["encoding"]["type"],
            ansatz_type=cfg["ansatz"]["type"],
            entanglement=cfg["ansatz"].get("entanglement", "full"),
            n_outputs=cfg["measurement"].get("n_outputs", 2),
            reps=cfg["ansatz"].get("reps", 3),
        )


# ═══════════════════════════════════════════════════════════
#  Hybrid Qiskit Net
# ═══════════════════════════════════════════════════════════

class HybridQiskitNet(nn.Module):
    """Hybrid quantum-classical network using Qiskit backend.

    Same high-level architecture as ``HybridPennyLaneNet`` but the
    quantum layer is backed by Qiskit's EstimatorQNN.
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
        n_qubits = qcfg.get("n_qubits", 4)

        # ── Pre ──────────────────────────────────────────
        pre: list[nn.Module] = []
        prev = input_dim
        for h in classical_pre_layers:
            pre.extend([nn.Linear(prev, h), act_cls(), nn.Dropout(dropout)])
            prev = h
        pre.append(nn.Linear(prev, n_qubits))
        pre.append(nn.Tanh())
        self.pre_net = nn.Sequential(*pre)

        # ── Quantum ──────────────────────────────────────
        self.quantum_layer = QiskitQuantumLayer.from_config(qcfg) if qcfg else QiskitQuantumLayer(n_qubits=n_qubits)
        q_out = qcfg.get("measurement", {}).get("n_outputs", n_qubits)

        # ── Post ─────────────────────────────────────────
        post: list[nn.Module] = []
        prev = q_out
        for h in classical_post_layers:
            post.extend([nn.Linear(prev, h), act_cls(), nn.Dropout(dropout)])
            prev = h
        post.append(nn.Linear(prev, output_dim))
        self.post_net = nn.Sequential(*post)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_net(x)
        x = self.quantum_layer(x)
        x = self.post_net(x)
        return x

    def count_parameters(self) -> dict[str, int]:
        pre = sum(p.numel() for p in self.pre_net.parameters() if p.requires_grad)
        q = sum(p.numel() for p in self.quantum_layer.parameters() if p.requires_grad)
        post = sum(p.numel() for p in self.post_net.parameters() if p.requires_grad)
        return {"classical_pre": pre, "quantum": q, "classical_post": post, "total": pre + q + post}

    @classmethod
    def from_config(cls, cfg: dict) -> "HybridQiskitNet":
        classical = cfg.get("classical", {})
        hybrid = cfg.get("hybrid", {})
        quantum = cfg.get("quantum", {})
        return cls(
            input_dim=classical.get("input_dim", 4),
            output_dim=classical.get("output_dim", 2),
            classical_pre_layers=hybrid.get("classical_pre_layers", [32]),
            classical_post_layers=hybrid.get("classical_post_layers", [16]),
            quantum_config=quantum,
            activation=classical.get("activation", "relu"),
            dropout=classical.get("dropout", 0.1),
        )
