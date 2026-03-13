"""
Reusable quantum layers as PyTorch nn.Module.

``PennyLaneQuantumLayer``
    Wraps a PennyLane QNode into a torch-compatible layer with
    trainable quantum parameters and support for backpropagation
    through the quantum circuit via parameter-shift or adjoint
    differentiation.

``QiskitQuantumLayer``
    Wraps a Qiskit EstimatorQNN, forwarding gradients through
    Qiskit's built-in parameter-shift implementation.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn as nn

import pennylane as qml

from ..quantum.circuits import build_ansatz, get_weight_shape
from ..quantum.encodings import get_encoding_fn
from ..quantum.measurements import measure_expectations


# ═══════════════════════════════════════════════════════════
#  PennyLane Quantum Layer
# ═══════════════════════════════════════════════════════════

class PennyLaneQuantumLayer(nn.Module):
    """VQC layer backed by PennyLane, fully differentiable in PyTorch.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    n_layers : int
        Depth of the variational ansatz.
    encoding_type : str
        ``"angle"`` | ``"amplitude"`` | ``"iqp"``.
    ansatz_type : str
        ``"strongly_entangling"`` | ``"hardware_efficient"`` | ``"basic_entangler"``.
    entanglement : str
        ``"full"`` | ``"linear"`` | ``"circular"``.
    rotation_gates : list[str]
        Gate names for each qubit rotation.
    observables : list[str]
        Pauli observables (``"X"``, ``"Y"``, ``"Z"``).
    n_outputs : int
        Number of expectation values returned.
    backend : str
        PennyLane device name (default ``"default.qubit"``).
    diff_method : str
        ``"backprop"`` | ``"parameter_shift"`` | ``"adjoint"``.
    shots : int | None
        ``None`` for exact simulation.
    encoding_kwargs : dict | None
        Extra keyword arguments forwarded to the encoding function.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 3,
        encoding_type: str = "angle",
        ansatz_type: str = "strongly_entangling",
        entanglement: str = "full",
        rotation_gates: Sequence[str] = ("RX", "RY", "RZ"),
        observables: Sequence[str] = ("Z",),
        n_outputs: int = 2,
        backend: str = "default.qubit",
        diff_method: str = "backprop",
        shots: int | None = None,
        encoding_kwargs: dict | None = None,
    ) -> None:
        super().__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_outputs = n_outputs
        self.ansatz_type = ansatz_type
        self.entanglement = entanglement
        self.rotation_gates = list(rotation_gates)
        self.observables = list(observables)

        # Encoding function
        self._encoding_fn = get_encoding_fn(encoding_type, **(encoding_kwargs or {}))

        # Weight shape
        w_shape = get_weight_shape(ansatz_type, n_qubits, n_layers, rotation_gates)
        self.weights = nn.Parameter(
            torch.randn(w_shape, dtype=torch.float32) * 0.1
        )

        # PennyLane device
        dev_kwargs = {}
        if shots is not None:
            dev_kwargs["shots"] = shots
        self._dev = qml.device(backend, wires=n_qubits, **dev_kwargs)

        # Build QNode
        self._qnode = qml.QNode(
            self._circuit,
            self._dev,
            interface="torch",
            diff_method=diff_method,
        )

    # ── Circuit definition ───────────────────────────────
    def _circuit(self, inputs, weights):
        """Quantum circuit: encoding → ansatz → measurements."""
        self._encoding_fn(inputs, self.n_qubits)
        build_ansatz(
            weights,
            self.n_qubits,
            self.n_layers,
            self.ansatz_type,
            self.entanglement,
            self.rotation_gates,
        )
        return measure_expectations(
            self.n_qubits, self.observables, self.n_outputs
        )

    # ── Forward ──────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process a batch through the quantum circuit.

        Parameters
        ----------
        x : Tensor, shape ``(batch, n_qubits)``

        Returns
        -------
        Tensor, shape ``(batch, n_outputs)``
        """
        batch_size = x.shape[0]
        outputs = []
        for i in range(batch_size):
            result = self._qnode(x[i], self.weights)
            if isinstance(result, (list, tuple)):
                outputs.append(torch.stack(list(result)))
            else:
                outputs.append(result.unsqueeze(0))
        return torch.stack(outputs)

    # ── Helpers ──────────────────────────────────────────
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def draw(self, sample_input: torch.Tensor | None = None) -> str:
        """Return a string diagram of the circuit."""
        if sample_input is None:
            sample_input = torch.zeros(self.n_qubits)
        return qml.draw(self._qnode)(sample_input, self.weights)

    @classmethod
    def from_config(cls, cfg: dict) -> "PennyLaneQuantumLayer":
        """Build from a YAML ``quantum`` config block."""
        return cls(
            n_qubits=cfg["n_qubits"],
            n_layers=cfg["n_layers"],
            encoding_type=cfg["encoding"]["type"],
            ansatz_type=cfg["ansatz"]["type"],
            entanglement=cfg["ansatz"].get("entanglement", "full"),
            rotation_gates=cfg["ansatz"].get("rotation_gates", ["RX", "RY", "RZ"]),
            observables=cfg["measurement"].get("observables", ["Z"]),
            n_outputs=cfg["measurement"].get("n_outputs", 2),
            backend=cfg["backend"].get("name", "default.qubit"),
            diff_method=cfg.get("gradient_method", "backprop"),
            shots=cfg["backend"].get("shots"),
            encoding_kwargs={"rotation_axes": cfg["encoding"].get("rotation_axes")},
        )
