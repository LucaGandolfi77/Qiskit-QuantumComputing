"""
Variational Quantum Circuit (VQC) ansatzes.

Implements parameterized circuit templates used as trainable layers in
hybrid quantum-classical models. All circuits are written in PennyLane's
functional API so they can be composed inside QNode definitions.

Supported ansatzes:
    • StronglyEntangling  — Schuld et al. (2020)
    • HardwareEfficient    — Kandala et al. (2017)
    • BasicEntangler       — simple Ry-CNOT pattern
"""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
import pennylane as qml


# ──────────────────────────────────────────────────────────
#  High-level builder
# ──────────────────────────────────────────────────────────

def build_ansatz(
    weights: np.ndarray,
    n_qubits: int,
    n_layers: int,
    ansatz_type: str = "strongly_entangling",
    entanglement: str = "full",
    rotation_gates: Sequence[str] | None = None,
) -> None:
    """Apply a variational ansatz to the current quantum register.

    Parameters
    ----------
    weights : array-like
        Trainable parameters. Shape depends on *ansatz_type*.
    n_qubits : int
        Number of qubits in the register.
    n_layers : int
        Circuit depth (number of variational repetitions).
    ansatz_type : str
        ``"strongly_entangling"`` | ``"hardware_efficient"`` | ``"basic_entangler"``.
    entanglement : str
        ``"full"`` | ``"linear"`` | ``"circular"``.
    rotation_gates : list[str] | None
        Gate names per qubit per layer (e.g. ``["RX", "RY", "RZ"]``).
    """
    dispatch = {
        "strongly_entangling": _strongly_entangling_ansatz,
        "hardware_efficient": _hardware_efficient_ansatz,
        "basic_entangler": _basic_entangler_ansatz,
    }
    fn = dispatch.get(ansatz_type)
    if fn is None:
        raise ValueError(
            f"Unknown ansatz '{ansatz_type}'. Choose from {list(dispatch)}."
        )
    fn(weights, n_qubits, n_layers, entanglement, rotation_gates or ["RY"])


def build_vqc_circuit(
    inputs: np.ndarray,
    weights: np.ndarray,
    n_qubits: int,
    n_layers: int,
    encoding_fn,
    ansatz_type: str = "strongly_entangling",
    entanglement: str = "full",
    rotation_gates: Sequence[str] | None = None,
) -> None:
    """Convenience helper: encode data then apply ansatz."""
    encoding_fn(inputs, n_qubits)
    build_ansatz(
        weights, n_qubits, n_layers, ansatz_type, entanglement, rotation_gates
    )


# ──────────────────────────────────────────────────────────
#  Ansatz implementations
# ──────────────────────────────────────────────────────────

_GATE_MAP = {
    "RX": qml.RX,
    "RY": qml.RY,
    "RZ": qml.RZ,
}


def _strongly_entangling_ansatz(
    weights: np.ndarray,
    n_qubits: int,
    n_layers: int,
    entanglement: str,
    rotation_gates: Sequence[str],
) -> None:
    """StronglyEntanglingLayers wrapper.

    ``weights`` shape: ``(n_layers, n_qubits, 3)``  — always three
    rotation angles per qubit per layer (Rz-Ry-Rz decomposition).
    """
    # PennyLane's built-in template handles entanglement internally
    ranges = _entanglement_ranges(n_qubits, entanglement, n_layers)
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits), ranges=ranges)


def _hardware_efficient_ansatz(
    weights: np.ndarray,
    n_qubits: int,
    n_layers: int,
    entanglement: str,
    rotation_gates: Sequence[str],
) -> None:
    """Parameterised single-qubit rotations + entangling CNOTs.

    ``weights`` shape: ``(n_layers, n_qubits, len(rotation_gates))``
    """
    n_gates = len(rotation_gates)
    for layer in range(n_layers):
        for q in range(n_qubits):
            for g_idx, gate_name in enumerate(rotation_gates):
                gate_cls = _GATE_MAP[gate_name]
                gate_cls(weights[layer, q, g_idx], wires=q)
        # Entangling block
        _apply_cnot_entanglement(n_qubits, entanglement)


def _basic_entangler_ansatz(
    weights: np.ndarray,
    n_qubits: int,
    n_layers: int,
    entanglement: str,
    rotation_gates: Sequence[str],
) -> None:
    """Simple Ry rotation + CNOT chain.

    ``weights`` shape: ``(n_layers, n_qubits)``
    """
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))


# ──────────────────────────────────────────────────────────
#  Entanglement helpers
# ──────────────────────────────────────────────────────────

def _apply_cnot_entanglement(n_qubits: int, pattern: str) -> None:
    """Insert a layer of CNOTs according to *pattern*."""
    if pattern == "full":
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                qml.CNOT(wires=[i, j])
    elif pattern == "linear":
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
    elif pattern == "circular":
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        if n_qubits > 1:
            qml.CNOT(wires=[n_qubits - 1, 0])
    else:
        raise ValueError(f"Unknown entanglement pattern '{pattern}'.")


def _entanglement_ranges(
    n_qubits: int, pattern: str, n_layers: int
) -> list[int]:
    """Return ``ranges`` parameter for ``StronglyEntanglingLayers``.

    The *ranges* list controls CNOT stride per layer.
    """
    if pattern == "full":
        # Cycle through all possible strides
        return [((l % (n_qubits - 1)) + 1) if n_qubits > 1 else 1 for l in range(n_layers)]
    elif pattern == "linear":
        return [1] * n_layers
    elif pattern == "circular":
        return [1] * n_layers
    else:
        raise ValueError(f"Unknown entanglement pattern '{pattern}'.")


# ──────────────────────────────────────────────────────────
#  Weight shape helpers
# ──────────────────────────────────────────────────────────

def get_weight_shape(
    ansatz_type: str,
    n_qubits: int,
    n_layers: int,
    rotation_gates: Sequence[str] | None = None,
) -> tuple[int, ...]:
    """Return the expected weight tensor shape for a given ansatz."""
    if ansatz_type == "strongly_entangling":
        return (n_layers, n_qubits, 3)
    elif ansatz_type == "hardware_efficient":
        n_gates = len(rotation_gates or ["RY"])
        return (n_layers, n_qubits, n_gates)
    elif ansatz_type == "basic_entangler":
        return (n_layers, n_qubits)
    else:
        raise ValueError(f"Unknown ansatz '{ansatz_type}'.")
