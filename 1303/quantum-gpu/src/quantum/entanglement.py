"""
Entanglement patterns for variational quantum circuits.

Provides composable entangling layers that can be inserted between
parameterised rotation blocks in any ansatz.

Patterns:
    • **full** — all-to-all CNOT connectivity
    • **linear** — nearest-neighbour chain
    • **circular** — nearest-neighbour ring
"""

from __future__ import annotations

import pennylane as qml


def apply_entanglement(n_qubits: int, pattern: str = "full") -> None:
    """Apply one layer of CNOT gates according to *pattern*.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    pattern : str
        ``"full"`` | ``"linear"`` | ``"circular"``.
    """
    if n_qubits < 2:
        return  # nothing to entangle

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
        qml.CNOT(wires=[n_qubits - 1, 0])

    else:
        raise ValueError(
            f"Unknown entanglement pattern '{pattern}'. "
            "Choose from: full, linear, circular."
        )


def get_entanglement_pairs(n_qubits: int, pattern: str) -> list[tuple[int, int]]:
    """Return the list of CNOT control-target pairs for analysis."""
    if pattern == "full":
        return [(i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)]
    elif pattern == "linear":
        return [(i, i + 1) for i in range(n_qubits - 1)]
    elif pattern == "circular":
        pairs = [(i, i + 1) for i in range(n_qubits - 1)]
        if n_qubits > 1:
            pairs.append((n_qubits - 1, 0))
        return pairs
    else:
        raise ValueError(f"Unknown entanglement pattern '{pattern}'.")


def count_cnot_gates(n_qubits: int, pattern: str) -> int:
    """Return the CNOT count for one entangling layer."""
    return len(get_entanglement_pairs(n_qubits, pattern))
