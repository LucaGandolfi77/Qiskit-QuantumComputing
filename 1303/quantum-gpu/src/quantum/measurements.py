"""
Observable definitions and measurement strategies.

Handles the final measurement stage of a VQC, translating quantum
expectation values into classical outputs for the hybrid network.
"""

from __future__ import annotations

from typing import Sequence

import pennylane as qml


# ──────────────────────────────────────────────────────────
#  Measurement functions (called inside QNodes)
# ──────────────────────────────────────────────────────────

def measure_expectations(
    n_qubits: int,
    observables: Sequence[str] = ("Z",),
    n_outputs: int | None = None,
) -> list:
    """Return PennyLane expectation values for the given observables.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    observables : sequence of str
        Pauli labels per qubit: ``"X"``, ``"Y"``, ``"Z"``, ``"I"``.
    n_outputs : int | None
        How many measurement outputs to return.  If *None*, returns
        one per qubit.

    Returns
    -------
    list of qml.measurements
        Expectation value objects for each measured qubit.
    """
    obs_map = {
        "X": qml.PauliX,
        "Y": qml.PauliY,
        "Z": qml.PauliZ,
        "I": qml.Identity,
    }

    n_out = n_outputs or n_qubits
    results = []
    for i in range(min(n_out, n_qubits)):
        obs_name = observables[i % len(observables)]
        obs_cls = obs_map.get(obs_name)
        if obs_cls is None:
            raise ValueError(
                f"Unknown observable '{obs_name}'. Choose from {list(obs_map)}."
            )
        results.append(qml.expval(obs_cls(wires=i)))

    return results


def measure_probabilities(wires: Sequence[int] | None = None) -> list:
    """Return computational basis probabilities."""
    return [qml.probs(wires=wires)]


def measure_samples(
    n_qubits: int,
    n_shots: int = 1024,
) -> list:
    """Return measurement samples (requires shot-based execution)."""
    return [qml.sample(qml.PauliZ(wires=i)) for i in range(n_qubits)]


# ──────────────────────────────────────────────────────────
#  Observable builder helper (for Qiskit compat)
# ──────────────────────────────────────────────────────────

def build_observable_list(
    n_qubits: int,
    observables: Sequence[str] = ("Z",),
    n_outputs: int | None = None,
) -> list[str]:
    """Return observable label strings for each measured qubit.

    Useful for constructing Qiskit ``SparsePauliOp`` instances.
    """
    n_out = n_outputs or n_qubits
    labels = []
    for i in range(min(n_out, n_qubits)):
        obs = observables[i % len(observables)]
        # Build n-qubit Pauli string: I ⊗ … ⊗ Obs ⊗ … ⊗ I
        pauli_str = ["I"] * n_qubits
        pauli_str[i] = obs
        labels.append("".join(pauli_str))
    return labels
