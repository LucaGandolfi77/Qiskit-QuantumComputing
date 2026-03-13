"""
Data encoding strategies for quantum circuits.

Maps classical feature vectors **x** ∈ ℝⁿ onto an n-qubit quantum state
|ψ(x)⟩ using different embedding strategies:

    • **Angle encoding** — rotations Rₐ(xᵢ) on each qubit
    • **Amplitude encoding** — encodes 2ⁿ amplitudes directly
    • **IQP encoding** — data re-uploading with ZZ interactions
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pennylane as qml


# ──────────────────────────────────────────────────────────
#  Angle (rotation) encoding
# ──────────────────────────────────────────────────────────

def angle_encoding(
    inputs: np.ndarray,
    n_qubits: int,
    rotation_axes: Sequence[str] | None = None,
) -> None:
    """Embed one scalar per qubit via a Pauli rotation.

    Parameters
    ----------
    inputs : array-like, shape ``(n_qubits,)``
        Classical features (≤ *n_qubits* values).
    n_qubits : int
        Register size.
    rotation_axes : list[str]
        ``"X"``, ``"Y"``, or ``"Z"`` per qubit (cycled if shorter).
    """
    axes = rotation_axes or ["Y"]
    gate_map = {"X": qml.RX, "Y": qml.RY, "Z": qml.RZ}

    for i in range(n_qubits):
        axis = axes[i % len(axes)]
        gate = gate_map[axis]
        # Use input value if available, else 0
        val = inputs[i] if i < len(inputs) else 0.0
        gate(val, wires=i)


# ──────────────────────────────────────────────────────────
#  Amplitude encoding
# ──────────────────────────────────────────────────────────

def amplitude_encoding(
    inputs: np.ndarray,
    n_qubits: int,
    **_kwargs,
) -> None:
    """Encode the feature vector as quantum amplitude vector.

    The input is padded / truncated to length ``2**n_qubits`` and
    L2-normalised to form a valid quantum state.

    Parameters
    ----------
    inputs : array-like
        Classical features.
    n_qubits : int
        Register size.
    """
    target_len = 2 ** n_qubits
    padded = np.zeros(target_len, dtype=float)
    n = min(len(inputs), target_len)
    padded[:n] = inputs[:n]

    # L2 normalise (must be a valid state vector)
    norm = np.linalg.norm(padded)
    if norm < 1e-12:
        padded[0] = 1.0
    else:
        padded = padded / norm

    qml.AmplitudeEmbedding(padded, wires=range(n_qubits), normalize=True)


# ──────────────────────────────────────────────────────────
#  IQP (Instantaneous Quantum Polynomial) encoding
# ──────────────────────────────────────────────────────────

def iqp_encoding(
    inputs: np.ndarray,
    n_qubits: int,
    n_repeats: int = 2,
    **_kwargs,
) -> None:
    """IQP-style data re-uploading encoding.

    For each repetition:
        1. Hadamard on all qubits
        2. RZ(xᵢ)  on qubit *i*
        3. ZZ(xᵢ · xⱼ) entangling on pairs

    Parameters
    ----------
    inputs : array-like, shape ``(n_qubits,)``
    n_qubits : int
    n_repeats : int
        Number of encoding repetitions (increases expressivity).
    """
    qml.IQPEmbedding(inputs[:n_qubits], wires=range(n_qubits), n_repeats=n_repeats)


# ──────────────────────────────────────────────────────────
#  Dispatcher
# ──────────────────────────────────────────────────────────

_ENCODING_MAP = {
    "angle": angle_encoding,
    "amplitude": amplitude_encoding,
    "iqp": iqp_encoding,
}


def get_encoding_fn(encoding_type: str, **kwargs):
    """Return a callable ``fn(inputs, n_qubits)`` for the requested encoding."""
    base_fn = _ENCODING_MAP.get(encoding_type)
    if base_fn is None:
        raise ValueError(
            f"Unknown encoding '{encoding_type}'. Choose from {list(_ENCODING_MAP)}."
        )

    def _wrapper(inputs, n_qubits):
        return base_fn(inputs, n_qubits, **kwargs)

    return _wrapper
