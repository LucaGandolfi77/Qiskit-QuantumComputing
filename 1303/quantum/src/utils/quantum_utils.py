"""
Quantum helper functions.

Miscellaneous utilities used across the quantum modules — reproducibility,
state fidelity, parameter counting, etc.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import pennylane as qml


# ──────────────────────────────────────────────────────────
#  Reproducibility
# ──────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # PennyLane uses numpy internally


# ──────────────────────────────────────────────────────────
#  State fidelity
# ──────────────────────────────────────────────────────────

def state_fidelity(state1: np.ndarray, state2: np.ndarray) -> float:
    """Compute the fidelity |⟨ψ₁|ψ₂⟩|² between two pure states."""
    return float(np.abs(np.dot(np.conj(state1), state2)) ** 2)


# ──────────────────────────────────────────────────────────
#  Parameter analysis
# ──────────────────────────────────────────────────────────

def count_circuit_parameters(
    n_qubits: int,
    n_layers: int,
    ansatz_type: str,
    rotation_gates: Sequence[str] | None = None,
) -> int:
    """Return total trainable parameter count for a VQC."""
    from ..quantum.circuits import get_weight_shape

    shape = get_weight_shape(ansatz_type, n_qubits, n_layers, rotation_gates)
    return int(np.prod(shape))


def count_model_parameters(model: torch.nn.Module) -> dict[str, int]:
    """Count trainable vs total parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


# ──────────────────────────────────────────────────────────
#  Gradient analysis
# ──────────────────────────────────────────────────────────

def compute_gradient_variance(
    qnode,
    weights: np.ndarray,
    sample_input: np.ndarray,
    n_samples: int = 100,
) -> dict[str, float]:
    """Estimate gradient variance to detect barren plateaus.

    Returns dict with mean and variance of gradient magnitudes across
    random parameter initialisations.
    """
    grad_norms = []
    for _ in range(n_samples):
        w = np.random.randn(*weights.shape) * 0.5
        w_tensor = torch.tensor(w, dtype=torch.float32, requires_grad=True)
        x_tensor = torch.tensor(sample_input, dtype=torch.float32)
        result = qnode(x_tensor, w_tensor)
        if isinstance(result, (list, tuple)):
            loss = sum(r for r in result)
        else:
            loss = result.sum()
        loss.backward()
        grad_norms.append(float(w_tensor.grad.norm().item()))

    return {
        "mean_grad_norm": float(np.mean(grad_norms)),
        "var_grad_norm": float(np.var(grad_norms)),
        "std_grad_norm": float(np.std(grad_norms)),
    }


# ──────────────────────────────────────────────────────────
#  Circuit info
# ──────────────────────────────────────────────────────────

def get_circuit_info(
    n_qubits: int,
    n_layers: int,
    ansatz_type: str,
    entanglement: str = "full",
) -> dict[str, int | str]:
    """Return a summary dict of circuit properties."""
    from ..quantum.entanglement import count_cnot_gates

    n_params = count_circuit_parameters(n_qubits, n_layers, ansatz_type)
    cnots_per_layer = count_cnot_gates(n_qubits, entanglement)

    return {
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "ansatz_type": ansatz_type,
        "entanglement": entanglement,
        "n_parameters": n_params,
        "cnots_per_layer": cnots_per_layer,
        "total_cnots": cnots_per_layer * n_layers,
        "circuit_depth_estimate": n_layers * (len(["RX", "RY", "RZ"]) + 1),
    }
