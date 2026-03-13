"""
Circuit analysis — expressibility and entangling capability.

Implements the metrics from Sim, Johnson & Aspuru-Guzik (2019):

    • **Expressibility** — KL divergence between the VQC state
      distribution and the Haar-random distribution on the Hilbert space.
    • **Entangling capability** — average Meyer-Wallach entanglement
      measure Q across random parameter samples.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pennylane as qml


# ──────────────────────────────────────────────────────────
#  Expressibility
# ──────────────────────────────────────────────────────────

def compute_expressibility(
    circuit_fn: Callable,
    n_qubits: int,
    weight_shape: tuple[int, ...],
    n_samples: int = 500,
    n_bins: int = 75,
) -> float:
    r"""Estimate expressibility via KL divergence.

    Samples pairs of random parameters :math:`(\theta, \phi)`, computes
    fidelities :math:`|\langle\psi(\theta)|\psi(\phi)\rangle|^2`,
    and compares the histogram to the Haar-random fidelity distribution
    :math:`P_{\mathrm{Haar}}(F) = (2^n - 1)(1 - F)^{2^n - 2}`.

    Lower KL → more expressible circuit.

    Parameters
    ----------
    circuit_fn : callable
        A function ``circuit_fn(weights) -> statevector`` that returns
        the state vector for given parameters.
    n_qubits : int
    weight_shape : tuple
    n_samples : int
    n_bins : int

    Returns
    -------
    float
        KL divergence (expressibility).
    """
    fidelities = []
    for _ in range(n_samples):
        theta = np.random.uniform(0, 2 * np.pi, size=weight_shape)
        phi = np.random.uniform(0, 2 * np.pi, size=weight_shape)
        psi_theta = circuit_fn(theta)
        psi_phi = circuit_fn(phi)
        fid = np.abs(np.dot(np.conj(psi_theta), psi_phi)) ** 2
        fidelities.append(fid)

    fidelities = np.array(fidelities)

    # Empirical histogram
    hist, bin_edges = np.histogram(fidelities, bins=n_bins, range=(0, 1), density=True)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Haar-random distribution: P(F) = (2^n - 1)(1 - F)^{2^n - 2}
    dim = 2 ** n_qubits
    haar_pdf = (dim - 1) * (1 - bin_centres) ** (dim - 2)

    # KL divergence (with epsilon for numerical stability)
    eps = 1e-10
    hist = hist + eps
    haar_pdf = haar_pdf + eps
    kl = float(np.sum(hist * np.log(hist / haar_pdf)) * (bin_edges[1] - bin_edges[0]))

    return kl


# ──────────────────────────────────────────────────────────
#  Entangling capability (Meyer-Wallach measure)
# ──────────────────────────────────────────────────────────

def compute_entangling_capability(
    circuit_fn: Callable,
    n_qubits: int,
    weight_shape: tuple[int, ...],
    n_samples: int = 200,
) -> float:
    r"""Estimate the Meyer-Wallach entanglement measure Q.

    .. math::

        Q = \frac{2}{n} \sum_{k=1}^{n} \left(
            1 - \mathrm{Tr}\left[\rho_k^2\right]
        \right)

    averaged over random parameter samples. Higher Q → more entanglement.

    Parameters
    ----------
    circuit_fn : callable
        ``circuit_fn(weights) -> statevector`` (length ``2**n_qubits``).
    n_qubits : int
    weight_shape : tuple
    n_samples : int

    Returns
    -------
    float
        Average Meyer-Wallach measure ∈ [0, 1].
    """
    q_values = []
    for _ in range(n_samples):
        weights = np.random.uniform(0, 2 * np.pi, size=weight_shape)
        state = circuit_fn(weights)
        q_values.append(_meyer_wallach(state, n_qubits))

    return float(np.mean(q_values))


def _meyer_wallach(state: np.ndarray, n_qubits: int) -> float:
    """Compute Meyer-Wallach Q for a single state vector."""
    dim = 2 ** n_qubits
    state = state.reshape(dim)
    rho = np.outer(state, np.conj(state))

    q_sum = 0.0
    for k in range(n_qubits):
        rho_k = _partial_trace(rho, k, n_qubits)
        purity = np.real(np.trace(rho_k @ rho_k))
        q_sum += 1.0 - purity

    return float(2.0 * q_sum / n_qubits)


def _partial_trace(rho: np.ndarray, keep: int, n_qubits: int) -> np.ndarray:
    """Partial trace: keep qubit *keep*, trace out the rest."""
    dim = 2 ** n_qubits
    rho = rho.reshape([2] * (2 * n_qubits))

    # Move kept qubit to first position
    axes_order = list(range(2 * n_qubits))
    # Bra indices: 0..n-1, Ket indices: n..2n-1
    # We want to trace over all except qubit `keep`
    trace_axes = [i for i in range(n_qubits) if i != keep]

    # Trace by contracting pairs
    result = rho
    offset = 0
    for ax in sorted(trace_axes, reverse=True):
        # axis1: bra index (shifted left by number of removed axes = offset)
        # axis2: corresponding ket index (originally ax + n_qubits),
        # after removing `offset` pairs the ket index shifts left by `offset` as well.
        axis1 = ax - offset
        axis2 = ax + n_qubits - offset
        result = np.trace(result, axis1=axis1, axis2=axis2)
        offset += 1

    return result.reshape(2, 2)


# ──────────────────────────────────────────────────────────
#  Helper: build statevector function from PennyLane circuit
# ──────────────────────────────────────────────────────────

def make_statevector_fn(
    n_qubits: int,
    n_layers: int,
    ansatz_type: str = "strongly_entangling",
    entanglement: str = "full",
    encoding_type: str | None = None,
) -> Callable:
    """Return a function ``fn(weights) -> statevector`` for analysis.

    Parameters
    ----------
    n_qubits, n_layers, ansatz_type, entanglement
        Circuit configuration.
    encoding_type : str | None
        If provided, prepend a data encoding with zero-inputs.

    Returns
    -------
    callable
    """
    from ..quantum.circuits import build_ansatz

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="numpy")
    def _circuit(weights):
        build_ansatz(weights, n_qubits, n_layers, ansatz_type, entanglement)
        return qml.state()

    return _circuit
