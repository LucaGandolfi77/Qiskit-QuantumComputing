"""
Tests for quantum circuit building blocks — ansatzes, encodings, entanglement.
"""

import pytest
import numpy as np
import pennylane as qml

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.quantum.circuits import build_ansatz, get_weight_shape
from src.quantum.encodings import angle_encoding, amplitude_encoding, iqp_encoding, get_encoding_fn
from src.quantum.entanglement import (
    apply_entanglement,
    get_entanglement_pairs,
    count_cnot_gates,
)
from src.quantum.measurements import measure_expectations, build_observable_list


# ═══════════════════════════════════════════════════════════
#  Ansatz circuits
# ═══════════════════════════════════════════════════════════

class TestAnsatz:
    """Test VQC ansatz circuits."""

    @pytest.mark.parametrize(
        "ansatz_type", ["strongly_entangling", "hardware_efficient", "basic_entangler"]
    )
    def test_weight_shape(self, ansatz_type):
        n_qubits, n_layers = 4, 3
        shape = get_weight_shape(ansatz_type, n_qubits, n_layers)
        assert isinstance(shape, tuple)
        assert all(s > 0 for s in shape)

    @pytest.mark.parametrize(
        "ansatz_type", ["strongly_entangling", "hardware_efficient", "basic_entangler"]
    )
    def test_ansatz_executes(self, ansatz_type):
        """Verify ansatz runs without error inside a QNode."""
        n_qubits, n_layers = 3, 2
        dev = qml.device("default.qubit", wires=n_qubits)
        w_shape = get_weight_shape(ansatz_type, n_qubits, n_layers)
        weights = np.random.randn(*w_shape)

        @qml.qnode(dev)
        def circuit(w):
            build_ansatz(w, n_qubits, n_layers, ansatz_type, "full")
            return qml.state()

        state = circuit(weights)
        # State vector should have 2^n_qubits amplitudes
        assert state.shape == (2**n_qubits,)
        # Normalisation
        assert abs(np.linalg.norm(state) - 1.0) < 1e-6

    def test_unknown_ansatz_raises(self):
        with pytest.raises(ValueError, match="Unknown ansatz"):
            get_weight_shape("bogus_ansatz", 4, 2)


# ═══════════════════════════════════════════════════════════
#  Encodings
# ═══════════════════════════════════════════════════════════

class TestEncodings:
    """Test data encoding strategies."""

    def test_angle_encoding_executes(self):
        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        def circuit(x):
            angle_encoding(x, 3, rotation_axes=["Y"])
            return qml.state()

        state = circuit(np.array([0.5, -0.3, 0.8]))
        assert state.shape == (8,)
        assert abs(np.linalg.norm(state) - 1.0) < 1e-6

    def test_amplitude_encoding_normalises(self):
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x):
            amplitude_encoding(x, 2)
            return qml.state()

        state = circuit(np.array([3.0, 4.0, 0.0, 0.0]))
        assert abs(np.linalg.norm(state) - 1.0) < 1e-6

    def test_iqp_encoding_executes(self):
        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        def circuit(x):
            iqp_encoding(x, 3, n_repeats=1)
            return qml.state()

        state = circuit(np.array([0.1, 0.2, 0.3]))
        assert state.shape == (8,)

    def test_encoding_dispatcher(self):
        fn = get_encoding_fn("angle", rotation_axes=["X"])
        assert callable(fn)

    def test_unknown_encoding_raises(self):
        with pytest.raises(ValueError, match="Unknown encoding"):
            get_encoding_fn("quantum_teleportation")


# ═══════════════════════════════════════════════════════════
#  Entanglement patterns
# ═══════════════════════════════════════════════════════════

class TestEntanglement:
    """Test entanglement pattern utilities."""

    @pytest.mark.parametrize("pattern", ["full", "linear", "circular"])
    def test_entanglement_pair_count(self, pattern):
        pairs = get_entanglement_pairs(4, pattern)
        n_cnots = count_cnot_gates(4, pattern)
        assert len(pairs) == n_cnots

    def test_full_entanglement_count(self):
        # n*(n-1)/2 pairs
        assert count_cnot_gates(4, "full") == 6

    def test_linear_entanglement_count(self):
        assert count_cnot_gates(4, "linear") == 3

    def test_circular_entanglement_count(self):
        assert count_cnot_gates(4, "circular") == 4

    def test_unknown_pattern_raises(self):
        with pytest.raises(ValueError, match="Unknown entanglement"):
            get_entanglement_pairs(4, "star")


# ═══════════════════════════════════════════════════════════
#  Measurements
# ═══════════════════════════════════════════════════════════

class TestMeasurements:
    """Test measurement utilities."""

    def test_observable_list_builder(self):
        labels = build_observable_list(3, observables=["Z"], n_outputs=2)
        assert len(labels) == 2
        # First label should be ZII (qubit 0 measured, others identity)
        assert labels[0] == "ZII"
        assert labels[1] == "IZI"

    def test_measure_expectations_qnode(self):
        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        def circuit():
            return measure_expectations(3, observables=["Z"], n_outputs=2)

        result = circuit()
        assert len(result) == 2
