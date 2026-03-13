"""
Tests for quantum layers — PennyLane quantum layer as nn.Module.
"""

import pytest
import numpy as np
import torch

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.quantum_layers import PennyLaneQuantumLayer
from src.models.classical_net import ClassicalNet
from src.models.hybrid_pennylane_net import HybridPennyLaneNet


# ═══════════════════════════════════════════════════════════
#  PennyLane Quantum Layer
# ═══════════════════════════════════════════════════════════

class TestPennyLaneQuantumLayer:
    """Test the PennyLane VQC as a PyTorch layer."""

    @pytest.fixture
    def layer(self):
        return PennyLaneQuantumLayer(
            n_qubits=3,
            n_layers=2,
            encoding_type="angle",
            ansatz_type="strongly_entangling",
            entanglement="full",
            n_outputs=2,
            diff_method="backprop",
        )

    def test_output_shape(self, layer):
        x = torch.randn(4, 3)  # batch=4, features=3
        out = layer(x)
        assert out.shape == (4, 2)

    def test_output_range(self, layer):
        """Expectation values of PauliZ are in [-1, 1]."""
        x = torch.randn(2, 3)
        out = layer(x)
        assert torch.all(out >= -1.0 - 1e-6)
        assert torch.all(out <= 1.0 + 1e-6)

    def test_gradient_flows(self, layer):
        """Verify backpropagation through the quantum circuit."""
        x = torch.randn(2, 3, requires_grad=False)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        assert layer.weights.grad is not None
        assert layer.weights.grad.shape == layer.weights.shape

    def test_parameter_count(self, layer):
        n_params = layer.count_parameters()
        # strongly_entangling: (n_layers, n_qubits, 3) = (2, 3, 3) = 18
        assert n_params == 18

    def test_draw_returns_string(self, layer):
        diagram = layer.draw()
        assert isinstance(diagram, str)
        assert len(diagram) > 0


# ═══════════════════════════════════════════════════════════
#  Classical Net
# ═══════════════════════════════════════════════════════════

class TestClassicalNet:
    """Test the pure PyTorch baseline model."""

    def test_output_shape(self):
        model = ClassicalNet(input_dim=10, hidden_dims=[32, 16], output_dim=2)
        x = torch.randn(8, 10)
        out = model(x)
        assert out.shape == (8, 2)

    def test_from_config(self):
        cfg = {
            "input_dim": 4,
            "hidden_dims": [16, 8],
            "output_dim": 2,
            "activation": "relu",
            "dropout": 0.1,
            "batch_norm": True,
        }
        model = ClassicalNet.from_config(cfg)
        x = torch.randn(4, 4)
        out = model(x)
        assert out.shape == (4, 2)

    def test_parameter_count(self):
        model = ClassicalNet(input_dim=4, hidden_dims=[8], output_dim=2, batch_norm=False, dropout=0.0)
        n = model.count_parameters()
        # Linear(4,8)=40, Linear(8,2)=18 → 58
        assert n == (4 * 8 + 8) + (8 * 2 + 2)


# ═══════════════════════════════════════════════════════════
#  Hybrid PennyLane Net
# ═══════════════════════════════════════════════════════════

class TestHybridPennyLaneNet:
    """Test the hybrid quantum-classical architecture."""

    @pytest.fixture
    def model(self):
        return HybridPennyLaneNet(
            input_dim=4,
            output_dim=2,
            classical_pre_layers=[8],
            classical_post_layers=[4],
            quantum_config={
                "n_qubits": 3,
                "n_layers": 1,
                "encoding": {"type": "angle", "rotation_axes": ["Y"]},
                "ansatz": {
                    "type": "strongly_entangling",
                    "entanglement": "linear",
                    "rotation_gates": ["RX", "RY", "RZ"],
                },
                "measurement": {"observables": ["Z"], "n_outputs": 2},
                "backend": {"name": "default.qubit", "shots": None},
                "gradient_method": "backprop",
            },
            activation="relu",
            dropout=0.0,
        )

    def test_output_shape(self, model):
        x = torch.randn(4, 4)
        out = model(x)
        assert out.shape == (4, 2)

    def test_gradient_flows_through_hybrid(self, model):
        x = torch.randn(2, 4)
        out = model(x)
        loss = out.sum()
        loss.backward()

        # Check both classical and quantum have gradients
        has_classical_grad = False
        has_quantum_grad = False
        for name, p in model.named_parameters():
            if p.grad is not None:
                if "quantum" in name or "weights" in name:
                    has_quantum_grad = True
                else:
                    has_classical_grad = True
        assert has_classical_grad
        assert has_quantum_grad

    def test_count_parameters(self, model):
        params = model.count_parameters()
        assert "total" in params
        assert "quantum" in params
        assert params["total"] > 0
        assert params["quantum"] > 0
