"""Model architectures: classical, hybrid, quantum layers."""

from .classical_net import ClassicalNet
from .quantum_layers import PennyLaneQuantumLayer

__all__ = ["ClassicalNet", "PennyLaneQuantumLayer"]
