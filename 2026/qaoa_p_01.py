"""
Environment setup

This file previously contained raw `pip install ...` lines at the top which
make the file invalid as Python. To install the required packages, either:

1) Run the helper script (recommended):

	bash 2026/setup_qiskit_env.sh

2) Or install into an existing environment with pip:

	python3 -m pip install -r 2026/requirements_qaoa.txt

After packages are installed you can run this script normally.
"""

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Sampler
from qiskit.circuit.library import qaoa_ansatz
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA, SPSA
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


# Esempio: x ∈ [0, 5] con 4 bit → 16 valori discreti
# x = 2.7 → [1,0,1,1] → decodifica a ≈2.67

# x ∈ {0.0, 0.5, 1.0, 1.5, 2.0} → 5 qubit
# [0,1,0,0,0] → x = 0.5

# Vincolo: sum(bits) = 1 (one-hot)
penalty = 10.0 * (sum(Z_i) - constant)^2

# Binary encoding con 3 bit per variabile
# 3 bit → 8 valori: [0.0, 0.14, 0.28, 0.43, 0.57, 0.71, 0.86, 1.0]

# Totale variabili continue: 3 (s) + 15 (v) = 18
# Totale qubit necessari: 18 × 3 = 54 qubit
# ✓ Fattibile su IBM Quantum (127+ qubit disponibili)
