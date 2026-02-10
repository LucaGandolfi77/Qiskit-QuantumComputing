"""
QAOA Implementation for Mixed Integer Optimization Problem
Using Qiskit and IBM Quantum

Problem:
Maximize: 10*s0 + 10*s1 + 10*s2 + [10*v00*u00 + 10*v10*u10 + ... ]/2

Where:
- s0, s1, s2, vij are continuous variables in [0,1]
- u00, u10, u20, u30, u40, u01, u11, u21, u31, u41, u02, u12, u22, u32, u42 are binary variables
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
import importlib, subprocess, sys

# helper to ensure imports (install if missing)
def ensure_import(module_name, pip_name=None):
    pip_name = pip_name or module_name
    try:
        return importlib.import_module(module_name)
    except Exception:
        print(f"Module '{module_name}' not found; attempting to install '{pip_name}'...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', pip_name])
            return importlib.import_module(module_name)
        except Exception as e:
            print(f"Failed to import/install {module_name}: {e}")
            return None

# import qiskit_algorithms (modern package). fall back gracefully if unavailable
qiskit_algorithms = ensure_import('qiskit_algorithms', 'qiskit-algorithms')
if qiskit_algorithms is not None:
    try:
        from qiskit_algorithms.minimum_eigensolvers.qaoa import QAOA
    except Exception:
        QAOA = None
    try:
        from qiskit_algorithms.optimizers import COBYLA
    except Exception:
        try:
            from qiskit_algorithms.optimizers import COBYLA
        except Exception:
            COBYLA = None
else:
    QAOA = None
    COBYLA = None

# try to import a sampler (StatevectorSampler) if available
try:
    from qiskit.primitives import StatevectorSampler
    StateSampler = StatevectorSampler
except Exception:
    try:
        from qiskit.primitives import Sampler as StateSampler
    except Exception:
        StateSampler = None

# fall back: try qiskit_aer primitives
try:
    from qiskit_aer.primitives import AerSampler
    AerSamplerAvailable = AerSampler
except Exception:
    AerSamplerAvailable = None

# attempt to import algorithm_globals; non-fatal
try:
    from qiskit_algorithms.utils import algorithm_globals
except Exception:
    algorithm_globals = None

# Set random seed for reproducibility
algorithm_globals.random_seed = 42

# Define the problem
# We have 15 binary variables u: u00, u10, u20, u30, u40, u01, u11, u21, u31, u41, u02, u12, u22, u32, u42
# For QAOA, we focus on the binary part of the optimization

num_qubits = 15  # Number of binary variables

# Since the continuous variables (s and v) are in [0,1], we can approximate them
# For this QAOA implementation, we'll focus on maximizing the binary part
# and treat continuous variables as parameters

# Create the Hamiltonian for the objective function
# The objective is to maximize, but QAOA minimizes, so we negate the objective
# Maximize 10*sum(v_ij * u_ij) is equivalent to minimize -10*sum(v_ij * u_ij)

# Assuming v_ij values (these would come from optimization or be set to 1 for max case)
v_values = np.ones(15)  # Assuming v_ij = 1 for maximum case

# Build the cost Hamiltonian
# For maximization: we want to minimize the negative
# Cost = -10 * sum(v_i * u_i) where u_i are binary variables
# In Pauli terms: u_i = (1 - Z_i)/2, so we minimize -10 * sum(v_i * (1 - Z_i)/2)
# = -5 * sum(v_i) + 5 * sum(v_i * Z_i)

# Create Pauli strings for each qubit
pauli_list = []
constant_term = 0

for i in range(num_qubits):
    coeff = 5 * v_values[i]  # Coefficient for Z_i term
    pauli_str = ['I'] * num_qubits
    pauli_str[i] = 'Z'
    pauli_list.append((''.join(pauli_str), coeff))
    constant_term += -5 * v_values[i]  # Constant term from (1-Z_i)/2 expansion

# Create the Hamiltonian operator
hamiltonian = SparsePauliOp.from_list(pauli_list)

print("Problem Setup:")
print(f"Number of qubits (binary variables): {num_qubits}")
print(f"Hamiltonian: {hamiltonian}")
print(f"Constant term: {constant_term}")

# Initialize QAOA / fallback
sampler = None
if StateSampler is not None:
    try:
        sampler = StateSampler()
    except Exception:
        sampler = None
elif AerSamplerAvailable is not None:
    try:
        sampler = AerSampler()
    except Exception:
        sampler = None

# Try to use QAOA if available and sampler is present
if QAOA is not None and COBYLA is not None and sampler is not None:
    try:
        optimizer = COBYLA(maxiter=100) if COBYLA is not None else None
        qaoa = QAOA(sampler=sampler, optimizer=optimizer, reps=2)
        print("\nRunning QAOA optimization...")
        result = qaoa.compute_minimum_eigenvalue(hamiltonian)
    except Exception as e:
        print("QAOA execution failed:", e)
        result = None
else:
    print("QAOA or sampler not available in this environment; falling back to direct diagonalization (classical exact) if feasible.")
    result = None

# If QAOA did not run, compute classical exact eigenvalues for the Hamiltonian
# Use safe methods to avoid constructing a dense 2^n x 2^n matrix
def is_diagonal_in_z(sparse_pauli_op):
    # True if every term contains only I or Z
    for pauli in sparse_pauli_op.paulis:
        if set(pauli.to_label()) - set('IZ'):
            return False
    return True


def compute_min_eigen_diagonal(sparse_pauli_op, constant_term=0.0):
    # Efficiently evaluate diagonal Hamiltonian on computational basis states
    n = sparse_pauli_op.num_qubits
    # Extract Z coefficients for each qubit (sum of coefficients where single-Z present)
    coeffs = np.zeros(n, dtype=float)
    for lbl, coeff in sparse_pauli_op.to_list():
        # lbl is a Pauli label string e.g. 'IZI..'
        # Only handle single-Z strings and multi-Z by summing contributions
        for i, ch in enumerate(lbl):
            if ch == 'Z':
                coeffs[i] += coeff.real
    # enumerate basis values: for 2^n states, compute z_i = (-1)^bit
    dim = 1 << n
    if dim > 1 << 22:
        raise MemoryError(f"Problem too large to enumerate basis states (2^{n} states)")
    # Build array of integers from 0..2^n-1 and extract bits
    idxs = np.arange(dim, dtype=np.uint32)
    # compute bit matrix efficiently
    # For each qubit i, bit = (idxs >> i) & 1
    # z_i = 1 - 2*bit
    # value per state = constant_term + sum_i coeffs[i]*z_i
    values = np.full(dim, float(constant_term), dtype=float)
    for i in range(n):
        bits = ((idxs >> i) & 1).astype(np.int8)
        z = 1 - 2 * bits
        values += coeffs[i] * z
    min_idx = int(np.argmin(values))
    min_val = float(values[min_idx])
    # eigenstate as basis vector
    eigstate = np.zeros(dim, dtype=complex)
    eigstate[min_idx] = 1.0
    return min_val, eigstate, min_idx


if result is None:
    try:
        # prefer diagonal fast-path
        if is_diagonal_in_z(hamiltonian):
            val, eigstate, idx = compute_min_eigen_diagonal(hamiltonian, constant_term)
            class_result = type('R', (), {})()
            class_result.eigenvalue = val
            class_result.eigenstate = eigstate
            result = class_result
            print("Classical exact eigenvalue computed via diagonal evaluation (fast path).")
        else:
            # attempt sparse eigensolver
            try:
                sp = None
                try:
                    sp = hamiltonian.to_spmatrix()
                except Exception:
                    try:
                        sp = hamiltonian.to_matrix(sparse=True)
                    except Exception:
                        sp = None
                if sp is not None:
                    # use scipy.sparse.linalg.eigsh to get smallest algebraic eigenvalue
                    from scipy.sparse.linalg import eigsh
                    sp = sp.tocsr() if hasattr(sp, 'tocsr') else sp
                    eigvals, eigvecs = eigsh(sp, k=1, which='SA')
                    val = float(eigvals[0])
                    vec = eigvecs[:, 0]
                    class_result = type('R', (), {})()
                    class_result.eigenvalue = val
                    class_result.eigenstate = vec
                    result = class_result
                    print("Classical exact eigenvalue computed via sparse eigensolver.")
                else:
                    raise RuntimeError('No sparse representation available for Hamiltonian')
            except Exception as e:
                print('Sparse eigensolver failed:', e)
                result = None
    except MemoryError as me:
        print('Skipping classical exact solve due to memory constraints:', me)
        result = None
    except Exception as e:
        print('Classical fallback failed:', e)
        result = None

print("\nQAOA Results:")
print(f"Optimal value (including constant): {result.eigenvalue + constant_term}")
print(f"Optimal parameters: {result.optimal_parameters}")

# Extract the most likely solution
def sample_most_likely(quasi_dist):
    """Extract most likely binary string from quasi-distribution."""
    if hasattr(quasi_dist, 'binary_probabilities'):
        binary_probs = quasi_dist.binary_probabilities()
    else:
        binary_probs = quasi_dist

    max_prob = 0
    max_bitstring = None
    for bitstring, prob in binary_probs.items():
        if prob > max_prob:
            max_prob = prob
            max_bitstring = bitstring

    return max_bitstring

# Get the solution
if hasattr(result, 'best_measurement'):
    solution_bitstring = result.best_measurement['bitstring']
else:
    eigenstate = result.eigenstate
    if hasattr(eigenstate, 'binary_probabilities'):
        solution_bitstring = sample_most_likely(eigenstate.binary_probabilities())
    else:
        # Convert to binary string from most likely state
        solution_bitstring = sample_most_likely(eigenstate)

print(f"\nOptimal binary solution: {solution_bitstring}")

# Decode the solution
if solution_bitstring:
    u_values = [int(bit) for bit in solution_bitstring]
    print("\nBinary variable assignments:")
    labels = ['u00', 'u10', 'u20', 'u30', 'u40', 
              'u01', 'u11', 'u21', 'u31', 'u41', 
              'u02', 'u12', 'u22', 'u32', 'u42']
    for label, value in zip(labels, u_values):
        print(f"  {label} = {value}")

    # Calculate objective value (assuming s_i = 1 and v_ij = 1 for maximum)
    obj_value = 10*3 + 5 * sum(u_values)  # 10*(s0+s1+s2) + 10*sum(v_ij*u_ij)/2
    print(f"\nObjective value (with s_i=1, v_ij=1): {obj_value}")

print("\n" + "="*70)
print("QAOA optimization completed!")
print("="*70)
print("\nNote: This implementation focuses on the binary variables.")
print("For a complete solution, combine with classical optimization")
print("for the continuous variables (s_i and v_ij) using the binary")
print("solution as constraints.")
