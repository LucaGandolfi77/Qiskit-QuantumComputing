"""
QAOA Implementation for Mixed Integer Optimization Problem - OPTIMIZED VERSION
Using Qiskit and IBM Quantum with Performance Monitoring

Problem:
Maximize: 10*s0 + 10*s1 + 10*s2 + [10*v00*u00 + 10*v10*u10 + ... ]/2

OPTIMIZATIONS:
- Performance monitoring wrapper (RAM, CPU, tempo)
- Efficient sparse matrix construction (dict-based, no SparseEfficiencyWarning)
- Memory-efficient diagonal Hamiltonian evaluation
- Graceful fallback chain
"""

import numpy as np
import time
import os
from functools import wraps
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
import importlib, subprocess, sys

# ============================================================================
# PERFORMANCE MONITORING WRAPPER
# ============================================================================

def monitor_performance(func):
    """Decorator to monitor execution time, RAM and CPU usage"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Try to import psutil for advanced monitoring
        try:
            import psutil
            process = psutil.Process(os.getpid())
            has_psutil = True
        except ImportError:
            has_psutil = False
            print("⚠️  psutil not available. Install with: pip install psutil")

        # Memory before execution
        if has_psutil:
            mem_before = process.memory_info().rss / 1024**2  # MB
            print(f"\n{'='*70}")
            print(f"🚀 STARTING: {func.__name__}")
            print(f"   RAM before: {mem_before:.1f} MB")
            print(f"{'='*70}\n")
        else:
            print(f"\n🚀 STARTING: {func.__name__}\n")

        # Execution timing
        start_time = time.perf_counter()

        # Execute function
        result = func(*args, **kwargs)

        # Timing results
        end_time = time.perf_counter()
        elapsed = end_time - start_time

        # Memory after execution
        if has_psutil:
            mem_after = process.memory_info().rss / 1024**2  # MB
            mem_delta = mem_after - mem_before
            cpu_percent = process.cpu_percent(interval=0.1)

            print(f"\n{'='*70}")
            print(f"✅ COMPLETED: {func.__name__}")
            print(f"   Execution time: {elapsed:.3f} seconds")
            print(f"   RAM after: {mem_after:.1f} MB")
            print(f"   RAM delta: {mem_delta:+.1f} MB")
            print(f"   CPU usage: {cpu_percent:.1f}%")
            print(f"{'='*70}\n")
        else:
            print(f"\n✅ COMPLETED in {elapsed:.3f} seconds\n")

        return result
    return wrapper

# ============================================================================
# DEPENDENCY MANAGEMENT
# ============================================================================

def ensure_import(module_name, pip_name=None):
    """Helper to ensure imports (install if missing)"""
    pip_name = pip_name or module_name
    try:
        return importlib.import_module(module_name)
    except Exception:
        print(f"📦 Module '{module_name}' not found; attempting to install '{pip_name}'...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', pip_name, '-q'])
            return importlib.import_module(module_name)
        except Exception as e:
            print(f"❌ Failed to import/install {module_name}: {e}")
            return None

# Import qiskit_algorithms (modern package)
qiskit_algorithms = ensure_import('qiskit_algorithms', 'qiskit-algorithms')
if qiskit_algorithms is not None:
    try:
        from qiskit_algorithms.minimum_eigensolvers.qaoa import QAOA
    except Exception:
        QAOA = None
    try:
        from qiskit_algorithms.optimizers import COBYLA
    except Exception:
        COBYLA = None
    try:
        from qiskit_algorithms.utils import algorithm_globals
        algorithm_globals.random_seed = 42
    except Exception:
        algorithm_globals = None
else:
    QAOA = None
    COBYLA = None
    algorithm_globals = None

# Try to import a sampler
try:
    from qiskit.primitives import StatevectorSampler
    StateSampler = StatevectorSampler
except Exception:
    try:
        from qiskit.primitives import Sampler as StateSampler
    except Exception:
        StateSampler = None

# Fallback: qiskit_aer primitives
try:
    from qiskit_aer.primitives import Sampler as AerSampler
    AerSamplerAvailable = AerSampler
except Exception:
    AerSamplerAvailable = None

# ============================================================================
# OPTIMIZED HAMILTONIAN CONSTRUCTION
# ============================================================================

@monitor_performance
def build_hamiltonian_optimized(num_qubits, v_values):
    """
    Build Hamiltonian using dict-based construction to avoid SparseEfficiencyWarning
    Uses efficient sparse list creation instead of incremental modifications
    """
    print(f"🔧 Building Hamiltonian for {num_qubits} qubits...")

    # OPTIMIZED: Build as list of tuples directly (no incremental sparse modifications)
    pauli_dict = {}
    constant_term = 0.0

    for i in range(num_qubits):
        coeff = 5.0 * v_values[i]  # Coefficient for Z_i term

        # Build Pauli string efficiently
        pauli_key = ['I'] * num_qubits
        pauli_key[i] = 'Z'
        pauli_string = ''.join(pauli_key)

        # Store in dict (avoids repeated sparse matrix modifications)
        pauli_dict[pauli_string] = coeff
        constant_term += -5.0 * v_values[i]

    # Create Hamiltonian from complete list (single sparse construction)
    hamiltonian = SparsePauliOp.from_list(list(pauli_dict.items()))

    print(f"✓ Hamiltonian built: {hamiltonian}")
    print(f"✓ Constant term: {constant_term:.2f}")

    return hamiltonian, constant_term

# ============================================================================
# DIAGONAL HAMILTONIAN SOLVER (MEMORY EFFICIENT)
# ============================================================================

def is_diagonal_in_z(sparse_pauli_op):
    """Check if Hamiltonian is diagonal in computational basis"""
    for pauli in sparse_pauli_op.paulis:
        if set(pauli.to_label()) - set('IZ'):
            return False
    return True

@monitor_performance
def compute_min_eigen_diagonal(sparse_pauli_op, constant_term=0.0):
    """
    Memory-efficient diagonal Hamiltonian evaluation
    Avoids constructing full 2^n x 2^n matrix
    """
    n = sparse_pauli_op.num_qubits

    # Extract Z coefficients efficiently
    coeffs = np.zeros(n, dtype=np.float64)
    for lbl, coeff in sparse_pauli_op.to_list():
        for i, ch in enumerate(lbl):
            if ch == 'Z':
                coeffs[i] += coeff.real

    # Enumerate basis states efficiently
    dim = 1 << n
    if dim > 1 << 20:  # Safety limit: 2^20 = 1M states
        raise MemoryError(f"Problem too large: 2^{n} = {dim} states")

    print(f"🔢 Evaluating {dim} basis states...")

    # Vectorized computation
    idxs = np.arange(dim, dtype=np.uint32)
    values = np.full(dim, constant_term, dtype=np.float64)

    for i in range(n):
        bits = ((idxs >> i) & 1).astype(np.int8)
        z = 1 - 2 * bits
        values += coeffs[i] * z

    min_idx = int(np.argmin(values))
    min_val = float(values[min_idx])

    # Create eigenstate
    eigstate = np.zeros(dim, dtype=np.complex128)
    eigstate[min_idx] = 1.0

    print(f"✓ Minimum found at state {min_idx}: {min_val:.4f}")

    return min_val, eigstate, min_idx

# ============================================================================
# QAOA EXECUTION
# ============================================================================

@monitor_performance
def run_qaoa(hamiltonian, sampler, reps=3):
    """Execute QAOA with monitoring"""
    if QAOA is None or COBYLA is None:
        raise RuntimeError("QAOA or COBYLA not available")

    optimizer = COBYLA(maxiter=100, tol=1e-6)
    qaoa = QAOA(sampler=sampler, optimizer=optimizer, reps=reps)

    print(f"⚛️  Running QAOA with {reps} layers...")
    result = qaoa.compute_minimum_eigenvalue(hamiltonian)

    return result

# ============================================================================
# SOLUTION EXTRACTION
# ============================================================================

def extract_solution(result, eigenstate_idx=None):
    """Extract binary solution from result"""
    if hasattr(result, 'best_measurement'):
        solution_bitstring = result.best_measurement['bitstring']
    elif eigenstate_idx is not None:
        # From classical diagonalization
        solution_bitstring = format(eigenstate_idx, f'0{15}b')
    elif hasattr(result, 'eigenstate'):
        eigenstate = result.eigenstate
        if isinstance(eigenstate, np.ndarray):
            # Find max amplitude state
            max_idx = int(np.argmax(np.abs(eigenstate)))
            solution_bitstring = format(max_idx, f'0{15}b')
        else:
            solution_bitstring = None
    else:
        solution_bitstring = None

    return solution_bitstring

# ============================================================================
# MAIN EXECUTION
# ============================================================================

@monitor_performance
def main():
    """Main execution with complete monitoring"""

    print("\n" + "="*70)
    print("QAOA OPTIMIZATION - MONITORED & OPTIMIZED")
    print("="*70)

    # Problem setup
    num_qubits = 15
    v_values = np.ones(num_qubits)

    print(f"\n📊 Problem Setup:")
    print(f"   Binary variables: {num_qubits}")
    print(f"   v_ij values: all = 1.0 (maximum case)")

    # Build Hamiltonian (OPTIMIZED)
    hamiltonian, constant_term = build_hamiltonian_optimized(num_qubits, v_values)

    # Initialize sampler
    sampler = None
    if StateSampler is not None:
        try:
            sampler = StateSampler()
            print(f"\n✓ Sampler initialized: {type(sampler).__name__}")
        except Exception as e:
            print(f"⚠️  StateSampler failed: {e}")

    if sampler is None and AerSamplerAvailable is not None:
        try:
            sampler = AerSamplerAvailable()
            print(f"✓ Fallback sampler: {type(sampler).__name__}")
        except Exception as e:
            print(f"⚠️  AerSampler failed: {e}")

    # Try QAOA execution
    result = None
    eigenstate_idx = None

    if QAOA is not None and sampler is not None:
        try:
            result = run_qaoa(hamiltonian, sampler, reps=3)
            print(f"\n✅ QAOA succeeded!")
            print(f"   Eigenvalue: {result.eigenvalue:.4f}")
            print(f"   Total energy: {result.eigenvalue + constant_term:.4f}")
        except Exception as e:
            print(f"\n⚠️  QAOA failed: {e}")
            result = None
    else:
        print(f"\n⚠️  QAOA not available - using classical fallback")

    # Classical fallback
    if result is None:
        print(f"\n🔄 Classical exact solver...")
        try:
            if is_diagonal_in_z(hamiltonian):
                val, eigstate, eigenstate_idx = compute_min_eigen_diagonal(hamiltonian, constant_term)
                class_result = type('Result', (), {})()
                class_result.eigenvalue = val
                class_result.eigenstate = eigstate
                result = class_result
                print(f"✓ Classical solution found (diagonal evaluation)")
            else:
                # Sparse eigensolver
                from scipy.sparse.linalg import eigsh
                sp = hamiltonian.to_matrix(sparse=True).tocsr()
                eigvals, eigvecs = eigsh(sp, k=1, which='SA')
                val = float(eigvals[0])
                vec = eigvecs[:, 0]
                class_result = type('Result', (), {})()
                class_result.eigenvalue = val
                class_result.eigenstate = vec
                result = class_result
                print(f"✓ Classical solution found (sparse eigensolver)")
        except Exception as e:
            print(f"❌ Classical fallback failed: {e}")
            return

    # Extract solution
    solution_bitstring = extract_solution(result, eigenstate_idx)

    if solution_bitstring:
        print(f"\n🎯 Optimal binary solution: {solution_bitstring}")

        u_values = [int(bit) for bit in solution_bitstring]
        labels = ['u00', 'u10', 'u20', 'u30', 'u40', 
                  'u01', 'u11', 'u21', 'u31', 'u41', 
                  'u02', 'u12', 'u22', 'u32', 'u42']

        print(f"\n📋 Binary variable assignments:")
        for i, (label, value) in enumerate(zip(labels, u_values)):
            if i % 5 == 0:
                print()
            print(f"   {label}={value}", end="")
        print()

        obj_value = 10*3 + 5 * sum(u_values)
        print(f"\n💰 Objective value: {obj_value:.1f}")
        print(f"   (assuming s_i=1, v_ij=1)")
    else:
        print(f"\n⚠️  No solution extracted")

    print(f"\n{'='*70}")
    print(f"OPTIMIZATION COMPLETED")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
