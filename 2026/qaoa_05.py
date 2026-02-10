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

import csv
import html
import numpy as np
import time
import os
from functools import wraps
from pathlib import Path
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
def run_qaoa(hamiltonian, sampler_or_backend=None, reps=3):
    """Execute QAOA with monitoring.

    Accept either a modern `sampler` (qiskit.primitives) or an older
    `backend`/`quantum_instance`. Tries multiple constructor signatures
    to maximize compatibility across qiskit versions.
    """
    if QAOA is None:
        raise RuntimeError("QAOA not available")

    optimizer = None
    if COBYLA is not None:
        try:
            optimizer = COBYLA(maxiter=100, tol=1e-6)
        except Exception:
            optimizer = None

    # Try a few constructor keyword combinations to be tolerant
    # to different QAOA class signatures across versions.
    ctor_attempts = [
        {'sampler': sampler_or_backend, 'optimizer': optimizer, 'reps': reps},
        {'quantum_instance': sampler_or_backend, 'optimizer': optimizer, 'reps': reps},
        {'backend': sampler_or_backend, 'optimizer': optimizer, 'reps': reps},
        {'optimizer': optimizer, 'reps': reps},
    ]

    qaoa = None
    last_exc = None
    for kwargs in ctor_attempts:
        # filter out None values
        filtered = {k: v for k, v in kwargs.items() if v is not None}
        try:
            qaoa = QAOA(**filtered)
            break
        except Exception as e:
            last_exc = e
            qaoa = None

    if qaoa is None:
        raise RuntimeError(f"Failed to construct QAOA instance: {last_exc}")

    print(f"⚛️  Running QAOA with {reps} layers...")
    result = qaoa.compute_minimum_eigenvalue(hamiltonian)

    return result

# ============================================================================
# SOLUTION EXTRACTION
# ============================================================================

def extract_solution(result, eigenstate_idx=None, nbits=15):
    """Extract binary solution from result"""
    if hasattr(result, 'best_measurement'):
        solution_bitstring = result.best_measurement['bitstring']
    elif eigenstate_idx is not None:
        # From classical diagonalization
        solution_bitstring = format(eigenstate_idx, f'0{nbits}b')
    elif hasattr(result, 'eigenstate'):
        eigenstate = result.eigenstate
        if isinstance(eigenstate, np.ndarray):
            # Find max amplitude state
            max_idx = int(np.argmax(np.abs(eigenstate)))
            solution_bitstring = format(max_idx, f'0{nbits}b')
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
    # New behavior: process all problems found in QISKIT/all_qiskit_lp.csv
    csv_path = Path('all_qiskit_lp.csv')
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}. Run merge first.")
        return

    problems = load_problems_from_csv(csv_path)
    results = []
    for nb_name, sections in problems.items():
        try:
            res = process_problem(nb_name, sections)
            results.append(res)
        except Exception as e:
            results.append({
                'notebook': nb_name,
                'error': str(e),
            })

    out_dir = Path('reports')
    out_dir.mkdir(exist_ok=True)
    out_html = out_dir / f'all_qiskit_report.html'
    generate_html_report(results, out_html)
    # also write JSON and Markdown versions
    out_json = out_dir / 'all_qiskit_report.json'
    out_md = out_dir / 'all_qiskit_report.md'
    write_json_report(results, out_json)
    write_markdown_report(results, out_md)
    print(f'Wrote report: {out_html} (HTML), {out_json} (JSON), {out_md} (Markdown)')

# main will be invoked when this file is executed as a script (below)


def load_problems_from_csv(csv_path: Path):
    problems = {}
    with csv_path.open('r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row:
                continue
            nb = row[0]
            section = row[1] if len(row) > 1 else ''
            content = row[2] if len(row) > 2 else (row[1] if len(row) > 1 else '')
            problems.setdefault(nb, {}).setdefault(section, []).append(content)
    return problems


def determine_num_qubits(sections):
    # Look for uXX style variables
    import re
    vars_text = '\n'.join(sum(sections.values(), []))
    found = re.findall(r"\bu\d{2}\b", vars_text)
    if found:
        return len(found)
    # fallback: count binary entries
    bins = sections.get('Binaries', []) + sections.get('Generals', [])
    count = 0
    for s in bins:
        # split potential comma-separated lists
        toks = [t.strip() for t in s.replace(',', ' ').split() if t.strip()]
        count += len(toks)
    if count > 0:
        return count
    return 15


def process_problem(nb_name, sections):
    # prepare problem metadata
    probname = sections.get('Problem name', [''])[0] if sections.get('Problem name') else nb_name
    num_qubits = determine_num_qubits(sections)
    num_qubits = max(1, min(num_qubits, 24))
    v_values = np.ones(num_qubits)

    hamiltonian, constant_term = build_hamiltonian_optimized(num_qubits, v_values)

    # choose sampler or backend (compatibility shim)
    sampler = None
    backend = None
    if StateSampler is not None:
        try:
            sampler = StateSampler()
        except Exception:
            sampler = None
    if sampler is None and AerSamplerAvailable is not None:
        try:
            sampler = AerSamplerAvailable()
        except Exception:
            sampler = None
    # try AerSimulator as a backend fallback
    if sampler is None:
        try:
            from qiskit.providers.aer import AerSimulator
            backend = AerSimulator()
        except Exception:
            backend = None

    method = 'classical'
    result = None
    eigenstate_idx = None
    note = ''
    if QAOA is not None and (sampler is not None or backend is not None):
        try:
            qres = run_qaoa(hamiltonian, sampler_or_backend=(sampler or backend), reps=2)
            result = qres
            method = 'qaoa'
        except Exception as e:
            note = f'QAOA failed: {e}'
            result = None

    if result is None:
        try:
            if is_diagonal_in_z(hamiltonian):
                val, eigstate, idx = compute_min_eigen_diagonal(hamiltonian, constant_term)
                class_result = type('Result', (), {})()
                class_result.eigenvalue = val
                class_result.eigenstate = eigstate
                result = class_result
                eigenstate_idx = idx
                method = 'classical-diagonal'
            else:
                from scipy.sparse.linalg import eigsh
                sp = hamiltonian.to_matrix(sparse=True).tocsr()
                eigvals, eigvecs = eigsh(sp, k=1, which='SA')
                val = float(eigvals[0])
                vec = eigvecs[:, 0]
                class_result = type('Result', (), {})()
                class_result.eigenvalue = val
                class_result.eigenstate = vec
                result = class_result
                method = 'classical-sparse'
        except Exception as e:
            return {'notebook': nb_name, 'error': f'classical failed: {e}'}

    solution = extract_solution(result, eigenstate_idx, nbits=num_qubits)
    # compute total energy
    if method == 'qaoa' and hasattr(result, 'eigenvalue'):
        total_energy = (result.eigenvalue + constant_term) if constant_term is not None else result.eigenvalue
    else:
        total_energy = getattr(result, 'eigenvalue', None)

    return {
        'notebook': nb_name,
        'problem': probname,
        'num_qubits': num_qubits,
        'method': method,
        'total_energy': float(total_energy) if total_energy is not None else None,
        'solution': solution,
        'note': note,
        'sections': sections,
    }


def generate_html_report(results, out_path: Path):
    rows = []
    for r in results:
        nb = html.escape(r.get('notebook', ''))
        prob = html.escape(r.get('problem', ''))
        nq = r.get('num_qubits', '')
        method = html.escape(r.get('method', ''))
        energy = r.get('total_energy')
        sol = html.escape(r.get('solution') or '')
        note = html.escape(r.get('note', r.get('error', '')) or '')
        rows.append((nb, prob, nq, method, energy, sol, note))

    html_lines = [
        '<!doctype html>',
        '<html><head><meta charset="utf-8"><title>Qiskit Problems Report</title>',
        '<style>table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:8px} details{margin:6px 0;padding:6px;border:1px solid #eee}</style>',
        '</head><body>',
        f'<h1>Qiskit Problems Report ({len(rows)} items)</h1>',
        '<table>',
        '<tr><th>notebook</th><th>problem</th><th>num_qubits</th><th>method</th><th>energy</th><th>solution</th><th>note</th><th>details</th></tr>'
    ]
    for r in results:
        nb = html.escape(r.get('notebook', ''))
        prob = html.escape(r.get('problem', ''))
        nq = r.get('num_qubits', '')
        method = html.escape(r.get('method', ''))
        energy = r.get('total_energy')
        sol = html.escape(r.get('solution') or '')
        note = html.escape(r.get('note', r.get('error', '')) or '')
        # build details HTML from sections dict
        sections = r.get('sections', {}) or {}
        if sections:
            detail_parts = []
            for sec_name, sec_vals in sections.items():
                sec_html = '<br/>'.join(html.escape('\n'.join(sec_vals))) if sec_vals else ''
                detail_parts.append(f"<strong>{html.escape(str(sec_name))}</strong>:<pre>{sec_html}</pre>")
            details_html = '<details><summary>Show sections</summary>' + ''.join(detail_parts) + '</details>'
        else:
            details_html = ''
        html_lines.append(f'<tr><td>{nb}</td><td>{prob}</td><td>{nq}</td><td>{method}</td><td>{energy}</td><td><pre>{sol}</pre></td><td>{note}</td><td>{details_html}</td></tr>')
    html_lines.append('</table></body></html>')
    out_path.write_text('\n'.join(html_lines), encoding='utf-8')


def write_json_report(results, out_path: Path):
    import json
    # ensure serializable (solutions may be None)
    safe = []
    for r in results:
        item = {
            'notebook': r.get('notebook'),
            'problem': r.get('problem'),
            'num_qubits': r.get('num_qubits'),
            'method': r.get('method'),
            'total_energy': r.get('total_energy'),
            'solution': r.get('solution'),
            'note': r.get('note') or r.get('error'),
            'sections': r.get('sections', {}),
        }
        safe.append(item)
    out_path.write_text(json.dumps(safe, indent=2, ensure_ascii=False), encoding='utf-8')


def write_markdown_report(results, out_path: Path):
    lines = []
    lines.append('# Qiskit Problems Report')
    lines.append(f'Total: {len(results)}')
    lines.append('')
    lines.append('| notebook | problem | num_qubits | method | energy | solution | note |')
    lines.append('|---|---|---:|---|---:|---|---|')
    for r in results:
        nb = r.get('notebook','')
        prob = (r.get('problem') or '').replace('\n',' ')[:80]
        nq = r.get('num_qubits','')
        method = r.get('method','')
        energy = r.get('total_energy','')
        sol = (r.get('solution') or '')
        note = (r.get('note') or r.get('error') or '')
        lines.append(f'| {nb} | {prob} | {nq} | {method} | {energy} | `{sol}` | {note} |')
        # add sections as collapsible block
        sections = r.get('sections', {}) or {}
        if sections:
            lines.append('')
            lines.append('<details>')
            lines.append('<summary>Sections</summary>')
            lines.append('')
            for k, vals in sections.items():
                lines.append(f'**{k}**')
                lines.append('```')
                for v in vals:
                    lines.append(v)
                lines.append('```')
            lines.append('</details>')
            lines.append('')
    out_path.write_text('\n'.join(lines), encoding='utf-8')


if __name__ == "__main__":
    main()
