#!/usr/bin/env python3
"""
qaoa_ex7.py

Costruisce il problema (ex7) usando DOcplex -> QuadraticProgram e lo risolve
con un solver classico (NumPyMinimumEigensolver) e con QAOA se disponibile.
Ispirato al tutorial IBM QAOA.
"""

import json
import sys
from pprint import pprint

import importlib
import subprocess

# Modeling imports with runtime install fallback
try:
    from docplex.mp.model import Model
except Exception as e:
    print("Missing docplex: install 'docplex' (pip install docplex)")
    raise


def ensure_import(module_name, pip_name=None):
    pip_name = pip_name or module_name
    try:
        return importlib.import_module(module_name)
    except Exception as e:
        print(f"Module '{module_name}' not found. Attempting to install '{pip_name}'...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', pip_name])
        except Exception as e2:
            print(f"Automatic install of '{pip_name}' failed: {e2}")
            raise
        # try import again
        return importlib.import_module(module_name)

# qiskit-optimization (needed to convert DOcplex -> QuadraticProgram)
try:
    qos = ensure_import('qiskit_optimization', 'qiskit-optimization')
    from qiskit_optimization.translators import from_docplex_mp
    from qiskit_optimization.algorithms import MinimumEigenOptimizer
except Exception as e:
    print("Failed to import or install 'qiskit-optimization'. Please install it manually.")
    raise

# MinEigen classical fallback
try:
    # modern location
    from qiskit.algorithms import NumPyMinimumEigensolver
except Exception:
    try:
        from qiskit_algorithms import NumPyMinimumEigensolver
    except Exception:
        NumPyMinimumEigensolver = None

# QAOA / Aer imports (best-effort)
QAOA = None
COBYLA = None
Aer = None
QuantumInstance = None
algorithm_globals = None
try:
    from qiskit.algorithms import QAOA
    from qiskit.algorithms.optimizers import COBYLA
    from qiskit import Aer
    from qiskit.utils import QuantumInstance, algorithm_globals
except Exception:
    # try alternate package locations
    try:
        from qiskit_algorithms import QAOA
    except Exception:
        QAOA = None
    try:
        from qiskit.algorithms.optimizers import COBYLA
    except Exception:
        COBYLA = None
    try:
        from qiskit_aer import Aer
    except Exception:
        Aer = None


def build_model():
    mdl = Model("ex7")
    # binary server switches
    s0 = mdl.binary_var(name="s0")
    s1 = mdl.binary_var(name="s1")
    s2 = mdl.binary_var(name="s2")

    # binary variables vxy
    v = {}
    for i in range(3):
        for j in range(5):
            v[(i, j)] = mdl.binary_var(name=f"v{i}{j}")

    # integer uxy (bounded: required for QUBO conversion)
    u = {}
    # set an upper bound for u variables so the quadratic expression is bounded
    UPPER_U = 10
    for i in range(3):
        for j in range(5):
            u[(i, j)] = mdl.integer_var(name=f"u{i}{j}", lb=0, ub=UPPER_U)

    pi = 10
    pd = 5

    # objective: 10*s0 + 10*s1 + 10*s2 + (10 * sum v_ij * u_ij)/2
    obj = 10 * (s0 + s1 + s2)
    sum_term = mdl.sum(10 * v[(i, j)] * u[(i, j)] for i in range(3) for j in range(5)) / 2
    mdl.maximize(obj + sum_term)

    # example constraints (kept minimal to match problem statement bounds)
    # The problem statement included only bounds; if domain constraints needed, add here.

    return mdl


def solve_classical(qp):
    # Attempt to use NumPyMinimumEigensolver only for very small problems (matrix-size limit)
    if NumPyMinimumEigensolver is None:
        print("NumPyMinimumEigensolver not available; skipping quantum-exact solver.")
        return None
    exact = MinimumEigenOptimizer(NumPyMinimumEigensolver())
    try:
        res = exact.solve(qp)
        return res
    except Exception as e:
        print(f"NumPyMinimumEigensolver failed (problem too large or other): {e}")
        return None


def solve_with_docplex(mdl):
    """Solve directly with DOcplex (classical integer programming solver)."""
    try:
        sol = mdl.solve()
        if sol is None:
            print("DOcplex did not return a solution object.")
            return None
        # extract variable values
        vals = {}
        for v in mdl.iter_variables():
            try:
                vals[v.name] = sol.get_value(v)
            except Exception:
                try:
                    vals[v.name] = v.solution_value
                except Exception:
                    vals[v.name] = None
        fval = mdl.objective_value
        return type('DocplexResult', (), {'x': vals, 'fval': fval})
    except Exception as e:
        print(f"DOcplex solve failed: {e}")
        return None


def solve_qaoa(qp, seed=123, shots=1024, reps=1, maxiter=100):
    if QAOA is None or Aer is None or QuantumInstance is None or COBYLA is None:
        print("QAOA/Aer/QuantumInstance/COBYLA not available: skipping quantum run")
        return None

    # set seed
    try:
        algorithm_globals.random_seed = seed
    except Exception:
        pass

    backend = None
    try:
        # prefer AerSimulator if exists
        backend = Aer.get_backend('aer_simulator')
    except Exception:
        try:
            backend = Aer.get_backend('qasm_simulator')
        except Exception:
            backend = None

    if backend is None:
        print("No Aer backend found; skipping QAOA")
        return None

    qi = QuantumInstance(backend, shots=shots, seed_simulator=seed, seed_transpiler=seed)

    optimizer = COBYLA(maxiter=maxiter)
    qaoa = QAOA(optimizer=optimizer, reps=reps, quantum_instance=qi)
    try:
        qaoa_opt = MinimumEigenOptimizer(qaoa)
        res = qaoa_opt.solve(qp)
        return res
    except Exception as e:
        print(f"QAOA run failed: {e}")
        return None


def main():
    mdl = build_model()
    print("Model constructed. Exporting LP (truncated):\n")
    try:
        # preferred pretty print
        try:
            lp = mdl.export_as_lp_string()
        except Exception:
            lp = mdl.to_string()
        print('\n'.join(lp.splitlines()[:80]))
    except Exception:
        pass

    qp = from_docplex_mp(mdl)

    results = {}

    # First try a direct classical solve with DOcplex (reliable for integer problems of this size)
    print('\nSolving with DOcplex (classical integer solver) ...')
    res_docplex = solve_with_docplex(mdl)
    if res_docplex is not None:
        print('DOcplex result:')
        try:
            pprint(res_docplex.x if hasattr(res_docplex, 'x') else res_docplex.__dict__)
        except Exception:
            print(res_docplex)
        results['docplex'] = { 'x': getattr(res_docplex, 'x', None), 'fval': getattr(res_docplex, 'fval', None)}
    else:
        print('DOcplex did not produce a solution; falling back to NumPyMinimumEigensolver (if small)')
        print('\nSolving with classical NumPyMinimumEigensolver (exact) ...')
        res_classic = solve_classical(qp)
        if res_classic is not None:
            print('Classical (NumPy) result:')
            pprint(res_classic.__dict__)
            results['classical'] = { 'x': getattr(res_classic, 'x', None), 'fval': getattr(res_classic, 'fval', None)}
        else:
            print('Classical solver not run')

    print('\nAttempting QAOA (simulator) ...')
    res_qaoa = solve_qaoa(qp)
    if res_qaoa is not None:
        print('QAOA result:')
        pprint(res_qaoa.__dict__)
        results['qaoa'] = { 'x': getattr(res_qaoa, 'x', None), 'fval': getattr(res_qaoa, 'fval', None)}
    else:
        print('QAOA not run or failed')

    # save results
    with open('2026/ex7_results.json', 'w', encoding='utf-8') as fh:
        json.dump(results, fh, indent=2)
    print('\nResults written to 2026/ex7_results.json')

    # attempt to generate a plot of QAOA optimizer history if available
    try:
        import matplotlib.pyplot as plt
        series = None
        # look for candidate series in result object
        if res_qaoa is not None:
            for attr in ('optimizer_history', 'optimizer_evals', 'optimizer_results', 'optimizer_eval_history'):
                seq = getattr(res_qaoa, attr, None)
                if seq:
                    # try to coerce to numeric list
                    try:
                        if isinstance(seq, (list, tuple)) and all(isinstance(x, (int, float)) for x in seq):
                            series = list(seq)
                            break
                        # handle list of dicts with 'value' or similar
                        if isinstance(seq, (list, tuple)) and all(isinstance(x, dict) for x in seq):
                            # extract numeric values if present
                            for key in ('value', 'energy', 'obj'):
                                maybe = [x.get(key) for x in seq if key in x]
                                if maybe and all(isinstance(y, (int, float)) for y in maybe):
                                    series = maybe
                                    break
                            if series is not None:
                                break
                    except Exception:
                        pass
            # deeper inspection
            raw = getattr(res_qaoa, 'raw_result', None) or getattr(res_qaoa, '_result', None) or getattr(res_qaoa, 'raw_results', None)
            if series is None and raw is not None:
                if isinstance(raw, list) and all(isinstance(x, (int, float)) for x in raw):
                    series = list(raw)
                elif isinstance(raw, dict):
                    for v in raw.values():
                        if isinstance(v, list) and all(isinstance(x, (int, float)) for x in v):
                            series = v
                            break
        if series is not None and len(series) > 0:
            plt.figure()
            plt.plot(series, marker='o')
            plt.xlabel('Iteration')
            plt.ylabel('Objective / metric')
            plt.title('QAOA optimizer history')
            plt.grid(True)
            plot_path = '2026/ex7_qaoa_history.png'
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close()
            print(f'Saved QAOA iteration plot: {plot_path}')
            results['qaoa_plot'] = plot_path
        else:
            print('No QAOA optimizer history found to plot.')
    except Exception as e:
        print('Plotting failed (matplotlib missing or unexpected data):', e)

    # create a small executed notebook that reproduces the run and includes the plot (if any)
    try:
        import nbformat
        from nbclient import NotebookClient
        nb = nbformat.v4.new_notebook()

        cells = []
        cells.append(nbformat.v4.new_markdown_cell('# ex7 run report'))
        cells.append(nbformat.v4.new_code_cell('from pprint import pprint\nprint("Reproducing ex7 run")'))
        # dump model summary
        cells.append(nbformat.v4.new_code_cell('from qaoa_ex7 import build_model, from_docplex_mp\nmdl = build_model()\nprint(mdl.export_as_lp_string()[:400])'))
        # solve and display classical
        cells.append(nbformat.v4.new_code_cell('from qaoa_ex7 import solve_classical, solve_qaoa, from_docplex_mp\nqp = from_docplex_mp(mdl)\nres = solve_classical(qp)\nprint("Classical:")\ntry:\n    pprint(res.__dict__)\nexcept Exception:\n    print(res)'))
        # QAOA cell that tries to produce the plot
        cells.append(nbformat.v4.new_code_cell('print("Attempt QAOA (simulator) ...")\nres_q = solve_qaoa(qp)\nprint(res_q)'))
        if 'qaoa_plot' in results:
            cells.append(nbformat.v4.new_markdown_cell('![QAOA history](ex7_qaoa_history.png)'))

        nb.cells = cells

        # execute the notebook
        client = NotebookClient(nb, timeout=1800, kernel_name='python3')
        try:
            # ensure proper kernel manager/client setup
            try:
                client.create_kernel_manager()
            except Exception:
                pass
            client.start_new_kernel()
            client.start_new_kernel_client()
        except Exception:
            # fallback to standard execute
            pass
        try:
            client.execute()
        except Exception as e:
            print('Notebook execution encountered an error:', e)
        exe_path = '2026/ex7_run.ipynb'
        nbformat.write(nb, exe_path)
        print('Saved executed notebook:', exe_path)
    except Exception as e:
        print('Failed to create executed notebook:', e)


if __name__ == '__main__':
    main()
