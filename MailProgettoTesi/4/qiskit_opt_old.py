# ============================================================
# Qiskit – 5 Servers / 5 VMs – Optimization Problem
# Updated for Qiskit 1.x + qiskit-algorithms + qiskit-aer
# ============================================================

# ── Package installation (executed only once) ────────────────
try:
    INSTALLED
except NameError:
    INSTALLED = None

if INSTALLED != 1:
    import subprocess, sys
    pkgs = [
        "qiskit",
        "qiskit-aer",
        "qiskit-algorithms",
        "qiskit-optimization",
        "docplex",
        "cplex",
        "matplotlib",
        "numpy",
    ]
    subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + pkgs, check=True)
    INSTALLED = 1

# ── Imports ──────────────────────────────────────────────────
import time
import random                          # ← ADDED
import numpy as np
import matplotlib                      # ← ADDED
matplotlib.use('Agg')                  # ← non-interactive backend, no window
import matplotlib.pyplot as plt
import json
from pathlib import Path
from datetime import datetime
import argparse
import sys
from docplex.mp.model import Model

# Qiskit 1.x – new packages
from qiskit_algorithms import QAOA, NumPyMinimumEigensolver
from qiskit_algorithms.utils import algorithm_globals
from qiskit_algorithms.optimizers import COBYLA as QiskitCOBYLA

from qiskit.primitives import StatevectorSampler as Sampler

from qiskit_optimization.algorithms import (
    CobylaOptimizer,
    MinimumEigenOptimizer,
)
from qiskit_optimization.algorithms.admm_optimizer import (
    ADMMParameters,
    ADMMOptimizer,
)
from qiskit_optimization.translators import from_docplex_mp

algorithm_globals.massive = True

# ── Dynamic DOcplex model with CLI parsing ───────────────────
MAX_N = 7
default_n_servers = 5
default_n_vms = 5
default_require_all_on = True
default_min_cpu_per_vm = 1

def parse_list_of_numbers(s, cast=float):
    if s is None or s == "":
        return None
    try:
        parts = [p.strip() for p in s.split(",") if p.strip() != ""]
        return [cast(p) for p in parts]
    except Exception:
        print("Error: invalid list format:\n", s)
        sys.exit(1)

parser = argparse.ArgumentParser(description="Run Qiskit optimization with dynamic parameters")
parser.add_argument("--n_servers", type=int, default=default_n_servers)
parser.add_argument("--n_vms", type=int, default=default_n_vms)
parser.add_argument("--require_all_on", type=int, choices=[0, 1], default=1,
                    help="1 to force all servers ON, 0 to allow them off")
parser.add_argument("--min_cpu_per_vm", type=float, default=default_min_cpu_per_vm)
parser.add_argument("--pi_list", type=str, default="",
                    help="Comma-separated list for pi (fixed server costs), e.g.: 1,1,1")
parser.add_argument("--pd_list", type=str, default="",
                    help="Comma-separated list for pd (dynamic server costs)")
parser.add_argument("--capacities", type=str, default="",
                    help="Comma-separated list for server capacities (int)")
parser.add_argument("--vm_allocation_limits", type=str, default="",
                    help="Comma-separated list for VM allocation limits (int)")
parser.add_argument("--fast", action="store_true", help="Fast mode: lower precision, significantly faster execution")

args, unknown = parser.parse_known_args()

fast_mode = args.fast

n_servers      = args.n_servers
n_vms          = args.n_vms
require_all_on = bool(args.require_all_on)
min_cpu_per_vm = args.min_cpu_per_vm

pi_list    = parse_list_of_numbers(args.pi_list, cast=float) or [1.0 for _ in range(n_servers)]
pd_list    = parse_list_of_numbers(args.pd_list, cast=float) or [1.0 for _ in range(n_servers)]
capacities = parse_list_of_numbers(args.capacities, cast=int) or [11 if i < 3 else 10 for i in range(n_servers)]

# ── Base optimizers ──────────────────────────────────────────
cobyla = CobylaOptimizer()

if fast_mode:
    # ── FAST MODE: shallow circuit, few inner iterations ────
    qaoa_algo = QAOA(
        sampler=Sampler(),
        optimizer=QiskitCOBYLA(maxiter=50),   # was 300
        reps=2,                                # was 3
    )
else:
    # ── FULL MODE: deeper circuit, higher precision ─────────
    qaoa_algo = QAOA(
        sampler=Sampler(),
        optimizer=QiskitCOBYLA(maxiter=300),
        reps=3,
    )
qaoa  = MinimumEigenOptimizer(qaoa_algo)
exact = MinimumEigenOptimizer(NumPyMinimumEigensolver())



# ── RANDOM GENERATION of default_vm_alloc ────────────────────
# Unique values between 8 and 12; if n_vms > 5 extra values may repeat
_pool = list(range(8, 13))             # [8, 9, 10, 11, 12] → 5 unique values available
if n_vms <= len(_pool):
    default_vm_alloc = random.sample(_pool, n_vms)
else:
    default_vm_alloc = random.sample(_pool, len(_pool))
    default_vm_alloc += [random.randint(8, 12) for _ in range(n_vms - len(_pool))]

vm_allocation_limits = parse_list_of_numbers(args.vm_allocation_limits, cast=int) or default_vm_alloc[:n_vms]

print(f"default_vm_alloc      : {default_vm_alloc}")
print(f"vm_allocation_limits  : {vm_allocation_limits}")

# ── Quick validations ────────────────────────────────────────
if n_servers > MAX_N or n_vms > MAX_N:
    print(f"Error: n_servers and n_vms must be <= {MAX_N}")
    sys.exit(1)

if len(pi_list) != n_servers:
    print(f"Error: pi_list must have length n_servers ({n_servers})")
    sys.exit(1)
if len(pd_list) != n_servers:
    print(f"Error: pd_list must have length n_servers ({n_servers})")
    sys.exit(1)
if len(capacities) != n_servers:
    print(f"Error: capacities must have length n_servers ({n_servers})")
    sys.exit(1)
if len(vm_allocation_limits) < n_vms:
    print(f"Error: vm_allocation_limits must have at least n_vms ({n_vms}) elements")
    sys.exit(1)

# ── Build DOcplex model ──────────────────────────────────────
mdl = Model("qiskit_server")

# Binary variables: si = 1 if server i is ON, 0 otherwise
s_vars = [mdl.binary_var(name=f"si{i}") for i in range(n_servers)]

# Continuous variables: vj{j}i{i} = CPU allocated from VM j on server i
v_vars = {}
for j in range(n_vms):
    for i in range(n_servers):
        v_vars[(j, i)] = mdl.continuous_var(lb=0.0, name=f"vj{j}i{i}")

# Continuous variables: uj{j} = overall utilization of VM j
u_vars = [mdl.continuous_var(lb=0.0, name=f"uj{j}") for j in range(n_vms)]

# Objective: minimize fixed server costs + dynamic VM utilization costs
obj = mdl.sum(
    pi_list[i] * s_vars[i] +
    pd_list[i] * mdl.sum(u_vars[j] * v_vars[(j, i)] for j in range(n_vms))
    for i in range(n_servers)
)
mdl.minimize(obj)

# Constraint: total CPU allocated on each server >= capacity - 1
for i in range(n_servers):
    mdl.add_constraint(
        mdl.sum(v_vars[(j, i)] for j in range(n_vms)) >= capacities[i] - 1,
        f"cons_server_load_{i}"
    )

# Constraint: first 3 VMs must fully cover capacity of the first 3 servers
for i in range(min(3, n_servers)):
    m = min(3, n_vms)
    if m > 0:
        mdl.add_constraint(
            mdl.sum(v_vars[(j, i)] for j in range(m)) >= capacities[i],
            f"cons_first3vm_server_{i}"
        )

# Constraint: force all servers ON if required
if require_all_on:
    for i in range(n_servers):
        mdl.add_constraint(s_vars[i] == 1, f"cons_server_on_{i}")

# Constraint: total CPU used by each VM across all servers <= allocation limit
for j in range(n_vms):
    limit = vm_allocation_limits[j] if j < len(vm_allocation_limits) else vm_allocation_limits[-1]
    mdl.add_constraint(
        mdl.sum(v_vars[(j, i)] for i in range(n_servers)) <= limit,
        f"cons_vm_alloc_{j}"
    )

# Constraint: each VM must have at least min_cpu_per_vm utilization
for j in range(n_vms):
    mdl.add_constraint(u_vars[j] >= min_cpu_per_vm, f"cons_min_cpu_vm_{j}")

# Translate DOcplex model to Qiskit QuadraticProgram
qp = from_docplex_mp(mdl)
print("=== Quadratic Program (LP) ===")
print(qp.export_as_lp_string())

script_name = "q"

# ── CLASSICAL SOLUTION ───────────────────────────────────────
print("\n" + "=" * 60)
print("  CLASSICAL SOLUTION  (Exact QUBO + COBYLA)")
print("=" * 60)

if fast_mode:
    admm_params_classic = ADMMParameters(
        rho_initial=100, beta=1000, factor_c=900,
        maxiter=20, three_block=True, tol=1e-3,
    )
else:
    admm_params_classic = ADMMParameters(
        rho_initial=100, beta=1000, factor_c=900,
        maxiter=100, three_block=True, tol=1e-6,
    )

admm_classic = ADMMOptimizer(
    params=admm_params_classic,
    qubo_optimizer=exact,
    continuous_optimizer=cobyla,
)

print("ADMM compatibility with the problem:")
print(admm_classic.get_compatibility_msg(qp))

t1 = time.perf_counter()
result_classic = admm_classic.solve(qp)
t2 = time.perf_counter()
duration_classic = t2 - t1
print(f"Time: {round(duration_classic, 2)} seconds")
result_classic.prettyprint()

# Classical plot – saved to file only, not displayed
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(result_classic.state.residuals, color="steelblue")
axes[0].set_xlabel("Iterations")
axes[0].set_ylabel("Residuals")
axes[0].set_title("Classical – ADMM Residuals")
axes[0].grid(True)

axes[1].bar(range(len(result_classic.x)), result_classic.x, color="steelblue")
axes[1].set_xlabel("Variable index")
axes[1].set_ylabel("Value")
axes[1].set_title("Classical – Solution")
axes[1].grid(True, axis="y")

plt.tight_layout()
plt.close()                            # ← closes without displaying

# ── QUANTUM SOLUTION ─────────────────────────────────────────
print("\n" + "=" * 60)
print("  QUANTUM SOLUTION  (QAOA + COBYLA)")
print("=" * 60)

if fast_mode:
    admm_params_quantum = ADMMParameters(
        rho_initial=100, beta=10, factor_c=100,
        maxiter=50, three_block=True, tol=1e-2,
    )
else:
    admm_params_quantum = ADMMParameters(
        rho_initial=100, beta=100, factor_c=100,
        maxiter=300, three_block=True, tol=1e-4,
    )

admm_quantum = ADMMOptimizer(
    params=admm_params_quantum,
    qubo_optimizer=qaoa,
    continuous_optimizer=cobyla,
)

t1 = time.perf_counter()
result_quantum = admm_quantum.solve(qp)
t2 = time.perf_counter()
duration_quantum = t2 - t1
print(f"Time: {round(duration_quantum, 2)} seconds")
result_quantum.prettyprint()

# Quantum plot – saved to file only, not displayed
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(result_quantum.state.residuals, color="darkorange")
axes[0].set_xlabel("Iterations")
axes[0].set_ylabel("Residuals")
axes[0].set_title("Quantum – ADMM Residuals")
axes[0].grid(True)

axes[1].bar(range(len(result_quantum.x)), result_quantum.x, color="darkorange")
axes[1].set_xlabel("Variable index")
axes[1].set_ylabel("Value")
axes[1].set_title("Quantum – Solution")
axes[1].grid(True, axis="y")

plt.tight_layout()
plt.close()                            # ← closes without displaying

# ── Final comparison ─────────────────────────────────────────
print("\n" + "=" * 60)
print("  COMPARISON")
print("=" * 60)
print(f"Classical  -> Objective: {result_classic.fval:.4f}  | Status: {result_classic.status.name}")
print(f"Quantum    -> Objective: {result_quantum.fval:.4f}  | Status: {result_quantum.status.name}")

# ── Serialize results to JSON ────────────────────────────────
def as_list(x):
    try:
        return list(x)
    except Exception:
        return x

results_data = {
    "script": script_name,
    "meta": {
        "n_servers": n_servers,
        "n_vms": n_vms,
        "require_all_on": require_all_on,
    },
    "input": {
        "pi": pi_list,
        "pd": pd_list,
        "capacities": capacities,
        "vm_allocation_limits": vm_allocation_limits,
        "min_cpu_per_vm": min_cpu_per_vm,
    },
    "qp_lp": qp.export_as_lp_string(),
    "classic": {
        "objective": float(result_classic.fval),
        "status": result_classic.status.name,
        "x": as_list(np.array(result_classic.x).tolist()),
        "residuals": as_list(getattr(result_classic.state, "residuals", [])),
        "duration_seconds": duration_classic,
    },
    "quantum": {
        "objective": float(result_quantum.fval),
        "status": result_quantum.status.name,
        "x": as_list(np.array(result_quantum.x).tolist()),
        "residuals": as_list(getattr(result_quantum.state, "residuals", [])),
        "duration_seconds": duration_quantum,
    },
}

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
combined_img = f"{script_name}_{ts}_results.png"

# Combined plot – saved to file only, not displayed
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].plot(result_classic.state.residuals, color="steelblue")
axes[0, 0].set_xlabel("Iterations")
axes[0, 0].set_ylabel("Residuals")
axes[0, 0].set_title("Classical – ADMM Residuals")
axes[0, 0].grid(True)

axes[0, 1].bar(range(len(result_classic.x)), result_classic.x, color="steelblue")
axes[0, 1].set_xlabel("Variable index")
axes[0, 1].set_ylabel("Value")
axes[0, 1].set_title("Classical – Solution")
axes[0, 1].grid(True, axis="y")

axes[1, 0].plot(result_quantum.state.residuals, color="darkorange")
axes[1, 0].set_xlabel("Iterations")
axes[1, 0].set_ylabel("Residuals")
axes[1, 0].set_title("Quantum – ADMM Residuals")
axes[1, 0].grid(True)

axes[1, 1].bar(range(len(result_quantum.x)), result_quantum.x, color="darkorange")
axes[1, 1].set_xlabel("Variable index")
axes[1, 1].set_ylabel("Value")
axes[1, 1].set_title("Quantum – Solution")
axes[1, 1].grid(True, axis="y")

plt.tight_layout()
plt.savefig(combined_img, dpi=150)    # ← saves the combined chart
plt.close()                            # ← closes without displaying

results_data["combined_image"] = combined_img

results_file = f"{script_name}_{ts}_results.json"
with open(results_file, "w", encoding="utf-8") as fh:
    json.dump(results_data, fh, indent=2, ensure_ascii=False)

print(f"Results saved to: {results_file}")
print(f"Combined chart saved to: {combined_img}")