# ============================================================
# Qiskit – N Servers / M VMs – Optimization Problem
# Updated for Qiskit 1.x + qiskit-algorithms + qiskit-aer
# ── FIXES applied (vs previous version) ────────────────────
#  1. vm_alloc_limits: now computed from capacities to guarantee
#     feasibility instead of random 8-12 (which was always too small)
#  2. Quantum ADMM: beta/factor_c raised to match classic solver
#     so penalty forces si=1 when require_all_on=True
#  3. pre_check_feasibility(): hard abort before wasting compute
#     time on a structurally infeasible problem
# ============================================================

try:
    INSTALLED
except NameError:
    INSTALLED = None

if INSTALLED != 1:
    import subprocess, sys
    pkgs = [
        "qiskit", "qiskit-aer", "qiskit-algorithms",
        "qiskit-optimization", "docplex", "cplex",
        "matplotlib", "numpy",
    ]
    subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + pkgs, check=True)
    INSTALLED = 1

import time, random, math, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import argparse
from docplex.mp.model import Model

from qiskit_algorithms import QAOA, NumPyMinimumEigensolver
from qiskit_algorithms.utils import algorithm_globals
from qiskit_algorithms.optimizers import COBYLA as QiskitCOBYLA
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_optimization.algorithms import CobylaOptimizer, MinimumEigenOptimizer
from qiskit_optimization.algorithms.admm_optimizer import ADMMParameters, ADMMOptimizer
from qiskit_optimization.translators import from_docplex_mp

algorithm_globals.massive = True

MAX_N = 7
default_n_servers      = 5
default_n_vms          = 5
default_require_all_on = True
default_min_cpu_per_vm = 1

def parse_list_of_numbers(s, cast=float):
    if s is None or s == "":
        return None
    try:
        return [cast(p.strip()) for p in s.split(",") if p.strip()]
    except Exception:
        print("Error: invalid list format:", s)
        sys.exit(1)

parser = argparse.ArgumentParser()
parser.add_argument("--n_servers",           type=int,   default=default_n_servers)
parser.add_argument("--n_vms",               type=int,   default=default_n_vms)
parser.add_argument("--require_all_on",      type=int,   choices=[0, 1], default=1)
parser.add_argument("--min_cpu_per_vm",      type=float, default=default_min_cpu_per_vm)
parser.add_argument("--pi_list",             type=str,   default="")
parser.add_argument("--pd_list",             type=str,   default="")
parser.add_argument("--capacities",          type=str,   default="")
parser.add_argument("--vm_allocation_limits",type=str,   default="")
parser.add_argument("--fast", action="store_true")
args, _ = parser.parse_known_args()

fast_mode      = args.fast
n_servers      = args.n_servers
n_vms          = args.n_vms
require_all_on = bool(args.require_all_on)
min_cpu_per_vm = args.min_cpu_per_vm

pi_list    = parse_list_of_numbers(args.pi_list,    float) or [1.0] * n_servers
pd_list    = parse_list_of_numbers(args.pd_list,    float) or [1.0] * n_servers
capacities = parse_list_of_numbers(args.capacities, int)   or [11 if i < 3 else 10 for i in range(n_servers)]

# ── FIX 1: compute safe vm_alloc_limits from capacities ─────
# Root cause of all INFEASIBLE results in the original code:
# the first min(3,n_vms) VMs must collectively cover EACH of the
# first min(3,n_servers) servers with >= cap[i] CPUs, while each
# VM's TOTAL budget across ALL servers is <= vm_alloc_limits[j].
#
# Necessary condition:
#   sum(lim[:n_first_vms]) >= sum(cap[:n_first_servers])   [first3vm]
#   sum(lim)               >= sum(cap) - n_servers          [server_load]
#
# We derive the minimum per-VM limit that satisfies both, then
# add a 25% safety margin and a small random perturbation.

def compute_safe_vm_alloc(capacities, n_vms, n_servers, margin=1.25):
    """Return a list of n_vms safe vm_alloc_limit values."""
    n_first_vms = min(3, n_vms)
    n_first_srv = min(3, n_servers)

    # first3vm constraint: first n_first_vms VMs cover first n_first_srv servers
    min_budget_first3 = sum(capacities[:n_first_srv])
    min_per_vm_first3 = math.ceil(min_budget_first3 / n_first_vms)

    # server_load constraint: all VMs cover all servers (cap - 1 each)
    min_budget_total  = sum(capacities) - n_servers
    min_per_vm_total  = math.ceil(max(min_budget_total, 0) / n_vms)

    base = math.ceil(max(min_per_vm_first3, min_per_vm_total) * margin)
    return [base + random.randint(0, 3) for _ in range(n_vms)]  # ← FIXED

if args.vm_allocation_limits:
    vm_allocation_limits = parse_list_of_numbers(args.vm_allocation_limits, int)
else:
    vm_allocation_limits = compute_safe_vm_alloc(capacities, n_vms, n_servers)

print(f"vm_allocation_limits  : {vm_allocation_limits}")
print(f"sum(vm_alloc_limits)  : {sum(vm_allocation_limits)}  "
      f"(need >= {sum(capacities) - n_servers} for server_load)")
print(f"sum(lim[:3])          : {sum(vm_allocation_limits[:min(3,n_vms)])}  "
      f"(need >= {sum(capacities[:min(3,n_servers)])} for first3vm)")

# ── FIX 2: pre-run feasibility check ────────────────────────
def pre_check_feasibility(capacities, vm_allocation_limits, n_vms, n_servers):
    """
    Returns (is_feasible: bool, reason: str).
    Checks the two necessary conditions analytically before building the circuit.
    """
    n_first_vms = min(3, n_vms)
    n_first_srv = min(3, n_servers)

    need_first3 = sum(capacities[:n_first_srv])
    have_first3 = sum(vm_allocation_limits[:n_first_vms])
    if have_first3 < need_first3:
        return False, (
            f"first3vm infeasible: sum(lim[:{n_first_vms}])={have_first3} "
            f"< sum(cap[:{n_first_srv}])={need_first3}. "
            f"Increase vm_allocation_limits so their sum >= {need_first3}."
        )

    need_load = sum(capacities) - n_servers
    have_load = sum(vm_allocation_limits)
    if have_load < need_load:
        return False, (
            f"server_load infeasible: sum(lim)={have_load} "
            f"< sum(cap)-n_servers={need_load}. "
            f"Increase vm_allocation_limits so their sum >= {need_load}."
        )

    return True, "OK"

feasible, reason = pre_check_feasibility(
    capacities, vm_allocation_limits, n_vms, n_servers
)
if not feasible:
    print("\n" + "=" * 60)
    print("  PROBLEM IS STRUCTURALLY INFEASIBLE — ABORTING")
    print("  " + reason)
    print("=" * 60)
    sys.exit(1)
else:
    print(" Feasibility pre-check passed — proceeding.\n")

# ── Validations ──────────────────────────────────────────────
if n_servers > MAX_N or n_vms > MAX_N:
    print(f"Error: n_servers and n_vms must be <= {MAX_N}")
    sys.exit(1)
for name, lst, expected in [
    ("pi_list",    pi_list,    n_servers),
    ("pd_list",    pd_list,    n_servers),
    ("capacities", capacities, n_servers),
]:
    if len(lst) != expected:
        print(f"Error: {name} must have length {expected}, got {len(lst)}")
        sys.exit(1)
if len(vm_allocation_limits) < n_vms:
    print(f"Error: vm_allocation_limits needs {n_vms} elements, got {len(vm_allocation_limits)}")
    sys.exit(1)

# ── Base optimizers ──────────────────────────────────────────
cobyla = CobylaOptimizer()

if fast_mode:
    qaoa_algo = QAOA(
        sampler=Sampler(),
        optimizer=QiskitCOBYLA(maxiter=50),
        reps=2,
    )
else:
    qaoa_algo = QAOA(
        sampler=Sampler(),
        optimizer=QiskitCOBYLA(maxiter=300),
        reps=3,
    )
qaoa  = MinimumEigenOptimizer(qaoa_algo)
exact = MinimumEigenOptimizer(NumPyMinimumEigensolver())

# ── Build DOcplex model ──────────────────────────────────────
mdl    = Model("qiskit_server")
s_vars = [mdl.binary_var(name=f"si{i}") for i in range(n_servers)]

v_vars = {}
for j in range(n_vms):
    for i in range(n_servers):
        v_vars[(j, i)] = mdl.continuous_var(lb=0.0, name=f"vj{j}i{i}")

u_vars = [mdl.continuous_var(lb=0.0, name=f"uj{j}") for j in range(n_vms)]

obj = mdl.sum(
    pi_list[i] * s_vars[i] +
    pd_list[i] * mdl.sum(u_vars[j] * v_vars[(j, i)] for j in range(n_vms))
    for i in range(n_servers)
)
mdl.minimize(obj)

for i in range(n_servers):
    mdl.add_constraint(
        mdl.sum(v_vars[(j, i)] for j in range(n_vms)) >= capacities[i] - 1,
        f"cons_server_load_{i}"
    )

for i in range(min(3, n_servers)):
    m = min(3, n_vms)
    if m > 0:
        mdl.add_constraint(
            mdl.sum(v_vars[(j, i)] for j in range(m)) >= capacities[i],
            f"cons_first3vm_server_{i}"
        )

if require_all_on:
    for i in range(n_servers):
        mdl.add_constraint(s_vars[i] == 1, f"cons_server_on_{i}")

for j in range(n_vms):
    limit = vm_allocation_limits[j] if j < len(vm_allocation_limits) else vm_allocation_limits[-1]
    mdl.add_constraint(
        mdl.sum(v_vars[(j, i)] for i in range(n_servers)) <= limit,
        f"cons_vm_alloc_{j}"
    )

for j in range(n_vms):
    mdl.add_constraint(u_vars[j] >= min_cpu_per_vm, f"cons_min_cpu_vm_{j}")

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

print("ADMM compatibility:", admm_classic.get_compatibility_msg(qp))

t1 = time.perf_counter()
result_classic = admm_classic.solve(qp)
t2 = time.perf_counter()
duration_classic = t2 - t1
print(f"Time: {round(duration_classic, 2)}s")
result_classic.prettyprint()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(result_classic.state.residuals, color="steelblue")
axes[0].set(xlabel="Iterations", ylabel="Residuals", title="Classical – ADMM Residuals")
axes[0].grid(True)
axes[1].bar(range(len(result_classic.x)), result_classic.x, color="steelblue")
axes[1].set(xlabel="Variable index", ylabel="Value", title="Classical – Solution")
axes[1].grid(True, axis="y")
plt.tight_layout()
plt.close()

# ── QUANTUM SOLUTION ─────────────────────────────────────────
print("\n" + "=" * 60)
print("  QUANTUM SOLUTION  (QAOA + COBYLA)")
print("=" * 60)

# ── FIX 3: raise beta/factor_c to match classic ─────────────
# Root cause of quantum always returning si=0 (all servers OFF):
# with beta=10/factor_c=100, the penalty for violating si=1 was
# smaller than the objective gain from setting si=0.
# Raising to beta=1000/factor_c=900 (same as classic) forces the
# QUBO to penalise constraint violations strongly enough.
if fast_mode:
    admm_params_quantum = ADMMParameters(
        rho_initial=100,
        beta=1000,       # ← FIXED: was 10  (too weak → si=0 always)
        factor_c=900,    # ← FIXED: was 100 (too weak → constraints ignored)
        maxiter=50, three_block=True, tol=1e-2,
    )
else:
    admm_params_quantum = ADMMParameters(
        rho_initial=100,
        beta=1000,       # ← FIXED: was 100
        factor_c=900,    # ← FIXED: was 100
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
print(f"Time: {round(duration_quantum, 2)}s")
result_quantum.prettyprint()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(result_quantum.state.residuals, color="darkorange")
axes[0].set(xlabel="Iterations", ylabel="Residuals", title="Quantum – ADMM Residuals")
axes[0].grid(True)
axes[1].bar(range(len(result_quantum.x)), result_quantum.x, color="darkorange")
axes[1].set(xlabel="Variable index", ylabel="Value", title="Quantum – Solution")
axes[1].grid(True, axis="y")
plt.tight_layout()
plt.close()

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

ts           = datetime.now().strftime("%Y%m%d_%H%M%S")
combined_img = f"{script_name}_{ts}_results.png"

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes[0, 0].plot(result_classic.state.residuals, color="steelblue")
axes[0, 0].set(xlabel="Iterations", ylabel="Residuals", title="Classical – ADMM Residuals")
axes[0, 0].grid(True)
axes[0, 1].bar(range(len(result_classic.x)), result_classic.x, color="steelblue")
axes[0, 1].set(xlabel="Variable index", ylabel="Value", title="Classical – Solution")
axes[0, 1].grid(True, axis="y")
axes[1, 0].plot(result_quantum.state.residuals, color="darkorange")
axes[1, 0].set(xlabel="Iterations", ylabel="Residuals", title="Quantum – ADMM Residuals")
axes[1, 0].grid(True)
axes[1, 1].bar(range(len(result_quantum.x)), result_quantum.x, color="darkorange")
axes[1, 1].set(xlabel="Variable index", ylabel="Value", title="Quantum – Solution")
axes[1, 1].grid(True, axis="y")
plt.tight_layout()
plt.savefig(combined_img, dpi=150)
plt.close()

results_data["combined_image"] = combined_img

results_file = f"{script_name}_{ts}_results.json"
with open(results_file, "w", encoding="utf-8") as fh:
    json.dump(results_data, fh, indent=2, ensure_ascii=False)

print(f"\nResults saved to : {results_file}")
print(f"Chart saved to   : {combined_img}")