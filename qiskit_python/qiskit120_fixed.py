# ============================================================
# Qiskit 120 – 5 Server / 5 VM – Ottimizzazione Energetica
# Aggiornato per Qiskit 1.x + qiskit-algorithms + qiskit-aer
# ============================================================

# ── Installazione pacchetti (eseguita una sola volta) ────────
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

# ── Import ───────────────────────────────────────────────────
import time
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from datetime import datetime
import argparse
import sys
from docplex.mp.model import Model

# Qiskit 1.x – nuovi package
from qiskit_algorithms import QAOA, NumPyMinimumEigensolver
from qiskit_algorithms.utils import algorithm_globals
from qiskit_algorithms.optimizers import COBYLA as QiskitCOBYLA

# Usa il Sampler statevector (esatto, no rumore, ideale per test)
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

# ── Ottimizzatori base ───────────────────────────────────────
cobyla = CobylaOptimizer()

qaoa_algo = QAOA(
    sampler=Sampler(),
    optimizer=QiskitCOBYLA(maxiter=300),
    reps=3,
)
qaoa  = MinimumEigenOptimizer(qaoa_algo)
exact = MinimumEigenOptimizer(NumPyMinimumEigensolver())

# ── Modello Docplex dinamico con parsing CLI ────────────────
# Parametri di default
MAX_N = 7
default_n_servers = 5
default_n_vms = 5
default_require_all_on = True
default_min_cpu_per_vm = 1

# helper per parsing liste
def parse_list_of_numbers(s, cast=float):
    if s is None or s == "":
        return None
    try:
        parts = [p.strip() for p in s.split(",") if p.strip() != ""]
        return [cast(p) for p in parts]
    except Exception:
        print("Errore: formato lista non valido:\n", s)
        sys.exit(1)

parser = argparse.ArgumentParser(description="Esegui Qiskit120 con parametri dinamici")
parser.add_argument("--n_servers", type=int, default=default_n_servers)
parser.add_argument("--n_vms", type=int, default=default_n_vms)
parser.add_argument("--require_all_on", type=int, choices=[0,1], default=1,
                    help="1 per forzare tutti i server ON, 0 per no")
parser.add_argument("--min_cpu_per_vm", type=float, default=default_min_cpu_per_vm)
parser.add_argument("--pi_list", type=str, default="",
                    help="Lista separata da virgole per pi, es: 1,1,1")
parser.add_argument("--pd_list", type=str, default="",
                    help="Lista separata da virgole per pd")
parser.add_argument("--capacities", type=str, default="",
                    help="Lista separata da virgole per capacities (int)")
parser.add_argument("--vm_allocation_limits", type=str, default="",
                    help="Lista separata da virgole per limiti di VM (int)")

args, unknown = parser.parse_known_args()

n_servers = args.n_servers
n_vms = args.n_vms
require_all_on = bool(args.require_all_on)
min_cpu_per_vm = args.min_cpu_per_vm

# costruisci liste usando input CLI o default
pi_list = parse_list_of_numbers(args.pi_list, cast=float) or [1.0 for _ in range(n_servers)]
pd_list = parse_list_of_numbers(args.pd_list, cast=float) or [1.0 for _ in range(n_servers)]
capacities = parse_list_of_numbers(args.capacities, cast=int) or [11 if i < 3 else 10 for i in range(n_servers)]
default_vm_alloc = [12, 11, 10, 10, 10, 10, 10]
vm_allocation_limits = parse_list_of_numbers(args.vm_allocation_limits, cast=int) or default_vm_alloc[:n_vms]

# Validazioni rapide
if n_servers > MAX_N or n_vms > MAX_N:
    print(f"Errore: n_servers e n_vms devono essere <= {MAX_N}")
    sys.exit(1)

if len(pi_list) != n_servers:
    print(f"Errore: la lista pi_list deve avere lunghezza n_servers ({n_servers})")
    sys.exit(1)
if len(pd_list) != n_servers:
    print(f"Errore: la lista pd_list deve avere lunghezza n_servers ({n_servers})")
    sys.exit(1)
if len(capacities) != n_servers:
    print(f"Errore: la lista capacities deve avere lunghezza n_servers ({n_servers})")
    sys.exit(1)
if len(vm_allocation_limits) < n_vms:
    print(f"Errore: la lista vm_allocation_limits deve avere almeno n_vms ({n_vms}) elementi")
    sys.exit(1)

mdl = Model("ex120_server")

# Variabili: server on/off
s_vars = [mdl.binary_var(name=f"si{i}") for i in range(n_servers)]

# Variabili: frazione di VM j su server i (non negative)
v_vars = {}
for j in range(n_vms):
    for i in range(n_servers):
        v_vars[(j, i)] = mdl.continuous_var(lb=0.0, name=f"vj{j}i{i}")

# Variabili: utilizzo CPU di ciascuna VM
u_vars = [mdl.continuous_var(lb=0.0, name=f"uj{j}") for j in range(n_vms)]

# ── Funzione obiettivo costruita dinamicamente
obj = mdl.sum(pi_list[i] * s_vars[i] + pd_list[i] * mdl.sum(u_vars[j] * v_vars[(j, i)] for j in range(n_vms)) for i in range(n_servers))
mdl.minimize(obj)

# ── Vincoli dinamici
# Carico totale per server >= capacity-1
for i in range(n_servers):
    mdl.add_constraint(mdl.sum(v_vars[(j, i)] for j in range(n_vms)) >= capacities[i] - 1, f"cons_server_load_{i}")

# Prime 3 VM coprono capacità piena per i < 3 (se presenti)
for i in range(min(3, n_servers)):
    m = min(3, n_vms)
    if m > 0:
        mdl.add_constraint(mdl.sum(v_vars[(j, i)] for j in range(m)) >= capacities[i], f"cons_first3vm_server_{i}")

# Tutti i server possono essere forzati accesi se richiesto
if require_all_on:
    for i in range(n_servers):
        mdl.add_constraint(s_vars[i] == 1, f"cons_server_on_{i}")

# Limite totale di allocazione per ogni VM (su tutti i server)
for j in range(n_vms):
    limit = vm_allocation_limits[j] if j < len(vm_allocation_limits) else vm_allocation_limits[-1]
    mdl.add_constraint(mdl.sum(v_vars[(j, i)] for i in range(n_servers)) <= limit, f"cons_vm_alloc_{j}")

# Utilizzo CPU minimo per ogni VM
for j in range(n_vms):
    mdl.add_constraint(u_vars[j] >= min_cpu_per_vm, f"cons_min_cpu_vm_{j}")

# ── Converti in QuadraticProgram ─────────────────────────────
qp = from_docplex_mp(mdl)
print("=== Quadratic Program (LP) ===")
print(qp.export_as_lp_string())

# Nome base file (usa il nome del file script senza estensione)
script_name = "q"

# ── SOLUZIONE CLASSICA: ADMM + Exact QUBO + COBYLA ───────────
print("\n" + "="*60)
print("  SOLUZIONE CLASSICA  (Exact QUBO + COBYLA)")
print("="*60)

admm_params_classic = ADMMParameters(
    rho_initial=100,
    beta=1000,
    factor_c=900,
    maxiter=100,
    three_block=True,
    tol=1e-6,
)

admm_classic = ADMMOptimizer(
    params=admm_params_classic,
    qubo_optimizer=exact,
    continuous_optimizer=cobyla,
)

print("Compatibilità ADMM con il problema:")
print(admm_classic.get_compatibility_msg(qp))

t1 = time.perf_counter()
result_classic = admm_classic.solve(qp)
t2 = time.perf_counter()
duration_classic = t2 - t1
print(f"Tempo: {round(duration_classic, 2)} secondi")
result_classic.prettyprint()

# Plot classico
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(result_classic.state.residuals, color="steelblue")
axes[0].set_xlabel("Iterazioni")
axes[0].set_ylabel("Residuals")
axes[0].set_title("Classico – Residuals ADMM")
axes[0].grid(True)

axes[1].bar(range(len(result_classic.x)), result_classic.x, color="steelblue")
axes[1].set_xlabel("Indice variabile")
axes[1].set_ylabel("Valore")
axes[1].set_title("Classico – Soluzione")
axes[1].grid(True, axis="y")

plt.tight_layout()
plt.show()

# ── SOLUZIONE QUANTISTICA: ADMM + QAOA + COBYLA ──────────────
print("\n" + "="*60)
print("  SOLUZIONE QUANTISTICA  (QAOA + COBYLA)")
print("="*60)

admm_params_quantum = ADMMParameters(
    rho_initial=100,
    beta=100,
    factor_c=100,
    maxiter=300,
    three_block=True,
    tol=1e-4,
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
print(f"Tempo: {round(duration_quantum, 2)} secondi")
result_quantum.prettyprint()

# Plot quantistico
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(result_quantum.state.residuals, color="darkorange")
axes[0].set_xlabel("Iterazioni")
axes[0].set_ylabel("Residuals")
axes[0].set_title("Quantistico – Residuals ADMM")
axes[0].grid(True)

axes[1].bar(range(len(result_quantum.x)), result_quantum.x, color="darkorange")
axes[1].set_xlabel("Indice variabile")
axes[1].set_ylabel("Valore")
axes[1].set_title("Quantistico – Soluzione")
axes[1].grid(True, axis="y")

plt.tight_layout()
plt.show()

# ── Confronto finale ─────────────────────────────────────────
print("\n" + "="*60)
print("  CONFRONTO")
print("="*60)
print(f"Classico  → Obiettivo: {result_classic.fval:.4f}  | Status: {result_classic.status.name}")
print(f"Quantistico → Obiettivo: {result_quantum.fval:.4f}  | Status: {result_quantum.status.name}")

# Salva JSON con dati di input e risultati
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
# Genera nome file con timestamp nel formato richiesto: q_annomesegiorno_oraminutosecondo
ts = datetime.now().strftime("%Y%m%d_%H%M%S")

# Grafico combinato (classico + quantistico) in un'unica immagine
combined_img = f"{script_name}_{ts}_risultati.png"
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Classico residuals
axes[0, 0].plot(result_classic.state.residuals, color="steelblue")
axes[0, 0].set_xlabel("Iterazioni")
axes[0, 0].set_ylabel("Residuals")
axes[0, 0].set_title("Classico – Residuals ADMM")
axes[0, 0].grid(True)

# Classico soluzione
axes[0, 1].bar(range(len(result_classic.x)), result_classic.x, color="steelblue")
axes[0, 1].set_xlabel("Indice variabile")
axes[0, 1].set_ylabel("Valore")
axes[0, 1].set_title("Classico – Soluzione")
axes[0, 1].grid(True, axis="y")

# Quantistico residuals
axes[1, 0].plot(result_quantum.state.residuals, color="darkorange")
axes[1, 0].set_xlabel("Iterazioni")
axes[1, 0].set_ylabel("Residuals")
axes[1, 0].set_title("Quantistico – Residuals ADMM")
axes[1, 0].grid(True)

# Quantistico soluzione
axes[1, 1].bar(range(len(result_quantum.x)), result_quantum.x, color="darkorange")
axes[1, 1].set_xlabel("Indice variabile")
axes[1, 1].set_ylabel("Valore")
axes[1, 1].set_title("Quantistico – Soluzione")
axes[1, 1].grid(True, axis="y")

plt.tight_layout()
plt.savefig(combined_img, dpi=150)
plt.show()

# Aggiorna JSON con riferimento all'immagine combinata e nome file con timestamp
results_data["combined_image"] = combined_img

results_file = f"{script_name}_{ts}_results.json"
with open(results_file, "w", encoding="utf-8") as fh:
    json.dump(results_data, fh, indent=2, ensure_ascii=False)

print(f"Risultati salvati in: {results_file}")
print(f"Immagine combinata salvata in: {combined_img}")
