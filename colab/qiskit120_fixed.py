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

# ── Modello Docplex ──────────────────────────────────────────
# 5 server (i0..i4), 5 VM (j0..j4)
mdl = Model("ex120_server")

# Variabili binarie: server acceso/spento
si0 = mdl.binary_var(name="si0")
si1 = mdl.binary_var(name="si1")
si2 = mdl.binary_var(name="si2")
si3 = mdl.binary_var(name="si3")
si4 = mdl.binary_var(name="si4")

# Variabili continue: frazione di VM j allocata su server i
vj0i0 = mdl.continuous_var(name="vj0i0")
vj1i0 = mdl.continuous_var(name="vj1i0")
vj2i0 = mdl.continuous_var(name="vj2i0")
vj3i0 = mdl.continuous_var(name="vj3i0")
vj4i0 = mdl.continuous_var(name="vj4i0")

vj0i1 = mdl.continuous_var(name="vj0i1")
vj1i1 = mdl.continuous_var(name="vj1i1")
vj2i1 = mdl.continuous_var(name="vj2i1")
vj3i1 = mdl.continuous_var(name="vj3i1")
vj4i1 = mdl.continuous_var(name="vj4i1")

vj0i2 = mdl.continuous_var(name="vj0i2")
vj1i2 = mdl.continuous_var(name="vj1i2")
vj2i2 = mdl.continuous_var(name="vj2i2")
vj3i2 = mdl.continuous_var(name="vj3i2")
vj4i2 = mdl.continuous_var(name="vj4i2")

vj0i3 = mdl.continuous_var(name="vj0i3")
vj1i3 = mdl.continuous_var(name="vj1i3")
vj2i3 = mdl.continuous_var(name="vj2i3")
vj3i3 = mdl.continuous_var(name="vj3i3")
vj4i3 = mdl.continuous_var(name="vj4i3")

vj0i4 = mdl.continuous_var(name="vj0i4")
vj1i4 = mdl.continuous_var(name="vj1i4")
vj2i4 = mdl.continuous_var(name="vj2i4")
vj3i4 = mdl.continuous_var(name="vj3i4")
vj4i4 = mdl.continuous_var(name="vj4i4")

# Variabili continue: utilizzo CPU di ciascuna VM
uj0 = mdl.continuous_var(name="uj0")
uj1 = mdl.continuous_var(name="uj1")
uj2 = mdl.continuous_var(name="uj2")
uj3 = mdl.continuous_var(name="uj3")
uj4 = mdl.continuous_var(name="uj4")

# Parametri energetici
pi0 = pi1 = pi2 = pi3 = pi4 = 1   # potenza idle
pd0 = pd1 = pd2 = pd3 = pd4 = 1   # potenza dinamica

# Capacità CPU dei server
ci0 = ci1 = ci2 = 11
ci3 = ci4 = 10

# ── Funzione obiettivo (minimizzazione energia totale) ───────
mdl.minimize(
    pi0*si0 + pd0*(uj0*vj0i0 + uj1*vj1i0 + uj2*vj2i0 + uj3*vj3i0 + uj4*vj4i0) +
    pi1*si1 + pd1*(uj0*vj0i1 + uj1*vj1i1 + uj2*vj2i1 + uj3*vj3i1 + uj4*vj4i1) +
    pi2*si2 + pd2*(uj0*vj0i2 + uj1*vj1i2 + uj2*vj2i2 + uj3*vj3i2 + uj4*vj4i2) +
    pi3*si3 + pd3*(uj0*vj0i3 + uj1*vj1i3 + uj2*vj2i3 + uj3*vj3i3 + uj4*vj4i3) +
    pi4*si4 + pd4*(uj0*vj0i4 + uj1*vj1i4 + uj2*vj2i4 + uj3*vj3i4 + uj4*vj4i4)
)

# ── Vincoli ──────────────────────────────────────────────────
# Carico totale per server >= capacità-1
mdl.add_constraint(vj0i0+vj1i0+vj2i0+vj3i0+vj4i0 >= ci0-1, "cons2")
mdl.add_constraint(vj0i1+vj1i1+vj2i1+vj3i1+vj4i1 >= ci1-1, "cons3")
mdl.add_constraint(vj0i2+vj1i2+vj2i2+vj3i2+vj4i2 >= ci2-1, "cons4")
mdl.add_constraint(vj0i3+vj1i3+vj2i3+vj3i3+vj4i3 >= ci3-1, "cons5")
mdl.add_constraint(vj0i4+vj1i4+vj2i4+vj3i4+vj4i4 >= ci4-1, "cons5a")

# Non-negatività esplicita
mdl.add_constraint(vj0i0+vj1i0+vj2i0+vj3i0+vj4i0 >= 0, "cons6")
mdl.add_constraint(vj0i1+vj1i1+vj2i1+vj3i1+vj4i1 >= 0, "cons7")
mdl.add_constraint(vj0i2+vj1i2+vj2i2+vj3i2+vj4i2 >= 0, "cons8")
mdl.add_constraint(vj0i3+vj1i3+vj2i3+vj3i3+vj4i3 >= 0, "cons9")
mdl.add_constraint(vj0i4+vj1i4+vj2i4+vj3i4+vj4i4 >= 0, "cons9a")

# Prime 3 VM coprono capacità piena (per server i0, i1, i2)
mdl.add_constraint(vj0i0+vj1i0+vj2i0 >= ci0, "cons10")
mdl.add_constraint(vj0i1+vj1i1+vj2i1 >= ci1, "cons11")
mdl.add_constraint(vj0i2+vj1i2+vj2i2 >= ci2, "cons12")

# Tutti i server devono essere accesi
mdl.add_constraint(si0 == 1, "cons14")
mdl.add_constraint(si1 == 1, "cons15")
mdl.add_constraint(si2 == 1, "cons16")
mdl.add_constraint(si3 == 1, "cons17")
mdl.add_constraint(si4 == 1, "cons17a")

# Limite totale di allocazione per ogni VM (su tutti i server)
mdl.add_constraint(vj0i0+vj0i1+vj0i2+vj0i3+vj0i4 <= 12, "cons18")
mdl.add_constraint(vj1i0+vj1i1+vj1i2+vj1i3+vj1i4 <= 11, "cons19")
mdl.add_constraint(vj2i0+vj2i1+vj2i2+vj2i3+vj2i4 <= 10, "cons20")
mdl.add_constraint(vj3i0+vj3i1+vj3i2+vj3i3+vj3i4 <= 10, "cons21")
mdl.add_constraint(vj4i0+vj4i1+vj4i2+vj4i3+vj4i4 <= 10, "cons21a")

# Utilizzo CPU minimo per ogni VM
mdl.add_constraint(uj0 >= 1, "cons22")
mdl.add_constraint(uj1 >= 1, "cons23")
mdl.add_constraint(uj2 >= 1, "cons24")
mdl.add_constraint(uj3 >= 1, "cons25")
mdl.add_constraint(uj4 >= 1, "cons25a")

# ── Converti in QuadraticProgram ─────────────────────────────
qp = from_docplex_mp(mdl)
print("=== Quadratic Program (LP) ===")
print(qp.export_as_lp_string())

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

print(f"Tempo: {round(t2 - t1, 2)} secondi")
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
plt.savefig("risultati_classici.png", dpi=150)
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

print(f"Tempo: {round(t2 - t1, 2)} secondi")
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
plt.savefig("risultati_quantistici.png", dpi=150)
plt.show()

# ── Confronto finale ─────────────────────────────────────────
print("\n" + "="*60)
print("  CONFRONTO")
print("="*60)
print(f"Classico  → Obiettivo: {result_classic.fval:.4f}  | Status: {result_classic.status.name}")
print(f"Quantistico → Obiettivo: {result_quantum.fval:.4f}  | Status: {result_quantum.status.name}")
