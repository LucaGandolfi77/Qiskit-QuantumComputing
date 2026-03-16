"""
admm_solver.py
==============
Configurazione e lancio dell'ADMMOptimizer per risolvere il problema
di allocazione VM-Server.

L'ADMM (Alternating Direction Method of Multipliers) decompone il problema
misto in due sottoproblemi:
    1. QUBO sub-problem: risolve le variabili binarie (s_i, v_ji)
       → può essere risolto con QAOA (quantistico) o NumPy (classico)
    2. Continuous sub-problem: risolve le variabili continue introdotte
       dal rilassamento ADMM (variabili di consenso z e moltiplicatori u)
       → risolto con un solver convesso (COBYLA o simile)

Parametri ADMM:
    rho (ρ) = 10    — parametro di penalità iniziale per il termine augmented Lagrangian
    factor_c = 100000 — penalità per i vincoli (deve dominare su ρ)
    beta = 10000    — penalità per i vincoli di uguaglianza
    max_iter = 100  — numero massimo di iterazioni ADMM
    tol = 1e-4      — tolleranza per la convergenza dei residui
"""

from __future__ import annotations

from qiskit_optimization.algorithms import ADMMOptimizer, ADMMParameters
from qiskit_optimization.algorithms.admm_optimizer import ADMMOptimizationResult
from qiskit_optimization.problems import QuadraticProgram


def get_admm_parameters() -> ADMMParameters:
    """
    Restituisce i parametri ADMM condivisi tra solver classico e quantistico.

    rho_initial: penalità iniziale del termine quadratico di consenso
    factor_c:    fattore di penalità per i vincoli (scaling)
    maxiter:     iterazioni massime
    tol:         tolleranza sui residui primali e duali
    """
    params = ADMMParameters(
        rho_initial=10,
        factor_c=100000,
        beta=10000,
        maxiter=100,
        tol=1e-4,
        three_block=True,    # decomposizione a 3 blocchi (binario, continuo, slack)
    )
    return params


def solve_classical(qp: QuadraticProgram) -> ADMMOptimizationResult:
    """
    Risolve il problema usando ADMM con sottoproblema QUBO risolto
    classicamente tramite NumPyMinimumEigensolver.

    Questa configurazione serve come baseline per confrontare
    con la soluzione quantistica.
    """
    from qiskit_algorithms import NumPyMinimumEigensolver
    from qiskit_optimization.algorithms import MinimumEigenOptimizer

    # Solver classico per il sottoproblema QUBO
    numpy_solver = NumPyMinimumEigensolver()
    qubo_optimizer = MinimumEigenOptimizer(numpy_solver)

    # Parametri ADMM
    params = get_admm_parameters()

    # Creazione e lancio dell'ADMMOptimizer
    admm = ADMMOptimizer(params=params, qubo_optimizer=qubo_optimizer)

    print("\n[ADMM Classico] Risoluzione in corso...")
    result = admm.solve(qp)
    print(f"[ADMM Classico] Completato — Obiettivo: {result.fval:.4f}")

    return result


def solve_qaoa(qp: QuadraticProgram) -> ADMMOptimizationResult:
    """
    Risolve il problema usando ADMM con sottoproblema QUBO risolto
    tramite QAOA (Quantum Approximate Optimization Algorithm).

    QAOA utilizza un circuito variazionale parametrizzato con p=2 ripetizioni
    (reps=2) e viene eseguito su un simulatore statevector.

    Importazioni:
    - QAOA da qiskit_optimization.minimum_eigensolvers (wrapper compatibile)
    - StatevectorSampler da qiskit.primitives (primitive V2)
    """
    from qiskit.primitives import StatevectorSampler
    from qiskit_optimization.minimum_eigensolvers import QAOA
    from qiskit_optimization.algorithms import MinimumEigenOptimizer
    from qiskit_algorithms.optimizers import COBYLA

    # Sampler statevector (simulazione esatta, noiseless)
    sampler = StatevectorSampler()

    # QAOA con p=2 livelli (reps=2)
    # L'optimizer classico interno a QAOA usa COBYLA per ottimizzare
    # i parametri gamma e beta del circuito variazionale
    qaoa = QAOA(
        sampler=sampler,
        reps=2,
        optimizer=COBYLA(maxiter=200),
    )

    qubo_optimizer = MinimumEigenOptimizer(qaoa)

    # Parametri ADMM
    params = get_admm_parameters()

    # Continuous optimizer per il sottoproblema convesso (CobylaOptimizer)
    # ADMMOptimizer accetta continuous_optimizer per risolvere il sub-problem continuo
    admm = ADMMOptimizer(
        params=params,
        qubo_optimizer=qubo_optimizer,
    )

    print("\n[ADMM + QAOA] Risoluzione in corso...")
    result = admm.solve(qp)
    print(f"[ADMM + QAOA] Completato — Obiettivo: {result.fval:.4f}")

    return result


# ──────────────────────────────────────────────────────────────────────
# Test standalone
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from problem_formulation import build_quadratic_program

    qp = build_quadratic_program()

    print("=" * 60)
    print("  TEST ADMM SOLVER")
    print("=" * 60)

    result_classical = solve_classical(qp)
    print(f"\n  Risultato classico: x = {result_classical.x}")
    print(f"  Obiettivo classico:     {result_classical.fval}")

    result_qaoa = solve_qaoa(qp)
    print(f"\n  Risultato QAOA:     x = {result_qaoa.x}")
    print(f"  Obiettivo QAOA:         {result_qaoa.fval}")
