"""
inspect_subproblems.py
======================
Ispezione e analisi dei sottoproblemi generati dall'ADMM durante la
risoluzione del problema di allocazione VM-Server.

Questa è la parte più importante dell'assignment: si esplora la struttura
interna dell'ADMMOptimizer per capire come il problema originale viene
decomposto nei sottoproblemi QUBO (binario) e convesso (continuo).

Struttura ADMM (3-block):
    L'ADMM decompone il problema in tre blocchi di variabili:
    - x0: variabili binarie originali (s_i, v_ji)
           → Risolte dal sottoproblema QUBO (con QAOA o classico)
    - z:  variabili di consenso continue
           → Copie continue delle variabili binarie, aggiornate
             dal sottoproblema convesso per "mediare" tra x0 e i vincoli
    - u:  moltiplicatori duali (variabili di Lagrange scalate)
           → Aggiornati ad ogni iterazione per far convergere x0 ≈ z

    Il Lagrangiano Aumentato è:
        L_ρ(x0, z, u) = f(x0) + g(z) + u^T (x0 - z) + (ρ/2) ||x0 - z||²

    dove:
        f(x0) = obiettivo ristretto alle variabili binarie
        g(z)  = obiettivo + vincoli sulle variabili continue
        ρ     = parametro di penalità (rho)

Il sottoproblema QUBO:
    min_x0  f(x0) + u^T x0 + (ρ/2) ||x0 - z||²
    → Questo è un problema quadratico binario (QUBO) perché x0 ∈ {0,1}^n
    → La matrice Q contiene i termini quadratici dall'obiettivo originale
      PIÙ i termini (ρ/2) dalla penalità augmented Lagrangian
    → Il vettore lineare contiene i costi originali PIÙ i moltiplicatori u
      PIÙ i termini -ρ*z dalla penalità

Il sottoproblema convesso:
    min_z  g(z) - u^T z + (ρ/2) ||x0 - z||²
    s.t.   vincoli continui su z
    → Questo è un QP convesso (quadratico con vincoli lineari)
    → Viene risolto con un solver convesso standard (CPLEX, COBYLA, etc.)
"""

from __future__ import annotations

import numpy as np

from qiskit_optimization.problems import QuadraticProgram
from qiskit_optimization.algorithms import ADMMOptimizer, ADMMParameters
from qiskit_optimization.algorithms.admm_optimizer import ADMMOptimizationResult


def inspect_admm_result(
    result: ADMMOptimizationResult,
    qp: QuadraticProgram,
    label: str = "ADMM",
) -> dict:
    """
    Ispeziona lo stato interno dell'ADMMOptimizer dopo la risoluzione.

    Analizza:
    - Lo stato ADMM (ADMMState) con variabili x0, z, u, lambda
    - Le dimensioni del sottoproblema QUBO
    - La struttura del sottoproblema convesso
    - I residui di convergenza

    Parametri
    ---------
    result : ADMMOptimizationResult
        Il risultato dell'ottimizzazione ADMM.
    qp : QuadraticProgram
        Il problema originale (per confronto dimensionale).
    label : str
        Etichetta per la stampa (es. "Classico" o "QAOA").

    Ritorna
    -------
    dict con le informazioni estratte.
    """
    print("\n" + "=" * 70)
    print(f"  ISPEZIONE SOTTOPROBLEMI — {label}")
    print("=" * 70)

    info = {"label": label}

    # ── Stato ADMM ──
    # L'ADMMOptimizationResult ha attributi che forniscono informazioni
    # sullo stato interno della convergenza
    state = result.state if hasattr(result, "state") else None

    if state is not None:
        print("\n  ─── Stato ADMM (ADMMState) ───")

        # x0: soluzione binaria corrente
        if hasattr(state, "x0"):
            x0 = np.array(state.x0)
            print(f"    x0 (variabili binarie):      dimensione = {len(x0)}")
            print(f"    x0 = {x0}")
            info["x0"] = x0

        # z: variabili di consenso
        if hasattr(state, "z"):
            z = np.array(state.z)
            print(f"    z  (variabili di consenso):   dimensione = {len(z)}")
            print(f"    z  = {z}")
            info["z"] = z

        # u: moltiplicatori duali (lambda scalati)
        if hasattr(state, "u"):
            u_dual = np.array(state.u)
            print(f"    u  (moltiplicatori duali):    dimensione = {len(u_dual)}")
            print(f"    u  = {u_dual}")
            info["u"] = u_dual

        # Residui
        if hasattr(state, "converge"):
            converge = state.converge
            print(f"\n    Convergenza raggiunta:        {converge}")
            info["converge"] = converge

        # Residui primali (lista dei residui ad ogni iterazione)
        if hasattr(state, "residuals"):
            residuals = state.residuals
            print(f"    Numero iterazioni eseguite:   {len(residuals)}")
            if len(residuals) > 0:
                print(f"    Residuo iniziale:             {residuals[0]:.6e}")
                print(f"    Residuo finale:               {residuals[-1]:.6e}")
            info["residuals"] = residuals

    # ── Analisi del QUBO sub-problem ──
    print("\n  ─── Sottoproblema QUBO (variabili binarie) ───")

    # L'ADMM gestisce internamente la decomposizione:
    # le variabili binarie vanno nel QUBO sub-problem,
    # le variabili continue vanno nel convex sub-problem.
    # QuadraticProgramToQubo non supporta variabili continue,
    # quindi analizziamo le dimensioni dalla struttura del QP.

    n_vars_original = qp.get_num_vars()
    n_binary = sum(1 for v in qp.variables if v.vartype.name == "BINARY")
    n_continuous = sum(1 for v in qp.variables if v.vartype.name == "CONTINUOUS")

    print(f"    Variabili nel QP originale:       {n_vars_original}")
    print(f"      - binarie (s_i, v_ji):          {n_binary}")
    print(f"      - continue (l_i):               {n_continuous}")
    print(f"    Nel QUBO sub-problem ADMM:")
    print(f"      L'ADMM costruisce internamente un QUBO con le {n_binary} variabili")
    print(f"      binarie, più termini di penalità ADMM (ρ/2)||x0 - z||²")
    print(f"      e i moltiplicatori duali u^T x0.")

    info["n_vars_original"] = n_vars_original
    info["n_binary"] = n_binary
    info["n_continuous"] = n_continuous
    # Stima variabili QUBO (binarie + slack per vincoli <=)
    n_ineq_for_qubo = sum(1 for c in qp.linear_constraints if c.sense.name in ("LE", "GE"))
    info["n_vars_qubo"] = n_binary  # nel contesto ADMM, il QUBO è sulle sole binarie
    info["n_slack"] = 0  # l'ADMM non aggiunge slack nel sub-problem QUBO

    # Mostra le variabili binarie e i loro ruoli
    print(f"\n    Variabili binarie nel QUBO sub-problem:")
    bin_vars = [v.name for v in qp.variables if v.vartype.name == "BINARY"]
    for vn in bin_vars:
        if vn.startswith("s_"):
            print(f"      {vn:>10s}  — server on/off")
        elif vn.startswith("v_"):
            parts = vn.split("_")
            print(f"      {vn:>10s}  — VM {parts[1]} → Server {parts[2]}")

    print(f"\n    Variabili continue nel convex sub-problem:")
    cont_vars = [v.name for v in qp.variables if v.vartype.name == "CONTINUOUS"]
    for vn in cont_vars:
        print(f"      {vn:>10s}  — carico CPU continuo")

    # ── Sottoproblema convesso (ADMM) ──
    print("\n  ─── Sottoproblema Convesso (variabili continue) ───")

    # Nel framework ADMM il sottoproblema continuo ottimizza le variabili
    # continue l_i (carico CPU) soggette ai vincoli lineari
    n_constraints = qp.get_num_linear_constraints()
    n_eq = sum(1 for c in qp.linear_constraints if c.sense.name == "EQ")
    n_ineq = sum(1 for c in qp.linear_constraints if c.sense.name in ("LE", "GE"))

    print(f"    Variabili continue (l_i):          {n_continuous}")
    print(f"    Vincoli lineari totali:            {n_constraints}")
    print(f"      - Vincoli di uguaglianza (==):   {n_eq}  (load_def + assign)")
    print(f"      - Vincoli di disuguaglianza:     {n_ineq}  (capacity)")
    print(f"\n    Il sottoproblema convesso nell'ADMM:")
    print(f"      Minimizza g(z) - u^T z + (ρ/2) ||x0 - z||²")
    print(f"      dove z include le variabili continue l_i")
    print(f"      soggetto ai vincoli: l_i = Σ_j u_j * v_ji, l_i <= C_i * s_i")
    print(f"    Questo è un QP convesso standard risolto con solver classico.")

    info["n_constraints"] = n_constraints
    info["n_eq"] = n_eq
    info["n_ineq"] = n_ineq

    # ── Hamiltoniano di Ising ──
    print("\n  ─── Hamiltoniano di Ising (per il QUBO sub-problem) ───")
    print(f"    Nel contesto ADMM, il sottoproblema QUBO sulle {n_binary} variabili")
    print(f"    binarie viene convertito in un Hamiltoniano di Ising con {n_binary} qubit.")
    print(f"    La mappatura QUBO → Ising usa: x_i = (1 - Z_i) / 2")
    print(f"    dove Z_i è l'operatore di Pauli Z sul qubit i.")
    info["n_qubits"] = n_binary

    print("\n" + "=" * 70)
    return info


def print_admm_decomposition_summary(info: dict) -> None:
    """
    Stampa un riassunto della decomposizione ADMM
    evidenziando il significato di x0, z, u.
    """
    print("\n" + "─" * 70)
    print("  RIASSUNTO DECOMPOSIZIONE ADMM")
    print("─" * 70)
    print("""
    L'ADMM decompone il problema originale di allocazione VM
    nei seguenti sottoproblemi:

    1. QUBO Sub-Problem (variabili binarie x0):
       - Contiene le decisioni discrete: s_i (server on/off), v_ji (assegnamento)
       - La funzione obiettivo include i costi energetici originali
         PIÙ i termini di penalità ADMM: u^T x0 + (ρ/2)||x0 - z||²
       - Risolto con QAOA (quantistico) o NumPy (classico)
       - La matrice Q del QUBO codifica TUTTE le interazioni:
         costi quadratici + penalità vincoli + penalità ADMM

    2. Continuous Sub-Problem (variabili di consenso z):
       - z è una copia continua (rilassata) delle variabili binarie
       - Minimizza: g(z) - u^T z + (ρ/2)||x0 - z||²
       - Soggetto ai vincoli lineari originali (capacità + assegnamento)
       - Risolto con un solver convesso (QP continuo standard)
       - z "media" tra la soluzione binaria e la fattibilità

    3. Aggiornamento moltiplicatori duali u:
       - u ← u + ρ (x0 - z)
       - Forza la convergenza verso x0 = z (consenso tra i blocchi)
       - Quando ||x0 - z|| → 0, l'ADMM è convergente
    """)

    if "n_vars_qubo" in info:
        print(f"    Dimensioni per M=2, N=3:")
        print(f"      Variabili binarie:            {info.get('n_binary', '?')}")
        print(f"      Variabili continue:           {info.get('n_continuous', '?')}")
        print(f"      Variabili originali totali:   {info.get('n_vars_original', '?')}")
        print(f"      Variabili QUBO (con slack):   {info.get('n_vars_qubo', '?')}")
        print(f"      Qubit Ising:                  {info.get('n_qubits', '?')}")
        print(f"      Vincoli uguaglianza:          {info.get('n_eq', '?')}")
        print(f"      Vincoli disuguaglianza:       {info.get('n_ineq', '?')}")

    print("─" * 70)


# ──────────────────────────────────────────────────────────────────────
# Test standalone
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from problem_formulation import build_quadratic_program
    from admm_solver import solve_classical

    qp = build_quadratic_program()
    result = solve_classical(qp)
    info = inspect_admm_result(result, qp, label="Classico (standalone)")
    print_admm_decomposition_summary(info)
