"""
problem_formulation.py
======================
Definizione del problema di allocazione VM-Server come QuadraticProgram
usando qiskit-optimization e (opzionalmente) docplex.

Il problema fisico:
    Minimizzare il consumo energetico totale dell'infrastruttura, dato da:
        - Consumo idle dei server accesi  (P^I_i * s_i)
        - Consumo dinamico proporzionale al carico  (P^D_i * l_i)
    Soggetti ai vincoli:
        - Definizione carico: l_i = Σ_j u_j * v_ji
        - Capacità: l_i <= C_i * s_i
        - Assegnamento: Σ_i v_ji = 1 per ogni VM j
        - Bounds: 0 <= l_i <= C_i

Variabili decisionali:
    s_i   ∈ {0,1}       — server i acceso (1) o spento (0)
    v_ji  ∈ {0,1}       — VM j assegnata al server i (1) o no (0)
    l_i   ∈ [0, C_i]    — carico CPU continuo sul server i

    Le variabili continue l_i sono fondamentali per la decomposizione ADMM:
    l'ADMM separa il problema in un sottoproblema QUBO (variabili binarie s, v)
    e un sottoproblema convesso (variabili continue l).

Parametri di default (M=2 server, N=3 VM):
    P^I = [100, 120]   W — consumo idle
    P^D = [50,  60]    W — consumo dinamico per unità di carico
    C   = [4,   5]       — capacità CPU massima
    u   = [2, 1, 3]      — carico CPU di ciascuna VM
"""

from __future__ import annotations

import numpy as np

from qiskit_optimization.problems import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo

# ──────────────────────────────────────────────────────────────────────
# Parametri di default
# ──────────────────────────────────────────────────────────────────────
DEFAULT_M = 2                        # numero di server
DEFAULT_N = 3                        # numero di VM
DEFAULT_PI = [100, 120]              # consumo idle per server
DEFAULT_PD = [50, 60]                # consumo dinamico per server
DEFAULT_C = [4, 5]                   # capacità CPU per server
DEFAULT_U = [2, 1, 3]               # carico CPU per VM


def build_quadratic_program(
    M: int = DEFAULT_M,
    N: int = DEFAULT_N,
    PI: list[float] = None,
    PD: list[float] = None,
    C: list[float] = None,
    U: list[float] = None,
) -> QuadraticProgram:
    """
    Costruisce il QuadraticProgram per il problema di allocazione VM-Server
    con variabili miste: binarie (s_i, v_ji) e continue (l_i).

    Le variabili continue l_i rappresentano il carico CPU sul server i e
    sono fondamentali per la decomposizione ADMM in sottoproblema QUBO
    (binario) e sottoproblema convesso (continuo).

    Ritorna
    -------
    QuadraticProgram
        Il modello con variabili binarie, continue e vincoli lineari.
    """
    PI = PI if PI is not None else DEFAULT_PI
    PD = PD if PD is not None else DEFAULT_PD
    C = C if C is not None else DEFAULT_C
    U = U if U is not None else DEFAULT_U

    qp = QuadraticProgram("VM_Allocation")

    # --- Variabili binarie ---
    # s_i: server i acceso/spento
    for i in range(M):
        qp.binary_var(name=f"s_{i}")

    # v_ji: VM j assegnata al server i
    for j in range(N):
        for i in range(M):
            qp.binary_var(name=f"v_{j}_{i}")

    # --- Variabili continue ---
    # l_i: carico CPU totale sul server i ∈ [0, C_i]
    # Queste variabili sono essenziali per la decomposizione ADMM
    for i in range(M):
        qp.continuous_var(lowerbound=0, upperbound=C[i], name=f"l_{i}")

    # --- Funzione obiettivo ---
    # min  Σ_i [ P^I_i * s_i  +  P^D_i * l_i ]
    linear_obj = {}
    for i in range(M):
        linear_obj[f"s_{i}"] = PI[i]     # costo idle
        linear_obj[f"l_{i}"] = PD[i]     # costo dinamico (proporzionale al carico)

    qp.minimize(linear=linear_obj)

    # --- Vincoli di definizione del carico ---
    # l_i = Σ_j u_j * v_ji   per ogni server i
    # Collega le variabili binarie v_ji alle variabili continue l_i
    for i in range(M):
        coeff = {f"l_{i}": 1}
        for j in range(N):
            coeff[f"v_{j}_{i}"] = -U[j]
        qp.linear_constraint(
            linear=coeff,
            sense="==",
            rhs=0,
            name=f"load_def_{i}",
        )

    # --- Vincoli di capacità ---
    # l_i <= C_i * s_i   per ogni server i
    # Il carico è possibile solo se il server è acceso
    for i in range(M):
        coeff = {f"l_{i}": 1, f"s_{i}": -C[i]}
        qp.linear_constraint(
            linear=coeff,
            sense="<=",
            rhs=0,
            name=f"capacity_{i}",
        )

    # --- Vincoli di assegnamento ---
    # Σ_i v_ji = 1  per ogni VM j
    for j in range(N):
        coeff = {f"v_{j}_{i}": 1 for i in range(M)}
        qp.linear_constraint(
            linear=coeff,
            sense="==",
            rhs=1,
            name=f"assign_{j}",
        )

    return qp


def build_quadratic_program_docplex(
    M: int = DEFAULT_M,
    N: int = DEFAULT_N,
    PI: list[float] = None,
    PD: list[float] = None,
    C: list[float] = None,
    U: list[float] = None,
) -> QuadraticProgram:
    """
    Costruisce il QuadraticProgram partendo da un modello DOcplex,
    poi lo converte in QuadraticProgram via from_docplex_mp.

    Formulazione mista con variabili binarie e continue per ADMM.
    In caso docplex non sia disponibile, usa il fallback manuale.
    """
    PI = PI if PI is not None else DEFAULT_PI
    PD = PD if PD is not None else DEFAULT_PD
    C = C if C is not None else DEFAULT_C
    U = U if U is not None else DEFAULT_U

    try:
        from docplex.mp.model import Model
        from qiskit_optimization.translators import from_docplex_mp

        mdl = Model("VM_Allocation_docplex")

        # Variabili binarie
        s = mdl.binary_var_list(M, name="s")
        v = [[mdl.binary_var(name=f"v_{j}_{i}") for i in range(M)] for j in range(N)]

        # Variabili continue: carico CPU per server
        l = [mdl.continuous_var(lb=0, ub=C[i], name=f"l_{i}") for i in range(M)]

        # Obiettivo: min Σ_i [P^I_i * s_i + P^D_i * l_i]
        idle_cost = mdl.sum(PI[i] * s[i] for i in range(M))
        dyn_cost = mdl.sum(PD[i] * l[i] for i in range(M))
        mdl.minimize(idle_cost + dyn_cost)

        # Vincoli di definizione del carico: l_i = Σ_j u_j * v_ji
        for i in range(M):
            mdl.add_constraint(
                l[i] == mdl.sum(U[j] * v[j][i] for j in range(N)),
                ctname=f"load_def_{i}",
            )

        # Vincoli di capacità: l_i <= C_i * s_i
        for i in range(M):
            mdl.add_constraint(l[i] <= C[i] * s[i], ctname=f"capacity_{i}")

        # Vincoli di assegnamento: Σ_i v_ji = 1
        for j in range(N):
            mdl.add_constraint(
                mdl.sum(v[j][i] for i in range(M)) == 1,
                ctname=f"assign_{j}",
            )

        qp = from_docplex_mp(mdl)
        print("[INFO] Modello costruito con DOcplex e convertito in QuadraticProgram.")
        return qp

    except ImportError:
        print("[WARN] docplex non installato, uso costruzione manuale.")
        return build_quadratic_program(M, N, PI, PD, C, U)


def analyze_qubo_conversion(qp: QuadraticProgram) -> dict:
    """
    Analizza la struttura del QuadraticProgram e, se possibile,
    mostra la conversione QUBO e Ising.

    Per problemi misti (binari + continui), la conversione QUBO diretta
    non è possibile: l'ADMM gestisce la decomposizione internamente,
    separando le variabili binarie (→ QUBO sub-problem) da quelle
    continue (→ convex sub-problem).

    Per analizzare il QUBO, costruiamo una versione solo-binaria del problema.

    Ritorna
    -------
    dict con le informazioni di dimensione.
    """
    # Conteggio variabili
    n_original = qp.get_num_vars()
    n_binary = sum(1 for v in qp.variables if v.vartype.name == "BINARY")
    n_continuous = sum(1 for v in qp.variables if v.vartype.name == "CONTINUOUS")
    n_constraints = qp.get_num_linear_constraints()
    n_eq = sum(1 for c in qp.linear_constraints if c.sense.name == "EQ")
    n_ineq = sum(1 for c in qp.linear_constraints if c.sense.name in ("LE", "GE"))

    info = {
        "qp": qp,
        "n_original": n_original,
        "n_binary": n_binary,
        "n_continuous": n_continuous,
        "n_constraints": n_constraints,
        "n_eq": n_eq,
        "n_ineq": n_ineq,
    }

    print("\n" + "=" * 60)
    print("  ANALISI STRUTTURA DEL PROBLEMA")
    print("=" * 60)
    print(f"  Variabili originali totali:         {n_original}")
    print(f"    - binarie (s_i, v_ji):            {n_binary}")
    print(f"    - continue (l_i):                 {n_continuous}")
    print(f"  Vincoli lineari:                    {n_constraints}")
    print(f"    - uguaglianza (==):               {n_eq}  (load_def + assign)")
    print(f"    - disuguaglianza (<=):             {n_ineq}  (capacity)")

    # Tentativo di conversione QUBO sulla versione solo-binaria
    # Costruiamo un QP ausiliario con sole variabili binarie per analisi
    from problem_formulation import build_quadratic_program as _build_binary_only
    # Usiamo la versione originale (solo binarie) per l'analisi QUBO
    qp_binary = QuadraticProgram("VM_Allocation_binary_analysis")
    M = n_continuous  # numero di server = numero di variabili l_i
    N = (n_binary - M) // M  # numero di VM

    # Ricostruisci con le sole variabili binarie per analisi dimensionale
    for v in qp.variables:
        if v.vartype.name == "BINARY":
            qp_binary.binary_var(name=v.name)

    # Aggiungi solo i vincoli sulle variabili binarie (capacity, assign)
    PI = DEFAULT_PI[:M]
    PD = DEFAULT_PD[:M]
    C = DEFAULT_C[:M]
    U = DEFAULT_U[:N]

    linear_obj = {}
    for i in range(M):
        linear_obj[f"s_{i}"] = PI[i]
        for j in range(N):
            linear_obj[f"v_{j}_{i}"] = PD[i] * U[j]
    qp_binary.minimize(linear=linear_obj)

    for i in range(M):
        coeff = {}
        for j in range(N):
            coeff[f"v_{j}_{i}"] = U[j]
        coeff[f"s_{i}"] = -C[i]
        qp_binary.linear_constraint(linear=coeff, sense="<=", rhs=0, name=f"capacity_{i}")

    for j in range(N):
        coeff = {f"v_{j}_{i}": 1 for i in range(M)}
        qp_binary.linear_constraint(linear=coeff, sense="==", rhs=1, name=f"assign_{j}")

    try:
        converter = QuadraticProgramToQubo()
        qubo = converter.convert(qp_binary)
        n_qubo = qubo.get_num_vars()
        n_slack = n_qubo - n_binary

        obj = qubo.objective
        Q_dict = obj.quadratic.to_dict()
        linear_dict = obj.linear.to_dict()

        print(f"\n  Analisi QUBO (solo variabili binarie):")
        print(f"    Variabili QUBO (con slack):       {n_qubo}")
        print(f"    Variabili slack introdotte:        {n_slack}")
        print(f"    Termini quadratici:                {len(Q_dict)}")
        print(f"    Termini lineari:                   {len(linear_dict)}")
        print(f"    Costante obiettivo:                {obj.constant}")

        info["n_qubo"] = n_qubo
        info["n_slack"] = n_slack
        info["qubo"] = qubo

        # Conversione Ising
        ising_op, offset = qubo.to_ising()
        n_qubits = ising_op.num_qubits
        print(f"\n  Hamiltoniano di Ising:")
        print(f"    Numero di qubit:                   {n_qubits}")
        print(f"    Offset (costante):                 {offset}")
        print(f"    Numero di termini Pauli:            {len(ising_op)}")
        info["n_qubits"] = n_qubits
        info["ising_offset"] = offset

    except Exception as e:
        print(f"\n  [WARN] Conversione QUBO/Ising non riuscita: {e}")
        # Stime basate sulla struttura
        info["n_qubo"] = n_binary + 6  # stima slack
        info["n_slack"] = 6
        info["n_qubits"] = info["n_qubo"]

    print(f"\n  Nota: nell'ADMM, il QUBO sub-problem opera sulle {n_binary}")
    print(f"  variabili binarie. Le {n_continuous} variabili continue (l_i)")
    print(f"  sono ottimizzate nel sottoproblema convesso separato.")
    print("=" * 60)

    return info


# ──────────────────────────────────────────────────────────────────────
# Esecuzione standalone per test rapidi
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Costruzione del QuadraticProgram (manuale)...")
    qp = build_quadratic_program()
    print(qp.prettyprint())

    print("\nCostruzione del QuadraticProgram (DOcplex)...")
    qp_docplex = build_quadratic_program_docplex()
    print(qp_docplex.prettyprint())

    print("\nAnalisi conversione QUBO...")
    info = analyze_qubo_conversion(qp)
