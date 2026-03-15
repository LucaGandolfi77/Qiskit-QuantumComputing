"""
results_analysis.py
===================
Visualizzazione e analisi dei risultati dell'ottimizzazione ADMM
per il problema di allocazione VM-Server.

Include:
    - Plot della convergenza (residui primali vs iterazioni)
    - Tabella dell'allocazione finale (quale VM → quale server)
    - Confronto obiettivo classico vs QAOA
    - Calcolo del consumo energetico totale e utilizzo CPU
"""

from __future__ import annotations

import numpy as np

from qiskit_optimization.algorithms.admm_optimizer import ADMMOptimizationResult

# Parametri di default (stessi di problem_formulation.py)
DEFAULT_M = 2
DEFAULT_N = 3
DEFAULT_PI = [100, 120]
DEFAULT_PD = [50, 60]
DEFAULT_C = [4, 5]
DEFAULT_U = [2, 1, 3]


def decode_solution(
    result: ADMMOptimizationResult,
    M: int = DEFAULT_M,
    N: int = DEFAULT_N,
) -> dict:
    """
    Decodifica il vettore soluzione x in variabili s_i, v_ji e l_i.

    L'ordinamento delle variabili nel QuadraticProgram è:
        s_0, s_1, v_0_0, v_0_1, v_1_0, v_1_1, v_2_0, v_2_1, l_0, l_1

    Ritorna
    -------
    dict con chiavi 's' (array server), 'v' (matrice NxM assegnamento),
    'l' (array carichi continui), 'fval' (valore obiettivo).
    """
    x = np.array(result.x)

    # Primi M valori sono le variabili s_i
    s = np.round(x[:M]).astype(int)

    # I successivi N*M valori sono v_ji (linearizzati per j, poi per i)
    v_flat = np.round(x[M: M + N * M]).astype(int)
    v = v_flat.reshape(N, M)

    # Gli ultimi M valori sono le variabili continue l_i (carico server)
    l = x[M + N * M: M + N * M + M]

    return {"s": s, "v": v, "l": l, "fval": result.fval}


def print_allocation_table(
    decoded: dict,
    M: int = DEFAULT_M,
    N: int = DEFAULT_N,
    PI: list[float] = None,
    PD: list[float] = None,
    C: list[float] = None,
    U: list[float] = None,
) -> None:
    """
    Stampa una tabella ASCII con l'allocazione finale VM → Server.
    """
    PI = PI if PI is not None else DEFAULT_PI
    PD = PD if PD is not None else DEFAULT_PD
    C = C if C is not None else DEFAULT_C
    U = U if U is not None else DEFAULT_U

    s = decoded["s"]
    v = decoded["v"]

    print("\n" + "=" * 60)
    print("  ALLOCAZIONE FINALE VM → SERVER")
    print("=" * 60)

    # Stato dei server
    print("\n  Server:")
    print(f"  {'Server':<10} {'Stato':<10} {'P_idle':<10} {'P_dyn':<10} {'Capacità':<10}")
    print("  " + "-" * 50)
    for i in range(M):
        stato = "ACCESO" if s[i] == 1 else "SPENTO"
        print(f"  Server {i:<3} {stato:<10} {PI[i]:<10.0f} {PD[i]:<10.0f} {C[i]:<10.0f}")

    # Assegnamento VM
    print("\n  Assegnamento VM:")
    print(f"  {'VM':<8} {'Carico':<10} {'Server':<10}")
    print("  " + "-" * 28)
    for j in range(N):
        server_assegnato = -1
        for i in range(M):
            if v[j][i] == 1:
                server_assegnato = i
                break
        server_str = f"Server {server_assegnato}" if server_assegnato >= 0 else "NESSUNO"
        print(f"  VM {j:<4} {U[j]:<10.0f} {server_str:<10}")

    # Utilizzo CPU per server
    print("\n  Utilizzo CPU per server:")
    print(f"  {'Server':<10} {'l_i (cont.)':<12} {'Carico bin.':<12} {'Capacità':<10} {'Utilizzo %':<10}")
    print("  " + "-" * 54)
    for i in range(M):
        if s[i] == 1:
            carico_binario = sum(U[j] * v[j][i] for j in range(N))
            l_val = decoded.get("l", [None] * M)[i]
            l_str = f"{l_val:.2f}" if l_val is not None else "N/A"
            utilizzo_pct = (carico_binario / C[i]) * 100 if C[i] > 0 else 0
            print(f"  Server {i:<3} {l_str:<12} {carico_binario:<12.0f} {C[i]:<10.0f} {utilizzo_pct:<10.1f}")
        else:
            print(f"  Server {i:<3} {'N/A':<12} {'N/A':<12} {C[i]:<10.0f} {'SPENTO':<10}")

    # Consumo energetico
    print("\n  Consumo energetico:")
    consumo_idle_totale = sum(PI[i] * s[i] for i in range(M))
    consumo_dyn_totale = sum(
        PD[i] * U[j] * v[j][i] for i in range(M) for j in range(N)
    )
    consumo_totale = consumo_idle_totale + consumo_dyn_totale

    print(f"    Consumo idle totale:       {consumo_idle_totale:.0f} W")
    print(f"    Consumo dinamico totale:   {consumo_dyn_totale:.0f} W")
    print(f"    CONSUMO TOTALE:            {consumo_totale:.0f} W")
    print(f"    Valore obiettivo ADMM:     {decoded['fval']:.4f}")
    print(f"    Server accesi:             {sum(s)}/{M}")
    print("=" * 60)


def compare_results(
    result_classical: ADMMOptimizationResult,
    result_qaoa: ADMMOptimizationResult,
    M: int = DEFAULT_M,
    N: int = DEFAULT_N,
) -> None:
    """
    Confronta i risultati del solver classico e QAOA.
    """
    dec_cl = decode_solution(result_classical, M, N)
    dec_qa = decode_solution(result_qaoa, M, N)

    print("\n" + "=" * 60)
    print("  CONFRONTO CLASSICO vs QAOA")
    print("=" * 60)
    print(f"  {'Metrica':<30} {'Classico':<15} {'QAOA':<15}")
    print("  " + "-" * 60)
    print(f"  {'Obiettivo':<30} {dec_cl['fval']:<15.4f} {dec_qa['fval']:<15.4f}")
    print(f"  {'Server accesi':<30} {sum(dec_cl['s']):<15} {sum(dec_qa['s']):<15}")

    # Verifica se le soluzioni coincidono
    match = np.array_equal(dec_cl['s'], dec_qa['s']) and np.array_equal(dec_cl['v'], dec_qa['v'])
    print(f"  {'Soluzioni identiche':<30} {'Sì' if match else 'No':<15}")

    gap = abs(dec_qa['fval'] - dec_cl['fval'])
    gap_pct = (gap / abs(dec_cl['fval'])) * 100 if dec_cl['fval'] != 0 else 0
    print(f"  {'Gap assoluto':<30} {gap:<15.4f}")
    print(f"  {'Gap relativo':<30} {gap_pct:<14.2f}%")
    print("=" * 60)


def plot_convergence(
    result_classical: ADMMOptimizationResult = None,
    result_qaoa: ADMMOptimizationResult = None,
    save_path: str = "convergence.png",
) -> None:
    """
    Genera il plot della convergenza dei residui primali
    in funzione delle iterazioni ADMM.

    I residui primali misurano ||x0 - z||, cioè la discrepanza
    tra la soluzione binaria e le variabili di consenso continue.
    Quando convergono a zero, x0 ≈ z e l'ADMM ha trovato consenso.
    """
    import matplotlib
    matplotlib.use("Agg")  # backend non interattivo
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    has_data = False

    if result_classical is not None:
        state_cl = result_classical.state if hasattr(result_classical, "state") else None
        if state_cl and hasattr(state_cl, "residuals") and len(state_cl.residuals) > 0:
            residuals_cl = state_cl.residuals
            ax.plot(range(1, len(residuals_cl) + 1), residuals_cl,
                    "b-o", markersize=3, label="Classico (NumPy)")
            has_data = True

    if result_qaoa is not None:
        state_qa = result_qaoa.state if hasattr(result_qaoa, "state") else None
        if state_qa and hasattr(state_qa, "residuals") and len(state_qa.residuals) > 0:
            residuals_qa = state_qa.residuals
            ax.plot(range(1, len(residuals_qa) + 1), residuals_qa,
                    "r-s", markersize=3, label="QAOA")
            has_data = True

    if has_data:
        ax.set_xlabel("Iterazione ADMM", fontsize=12)
        ax.set_ylabel("Residuo primale ||x0 - z||", fontsize=12)
        ax.set_title("Convergenza ADMM — Residui Primali", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"\n[PLOT] Convergenza salvata in: {save_path}")
    else:
        print("\n[WARN] Nessun dato di convergenza disponibile per il plot.")
        print("       Possibile che ADMMOptimizationResult non esponga 'state.residuals'.")
        print("       Generazione di un plot alternativo basato sui valori obiettivo...")

        # Fallback: plot semplice con barre per confronto obiettivi
        labels = []
        values = []
        if result_classical is not None:
            labels.append("Classico")
            values.append(result_classical.fval)
        if result_qaoa is not None:
            labels.append("QAOA")
            values.append(result_qaoa.fval)

        if values:
            ax.bar(labels, values, color=["steelblue", "coral"][:len(values)])
            ax.set_ylabel("Valore Obiettivo (W)", fontsize=12)
            ax.set_title("Confronto Valore Obiettivo", fontsize=14)
            ax.grid(True, alpha=0.3, axis="y")
            for idx, val in enumerate(values):
                ax.text(idx, val + 2, f"{val:.1f}", ha="center", fontsize=11)
            plt.tight_layout()
            plt.savefig(save_path, dpi=150)
            print(f"[PLOT] Confronto obiettivi salvato in: {save_path}")

    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# Test standalone
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from problem_formulation import build_quadratic_program
    from admm_solver import solve_classical, solve_qaoa

    qp = build_quadratic_program()

    result_cl = solve_classical(qp)
    result_qa = solve_qaoa(qp)

    dec_cl = decode_solution(result_cl)
    print_allocation_table(dec_cl, label_prefix="Classico")

    dec_qa = decode_solution(result_qa)
    print_allocation_table(dec_qa, label_prefix="QAOA")

    compare_results(result_cl, result_qa)
    plot_convergence(result_cl, result_qa)
