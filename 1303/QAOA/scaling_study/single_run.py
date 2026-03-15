"""
single_run.py
=============
Esegue UN singolo run del solver ADMM (classico + QAOA) per una data
configurazione (M, N) e restituisce un dict di metriche.
"""

from __future__ import annotations

import sys
import os
import time
import traceback
import numpy as np

# Aggiungi la cartella QAOA al path per importare i moduli del progetto
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from qiskit_optimization.problems import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.algorithms import ADMMOptimizer, ADMMParameters
from qiskit_optimization.algorithms.admm_optimizer import ADMMOptimizationResult


def _build_qp(M: int, N: int, params: dict) -> QuadraticProgram:
    """Costruisce il QuadraticProgram misto (binarie + continue) per (M, N)."""
    PI = params["P_idle"]
    PD = params["P_dynamic"]
    C = params["C_capacity"]
    U = params["u_cpu"]

    try:
        from docplex.mp.model import Model
        from qiskit_optimization.translators import from_docplex_mp

        mdl = Model(f"VM_Alloc_{M}x{N}")
        s = mdl.binary_var_list(M, name="s")
        v = [[mdl.binary_var(name=f"v_{j}_{i}") for i in range(M)] for j in range(N)]
        l = [mdl.continuous_var(lb=0, ub=C[i], name=f"l_{i}") for i in range(M)]

        mdl.minimize(
            mdl.sum(PI[i] * s[i] for i in range(M))
            + mdl.sum(PD[i] * l[i] for i in range(M))
        )

        for i in range(M):
            mdl.add_constraint(
                l[i] == mdl.sum(U[j] * v[j][i] for j in range(N)),
                ctname=f"load_def_{i}",
            )
        for i in range(M):
            mdl.add_constraint(l[i] <= C[i] * s[i], ctname=f"capacity_{i}")
        for j in range(N):
            mdl.add_constraint(
                mdl.sum(v[j][i] for i in range(M)) == 1,
                ctname=f"assign_{j}",
            )

        return from_docplex_mp(mdl)
    except ImportError:
        pass

    # Fallback manuale
    qp = QuadraticProgram(f"VM_Alloc_{M}x{N}")
    for i in range(M):
        qp.binary_var(name=f"s_{i}")
    for j in range(N):
        for i in range(M):
            qp.binary_var(name=f"v_{j}_{i}")
    for i in range(M):
        qp.continuous_var(lowerbound=0, upperbound=C[i], name=f"l_{i}")

    linear_obj = {}
    for i in range(M):
        linear_obj[f"s_{i}"] = PI[i]
        linear_obj[f"l_{i}"] = PD[i]
    qp.minimize(linear=linear_obj)

    for i in range(M):
        coeff = {f"l_{i}": 1}
        for j in range(N):
            coeff[f"v_{j}_{i}"] = -U[j]
        qp.linear_constraint(linear=coeff, sense="==", rhs=0, name=f"load_def_{i}")

    for i in range(M):
        coeff = {f"l_{i}": 1, f"s_{i}": -C[i]}
        qp.linear_constraint(linear=coeff, sense="<=", rhs=0, name=f"capacity_{i}")

    for j in range(N):
        coeff = {f"v_{j}_{i}": 1 for i in range(M)}
        qp.linear_constraint(linear=coeff, sense="==", rhs=1, name=f"assign_{j}")

    return qp


def _build_binary_only_qp(M: int, N: int, params: dict) -> QuadraticProgram:
    """Versione solo-binaria per analisi dimensionale QUBO."""
    PI = params["P_idle"]
    PD = params["P_dynamic"]
    C = params["C_capacity"]
    U = params["u_cpu"]

    qp = QuadraticProgram(f"VM_Alloc_bin_{M}x{N}")
    for i in range(M):
        qp.binary_var(name=f"s_{i}")
    for j in range(N):
        for i in range(M):
            qp.binary_var(name=f"v_{j}_{i}")

    linear_obj = {}
    for i in range(M):
        linear_obj[f"s_{i}"] = PI[i]
        for j in range(N):
            linear_obj[f"v_{j}_{i}"] = PD[i] * U[j]
    qp.minimize(linear=linear_obj)

    for i in range(M):
        coeff = {}
        for j in range(N):
            coeff[f"v_{j}_{i}"] = U[j]
        coeff[f"s_{i}"] = -C[i]
        qp.linear_constraint(linear=coeff, sense="<=", rhs=0, name=f"capacity_{i}")

    for j in range(N):
        coeff = {f"v_{j}_{i}": 1 for i in range(M)}
        qp.linear_constraint(linear=coeff, sense="==", rhs=1, name=f"assign_{j}")

    return qp


def _get_qubo_info(M: int, N: int, params: dict) -> dict:
    """Calcola dimensioni QUBO e sparsità."""
    n_binary = M + M * N
    qp_bin = _build_binary_only_qp(M, N, params)
    try:
        converter = QuadraticProgramToQubo()
        qubo = converter.convert(qp_bin)
        n_qubo = qubo.get_num_vars()
        n_slack = n_qubo - n_binary

        obj = qubo.objective
        Q_dict = obj.quadratic.to_dict()
        L_dict = obj.linear.to_dict()
        total_cells = n_qubo * n_qubo
        non_zero = len(Q_dict) + len(L_dict)
        sparsity = (non_zero / total_cells * 100) if total_cells > 0 else 0.0

        return {
            "n_qubo": n_qubo,
            "n_slack": n_slack,
            "n_qubits": n_qubo,
            "sparsity": round(sparsity, 2),
        }
    except Exception:
        return {
            "n_qubo": n_binary + M + N,
            "n_slack": M + N,
            "n_qubits": n_binary + M + N,
            "sparsity": 0.0,
        }


def _decode(result: ADMMOptimizationResult, M: int, N: int) -> dict:
    """Decodifica la soluzione x in s, v, l."""
    x = np.array(result.x)
    s = np.round(x[:M]).astype(int)
    v_flat = np.round(x[M: M + N * M]).astype(int)
    v = v_flat.reshape(N, M)
    l = x[M + N * M: M + N * M + M]
    return {"s": s, "v": v, "l": l}


def _extract_residuals(result: ADMMOptimizationResult) -> tuple[list[float], int, bool]:
    """Estrae residui, numero iterazioni e flag convergenza."""
    state = getattr(result, "state", None)
    residuals: list[float] = []
    converged = False
    if state is not None:
        if hasattr(state, "residuals") and len(state.residuals) > 0:
            residuals = [float(r) for r in state.residuals]
        if hasattr(state, "converge"):
            converged = bool(state.converge)
    return residuals, len(residuals), converged


def _compute_energy(dec: dict, M: int, N: int, params: dict) -> tuple[float, int]:
    """Calcola energia totale e server accesi."""
    s, v = dec["s"], dec["v"]
    PI, PD, U = params["P_idle"], params["P_dynamic"], params["u_cpu"]
    total = 0.0
    servers_on = 0
    for i in range(M):
        if s[i] == 1:
            servers_on += 1
            total += PI[i]
            for j in range(N):
                total += PD[i] * U[j] * int(v[j][i])
    return total, servers_on


def check_feasibility(result: ADMMOptimizationResult, M: int, N: int, params: dict) -> bool:
    """
    Verifica manualmente che la soluzione soddisfi tutti i vincoli:
        1. Ogni VM assegnata a esattamente 1 server
        2. Capacità: carico ≤ C_i * s_i per ogni server
        3. Variabili binarie in {0, 1}
    """
    dec = _decode(result, M, N)
    s, v = dec["s"], dec["v"]
    C, U = params["C_capacity"], params["u_cpu"]

    # Assegnamento: ogni VM su esattamente 1 server
    for j in range(N):
        if v[j].sum() != 1:
            return False

    # Capacità
    for i in range(M):
        load = sum(U[j] * int(v[j][i]) for j in range(N))
        if load > C[i] * int(s[i]) + 1e-6:
            return False

    return True


def _get_admm_params() -> ADMMParameters:
    """Parametri ADMM condivisi (stessi del progetto principale)."""
    return ADMMParameters(
        rho_initial=10,
        factor_c=100000,
        beta=10000,
        maxiter=100,
        tol=1e-4,
        three_block=True,
    )


def run_configuration(
    M: int,
    N: int,
    params: dict,
    timeout_sec: int = 120,
) -> dict:
    """
    Esegue un singolo run (M, N): classico + QAOA.

    Ritorna un dict con tutte le metriche richieste dallo scaling study.
    """
    qp = _build_qp(M, N, params)
    qubo_info = _get_qubo_info(M, N, params)

    n_binary = M + M * N
    admm_params = _get_admm_params()

    # ── Classico ──
    from qiskit_algorithms import NumPyMinimumEigensolver
    from qiskit_optimization.algorithms import MinimumEigenOptimizer

    numpy_solver = NumPyMinimumEigensolver()
    qubo_optimizer_cl = MinimumEigenOptimizer(numpy_solver)
    admm_cl = ADMMOptimizer(params=admm_params, qubo_optimizer=qubo_optimizer_cl)

    t0 = time.perf_counter()
    result_cl = admm_cl.solve(qp)
    cl_time = time.perf_counter() - t0

    dec_cl = _decode(result_cl, M, N)
    cl_residuals, cl_iter, cl_converged = _extract_residuals(result_cl)
    cl_energy, cl_servers_on = _compute_energy(dec_cl, M, N, params)
    cl_feasible = check_feasibility(result_cl, M, N, params)

    # ── QAOA ──
    qaoa_obj = None
    qaoa_iter = None
    qaoa_converged = False
    qaoa_time = None
    qaoa_residuals = None
    qaoa_energy = None
    qaoa_error = False
    qaoa_error_msg = ""

    try:
        from qiskit.primitives import StatevectorSampler
        from qiskit_optimization.minimum_eigensolvers import QAOA
        from qiskit_algorithms.optimizers import COBYLA

        sampler = StatevectorSampler()
        qaoa = QAOA(sampler=sampler, reps=1, optimizer=COBYLA(maxiter=200))
        qubo_optimizer_qa = MinimumEigenOptimizer(qaoa)
        admm_qa = ADMMOptimizer(params=admm_params, qubo_optimizer=qubo_optimizer_qa)

        t0 = time.perf_counter()
        result_qa = admm_qa.solve(qp)
        qaoa_time = time.perf_counter() - t0

        dec_qa = _decode(result_qa, M, N)
        qaoa_residuals_list, qa_iter, qa_conv = _extract_residuals(result_qa)
        qa_energy, _ = _compute_energy(dec_qa, M, N, params)

        qaoa_obj = float(result_qa.fval)
        qaoa_iter = qa_iter
        qaoa_converged = qa_conv
        qaoa_residuals = qaoa_residuals_list
        qaoa_energy = qa_energy

    except Exception as e:
        qaoa_error = True
        qaoa_error_msg = str(e)

    # ── Metriche derivate ──
    obj_diff_pct = None
    iter_ratio = None
    time_ratio = None
    if qaoa_obj is not None and result_cl.fval != 0:
        obj_diff_pct = (qaoa_obj - result_cl.fval) / abs(result_cl.fval) * 100
    if qaoa_iter is not None and cl_iter > 0:
        iter_ratio = qaoa_iter / cl_iter
    if qaoa_time is not None and cl_time > 0:
        time_ratio = qaoa_time / cl_time

    return {
        "M": M,
        "N": N,
        "n_binary_vars_original": n_binary,
        "n_slack_vars": qubo_info["n_slack"],
        "n_qubo_vars": qubo_info["n_qubo"],
        "n_qubits": qubo_info["n_qubits"],
        "qubo_sparsity": qubo_info["sparsity"],
        "continuous_vars": M,
        "feasible": cl_feasible,

        "classical_obj": float(result_cl.fval),
        "classical_iter": cl_iter,
        "classical_converged": cl_converged,
        "classical_time_sec": round(cl_time, 4),
        "classical_residuals": cl_residuals,
        "classical_energy_total": cl_energy,
        "classical_servers_on": cl_servers_on,

        "qaoa_obj": qaoa_obj,
        "qaoa_iter": qaoa_iter,
        "qaoa_converged": qaoa_converged,
        "qaoa_time_sec": round(qaoa_time, 4) if qaoa_time is not None else None,
        "qaoa_residuals": qaoa_residuals,
        "qaoa_energy_total": qaoa_energy,
        "qaoa_error": qaoa_error,
        "qaoa_error_msg": qaoa_error_msg,

        "obj_diff_pct": round(obj_diff_pct, 4) if obj_diff_pct is not None else None,
        "iter_ratio": round(iter_ratio, 4) if iter_ratio is not None else None,
        "time_ratio": round(time_ratio, 4) if time_ratio is not None else None,
    }


if __name__ == "__main__":
    # Quick test con la config più piccola
    params = {
        "M": 1, "N": 1,
        "P_idle": [100], "P_dynamic": [50],
        "C_capacity": [4], "u_cpu": [2],
    }
    result = run_configuration(1, 1, params)
    print(f"M=1, N=1 → obj_cl={result['classical_obj']:.2f}, "
          f"qubits={result['n_qubits']}, "
          f"qaoa_error={result['qaoa_error']}")
