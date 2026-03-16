import os
import json
import time
import math
import traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError

import pandas as pd
import sys

# Try import local `cupy_utils` (one level up); provide NumPy fallback if missing
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
try:
    import cupy_utils as cu
except Exception:
    import numpy as _np

    class _CuFallback:
        xp = _np

        @staticmethod
        def asarray(x):
            return _np.asarray(x)

        @staticmethod
        def asnumpy(x):
            return _np.asarray(x)

    cu = _CuFallback()

# ============================================================
# IMPORT QISKIT A LIVELLO MODULO
# ============================================================

from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import StatevectorSampler
from qiskit_optimization.algorithms import MinimumEigenOptimizer, ADMMOptimizer
from qiskit_optimization.minimum_eigensolvers import QAOA
from qiskit_optimization.problems import QuadraticProgram

from qiskit_optimization.problems import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.algorithms import ADMMOptimizer, ADMMParameters
from qiskit_optimization.algorithms.admm_optimizer import ADMMOptimizationResult

# ============================================================
# QUI DEVONO ESISTERE NEL TUO PROGETTO
# ============================================================
# - _build_qp(M, N, params)
# - _get_qubo_info(M, N, params)
# - _get_admm_params()
# - _decode(result, M, N)
# - _extract_residuals(result)
# - _compute_energy(decoded, M, N, params)
# - check_feasibility(result, M, N, params)
#
# Lascio i nomi invariati così puoi incollare questo file e
# importare o mantenere le tue funzioni originali.
# ============================================================


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
        from performance_utils import convert_qubo_cached

        qubo = convert_qubo_cached(qp_bin)
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
    # Use cupy_utils to place arrays on GPU when available (best-effort)
    x_dev = cu.asarray(result.x)
    # Round on device
    s_dev = cu.xp.rint(x_dev[:M]).astype(int)
    v_flat_dev = cu.xp.rint(x_dev[M: M + N * M]).astype(int)
    # Bring back to numpy for compatibility with rest of code
    s = cu.asnumpy(s_dev).astype(int)
    v = cu.asnumpy(v_flat_dev).reshape(N, M)
    l = cu.asnumpy(x_dev[M + N * M: M + N * M + M])
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
    run_qaoa: bool = True,
    qaoa_max_qubits: int | None = 20,
    qaoa_reps: int = 1,
    qaoa_maxiter: int = 100,
) -> dict:
    """
    Esegue un singolo run (M, N): classico + QAOA.
    Restituisce un dict con metriche e diagnostica.
    """

    t_run0 = time.perf_counter()

    qp = _build_qp(M, N, params)
    qubo_info = _get_qubo_info(M, N, params)

    n_binary = M + M * N
    admm_params = _get_admm_params()

    # ── Classico ──
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
    qaoa_skipped = False
    qaoa_skip_reason = ""

    if not run_qaoa:
        qaoa_skipped = True
        qaoa_skip_reason = "run_qaoa=False"
    elif qaoa_max_qubits is not None and qubo_info["n_qubits"] > qaoa_max_qubits:
        qaoa_skipped = True
        qaoa_skip_reason = f"n_qubits>{qaoa_max_qubits}"
    else:
        try:
            sampler = StatevectorSampler()
            qaoa = QAOA(
                sampler=sampler,
                reps=qaoa_reps,
                optimizer=COBYLA(maxiter=qaoa_maxiter),
            )
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

    total_wall = time.perf_counter() - t_run0

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
        "qaoa_skipped": qaoa_skipped,
        "qaoa_skip_reason": qaoa_skip_reason,

        "obj_diff_pct": round(obj_diff_pct, 4) if obj_diff_pct is not None else None,
        "iter_ratio": round(iter_ratio, 4) if iter_ratio is not None else None,
        "time_ratio": round(time_ratio, 4) if time_ratio is not None else None,

        "run_wall_time_sec": round(total_wall, 4),
    }


def _worker_run(job: tuple) -> dict:
    """
    Worker top-level, picklable.
    job = (M, N, params, timeout_sec, run_qaoa, qaoa_max_qubits, qaoa_reps, qaoa_maxiter)
    """
    M, N, params, timeout_sec, run_qaoa, qaoa_max_qubits, qaoa_reps, qaoa_maxiter = job
    started = time.perf_counter()

    try:
        result = run_configuration(
            M=M,
            N=N,
            params=params,
            timeout_sec=timeout_sec,
            run_qaoa=run_qaoa,
            qaoa_max_qubits=qaoa_max_qubits,
            qaoa_reps=qaoa_reps,
            qaoa_maxiter=qaoa_maxiter,
        )
        result["worker_status"] = "ok"
        result["worker_error"] = ""
    except Exception as e:
        result = {
            "M": M,
            "N": N,
            "worker_status": "error",
            "worker_error": str(e),
            "worker_traceback": traceback.format_exc(),

            "n_binary_vars_original": None,
            "n_slack_vars": None,
            "n_qubo_vars": None,
            "n_qubits": None,
            "qubo_sparsity": None,
            "continuous_vars": None,
            "feasible": None,

            "classical_obj": None,
            "classical_iter": None,
            "classical_converged": None,
            "classical_time_sec": None,
            "classical_residuals": None,
            "classical_energy_total": None,
            "classical_servers_on": None,

            "qaoa_obj": None,
            "qaoa_iter": None,
            "qaoa_converged": None,
            "qaoa_time_sec": None,
            "qaoa_residuals": None,
            "qaoa_energy_total": None,
            "qaoa_error": True,
            "qaoa_error_msg": str(e),
            "qaoa_skipped": None,
            "qaoa_skip_reason": "",

            "obj_diff_pct": None,
            "iter_ratio": None,
            "time_ratio": None,
        }

    result["worker_wall_time_sec"] = round(time.perf_counter() - started, 4)
    return result


def run_configurations_parallel(
    configurations: list[tuple[int, int]],
    params: dict,
    timeout_sec: int = 120,
    max_workers: int | None = None,
    run_qaoa: bool = True,
    qaoa_max_qubits: int | None = 20,
    qaoa_reps: int = 1,
    qaoa_maxiter: int = 100,
    global_timeout_sec: int | None = None,
    save_csv_path: str | None = None,
    save_json_path: str | None = None,
    verbose: bool = True,
) -> list[dict]:
    """
    Esegue in parallelo una lista di configurazioni [(M, N), ...].
    """

    if max_workers is None:
        cpu = os.cpu_count() or 2
        max_workers = max(1, cpu - 1)

    jobs = [
        (M, N, params, timeout_sec, run_qaoa, qaoa_max_qubits, qaoa_reps, qaoa_maxiter)
        for (M, N) in configurations
    ]

    if verbose:
        print(f"[PARALLEL] jobs={len(jobs)} max_workers={max_workers}")

    results = []
    t0 = time.perf_counter()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_job = {executor.submit(_worker_run, job): job for job in jobs}

        try:
            iterator = as_completed(future_to_job, timeout=global_timeout_sec)
            for k, future in enumerate(iterator, start=1):
                job = future_to_job[future]
                M, N = job[0], job[1]

                try:
                    res = future.result()
                except Exception as e:
                    res = {
                        "M": M,
                        "N": N,
                        "worker_status": "future_error",
                        "worker_error": str(e),
                    }

                results.append(res)

                if verbose:
                    status = res.get("worker_status", "unknown")
                    ctime = res.get("classical_time_sec")
                    qtime = res.get("qaoa_time_sec")
                    print(
                        f"[DONE {k}/{len(jobs)}] "
                        f"(M={M}, N={N}) status={status} "
                        f"classical={ctime}s qaoa={qtime}s"
                    )

        except TimeoutError:
            if verbose:
                print("[PARALLEL] Global timeout reached while waiting for futures.")

            for future, job in future_to_job.items():
                if not future.done():
                    future.cancel()

    elapsed = time.perf_counter() - t0

    results.sort(key=lambda x: (x.get("M", math.inf), x.get("N", math.inf)))

    if verbose:
        print(f"[PARALLEL] total_wall_time_sec={elapsed:.4f}")

    if save_csv_path:
        df = pd.DataFrame(results)
        Path(save_csv_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_csv_path, index=False)

    if save_json_path:
        Path(save_json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    return results


def run_configurations_serial(
    configurations: list[tuple[int, int]],
    params: dict,
    timeout_sec: int = 120,
    run_qaoa: bool = True,
    qaoa_max_qubits: int | None = 20,
    qaoa_reps: int = 1,
    qaoa_maxiter: int = 100,
    save_csv_path: str | None = None,
    save_json_path: str | None = None,
    verbose: bool = True,
) -> list[dict]:
    """
    Versione seriale per benchmark e debug.
    """
    results = []
    t0 = time.perf_counter()

    for i, (M, N) in enumerate(configurations, start=1):
        if verbose:
            print(f"[SERIAL {i}/{len(configurations)}] M={M}, N={N}")

        res = _worker_run((M, N, params, timeout_sec, run_qaoa, qaoa_max_qubits, qaoa_reps, qaoa_maxiter))
        results.append(res)

    elapsed = time.perf_counter() - t0
    results.sort(key=lambda x: (x.get("M", math.inf), x.get("N", math.inf)))

    if verbose:
        print(f"[SERIAL] total_wall_time_sec={elapsed:.4f}")

    if save_csv_path:
        df = pd.DataFrame(results)
        Path(save_csv_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_csv_path, index=False)

    if save_json_path:
        Path(save_json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    return results


def summarize_results(results: list[dict]) -> dict:
    """
    Piccolo riassunto numerico per confrontare i batch.
    """
    if not results:
        return {}

    total = len(results)
    ok = sum(r.get("worker_status") == "ok" for r in results)
    err = sum(r.get("worker_status") != "ok" for r in results)
    qaoa_ok = sum((r.get("qaoa_obj") is not None) for r in results)
    qaoa_err = sum(bool(r.get("qaoa_error")) for r in results)
    qaoa_skipped = sum(bool(r.get("qaoa_skipped")) for r in results)

    classical_times = [r["classical_time_sec"] for r in results if r.get("classical_time_sec") is not None]
    qaoa_times = [r["qaoa_time_sec"] for r in results if r.get("qaoa_time_sec") is not None]
    wall_times = [r["worker_wall_time_sec"] for r in results if r.get("worker_wall_time_sec") is not None]

    return {
        "total_jobs": total,
        "worker_ok": ok,
        "worker_error": err,
        "qaoa_ok": qaoa_ok,
        "qaoa_error": qaoa_err,
        "qaoa_skipped": qaoa_skipped,
        "avg_classical_time_sec": round(sum(classical_times) / len(classical_times), 4) if classical_times else None,
        "avg_qaoa_time_sec": round(sum(qaoa_times) / len(qaoa_times), 4) if qaoa_times else None,
        "avg_worker_wall_time_sec": round(sum(wall_times) / len(wall_times), 4) if wall_times else None,
    }


if __name__ == "__main__":
    # ESEMPIO CONFIGURAZIONI
    configurations = [
        (2, 3),
        (3, 3),
        (3, 4),
        (4, 4),
        (4, 5),
        (5, 5),
    ]

    # SOSTITUISCI CON I TUOI PARAMETRI
    params = {
        # ...
    }

    # Benchmark seriale
    serial_results = run_configurations_serial(
        configurations=configurations,
        params=params,
        timeout_sec=120,
        run_qaoa=True,
        qaoa_max_qubits=20,
        qaoa_reps=1,
        qaoa_maxiter=80,
        save_csv_path="results/scaling_serial.csv",
        save_json_path="results/scaling_serial.json",
        verbose=True,
    )
    print("[SERIAL SUMMARY]", summarize_results(serial_results))

    # Benchmark parallelo
    parallel_results = run_configurations_parallel(
        configurations=configurations,
        params=params,
        timeout_sec=120,
        max_workers=max(1, (os.cpu_count() or 2) - 1),
        run_qaoa=True,
        qaoa_max_qubits=20,
        qaoa_reps=1,
        qaoa_maxiter=80,
        global_timeout_sec=None,
        save_csv_path="results/scaling_parallel.csv",
        save_json_path="results/scaling_parallel.json",
        verbose=True,
    )
    print("[PARALLEL SUMMARY]", summarize_results(parallel_results))


