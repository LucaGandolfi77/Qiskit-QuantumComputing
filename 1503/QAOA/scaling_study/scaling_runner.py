"""
scaling_runner.py
=================
Entry point dello scaling study. Itera tutte le configurazioni (M, N),
esegue il solver ADMM classico + QAOA per ciascuna e salva i risultati
in scaling_results.json.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime

# Aggiungi directory corrente e QAOA al path
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

from config_generator import generate_all_configs, print_config_summary
from single_run import run_configuration


def _progress_bar(current: int, total: int, width: int = 30, label: str = "") -> str:
    pct = current / total
    filled = int(width * pct)
    bar = "█" * filled + "░" * (width - filled)
    return f"  [{bar}] {current}/{total} ({pct * 100:.0f}%) — {label}"


def main() -> None:
    print("╔" + "═" * 68 + "╗")
    print("║         SCALING STUDY — QAOA VM Allocation                       ║")
    print("╚" + "═" * 68 + "╝")

    configs = generate_all_configs()
    print_config_summary(configs)

    total = len(configs)
    results: list[dict] = []
    n_qaoa_ok = 0

    t_start = time.perf_counter()

    for idx, cfg in enumerate(configs, start=1):
        M, N = cfg["M"], cfg["N"]
        label = f"Running M={M}, N={N} ..."
        print(_progress_bar(idx, total, label=label), flush=True)

        try:
            r = run_configuration(M, N, cfg, timeout_sec=120)
        except Exception as e:
            # Fallback: salva un entry con errore completo
            r = {
                "M": M, "N": N,
                "n_binary_vars_original": M + M * N,
                "n_slack_vars": 0, "n_qubo_vars": 0, "n_qubits": 0,
                "qubo_sparsity": 0.0, "continuous_vars": M, "feasible": False,
                "classical_obj": 0.0, "classical_iter": 0,
                "classical_converged": False, "classical_time_sec": 0.0,
                "classical_residuals": [], "classical_energy_total": 0.0,
                "classical_servers_on": 0,
                "qaoa_obj": None, "qaoa_iter": None, "qaoa_converged": False,
                "qaoa_time_sec": None, "qaoa_residuals": None,
                "qaoa_energy_total": None, "qaoa_error": True,
                "qaoa_error_msg": str(e),
                "obj_diff_pct": None, "iter_ratio": None, "time_ratio": None,
            }
            print(f"    ⚠️  Config M={M}, N={N} fallita completamente: {e}")

        if not r.get("qaoa_error", True):
            n_qaoa_ok += 1
        results.append(r)

    elapsed = time.perf_counter() - t_start

    n_classical_ok = sum(1 for r in results if r["classical_iter"] > 0)

    # ── Salvataggio ──
    output = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "total_configs": total,
            "successful_classical": n_classical_ok,
            "successful_qaoa": n_qaoa_ok,
            "qiskit_optimization_version": "0.7.0",
            "max_M": max(c["M"] for c in configs),
            "max_N": max(c["N"] for c in configs),
            "total_time_sec": round(elapsed, 2),
        },
        "results": results,
    }

    out_path = os.path.join(HERE, "scaling_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"  RIEPILOGO SCALING STUDY")
    print(f"{'=' * 60}")
    print(f"  Tempo totale:            {elapsed:.1f} s")
    print(f"  Config classico OK:      {n_classical_ok}/{total}")
    print(f"  Config QAOA OK:          {n_qaoa_ok}/{total}")

    # Config con più qubits
    max_qubits_r = max(results, key=lambda r: r["n_qubits"])
    print(f"\n  🔬 QUBO più grande:      M={max_qubits_r['M']}, N={max_qubits_r['N']} "
          f"→ {max_qubits_r['n_qubits']} qubit")

    # Config con maggior deviazione QAOA
    valid_diff = [r for r in results if r["obj_diff_pct"] is not None]
    if valid_diff:
        worst_diff = max(valid_diff, key=lambda r: abs(r["obj_diff_pct"]))
        print(f"  📊 Max diff QAOA:        M={worst_diff['M']}, N={worst_diff['N']} "
              f"→ {worst_diff['obj_diff_pct']:.2f}%")

    # Config più lenta QAOA
    valid_time = [r for r in results if r["qaoa_time_sec"] is not None]
    if valid_time:
        slowest = max(valid_time, key=lambda r: r["qaoa_time_sec"])
        print(f"  ⏱️  QAOA più lento:      M={slowest['M']}, N={slowest['N']} "
              f"→ {slowest['qaoa_time_sec']:.2f} s")

    # Feasibility
    n_feasible = sum(1 for r in results if r["feasible"])
    print(f"  ✅ Soluzioni feasible:   {n_feasible}/{total}")

    print(f"\n  [JSON] Risultati salvati in: {out_path}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
