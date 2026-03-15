"""
config_generator.py
===================
Genera tutte le combinazioni (M, N) per lo scaling study e, per ciascuna,
un set di parametri realistici deterministici (seed = M*100 + N).
"""

from __future__ import annotations

import random

MAX_M = 4   # numero massimo di server
MAX_N = 5   # numero massimo di VM


def generate_all_configs(
    max_M: int = MAX_M,
    max_N: int = MAX_N,
) -> list[dict]:
    """
    Restituisce una lista di dict, uno per ogni coppia (M, N).

    Ogni dict contiene:
        M, N, P_idle, P_dynamic, C_capacity, u_cpu
    """
    configs: list[dict] = []

    for M in range(1, max_M + 1):
        for N in range(1, max_N + 1):
            rng = random.Random(M * 100 + N)

            P_idle = [rng.randint(80, 150) for _ in range(M)]
            P_dynamic = [rng.randint(40, 80) for _ in range(M)]
            C_capacity = [rng.randint(4, 10) for _ in range(M)]
            c_min = min(C_capacity)

            # Garantisci che ogni VM abbia carico ≤ c_min − 1 (almeno 1)
            u_max = max(c_min - 1, 1)
            u_cpu = [rng.randint(1, u_max) for _ in range(N)]

            configs.append({
                "M": M,
                "N": N,
                "P_idle": P_idle,
                "P_dynamic": P_dynamic,
                "C_capacity": C_capacity,
                "u_cpu": u_cpu,
            })

    return configs


def print_config_summary(configs: list[dict]) -> None:
    """Stampa un riepilogo delle configurazioni generate."""
    print(f"\n{'=' * 60}")
    print(f"  PIANO DI ESECUZIONE — Scaling Study")
    print(f"{'=' * 60}")
    print(f"  Configurazioni totali: {len(configs)}")
    print(f"  Range M (server):      1 → {max(c['M'] for c in configs)}")
    print(f"  Range N (VM):          1 → {max(c['N'] for c in configs)}")
    print(f"\n  {'M':>3} {'N':>3}  {'Bin vars':>9}  {'Cont':>5}  {'Tot orig':>9}  {'P_idle':>18}  {'u_cpu':>18}")
    print(f"  {'-' * 80}")
    for c in configs:
        n_bin = c["M"] + c["M"] * c["N"]
        n_cont = c["M"]
        print(
            f"  {c['M']:>3} {c['N']:>3}  {n_bin:>9}  {n_cont:>5}  {n_bin + n_cont:>9}"
            f"  {str(c['P_idle']):>18}  {str(c['u_cpu']):>18}"
        )
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    configs = generate_all_configs()
    print_config_summary(configs)
