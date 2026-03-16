"""
performance_utils.py
====================
Utility per migliorare le performance: caching delle conversioni QP->QUBO
e wrapper opzionale per spostare matrici sparse su GPU (se CuPy disponibile).

Nota: il caching qui è semplice e serve a evitare conversioni ripetute
dello stesso `QuadraticProgram` durante esecuzioni ripetute nello stesso
processo (es. scaling loop). La chiave è basata su nome e dimensioni.
"""
from __future__ import annotations

from typing import Any, Tuple

cache: dict[Tuple[str, int, int], Any] = {}


def _make_qp_key(qp) -> Tuple[str, int, int]:
    try:
        name = getattr(qp, "name", "")
        nvars = qp.get_num_vars()
        nconstr = qp.get_num_linear_constraints()
        return (name, int(nvars), int(nconstr))
    except Exception:
        return (str(id(qp)), 0, 0)


def convert_qubo_cached(qp):
    """Converti `qp` in QUBO usando QuadraticProgramToQubo e cache il risultato.

    Ritorna l'oggetto QUBO restituito dal converter (compatibile con qiskit-optimization).
    """
    from qiskit_optimization.converters import QuadraticProgramToQubo

    key = _make_qp_key(qp)
    if key in cache:
        return cache[key]

    converter = QuadraticProgramToQubo()
    qubo = converter.convert(qp)
    cache[key] = qubo
    return qubo


def clear_qubo_cache() -> None:
    """Svuota la cache (utile per test/unit)."""
    cache.clear()


# Optional GPU helpers (best-effort)
try:
    import cupy as cp
    import cupyx.scipy.sparse as cusparse

    def scipy_to_cupy(scipy_mat):
        """Converti una matrice scipy.sparse in cupyx.sparse CSR, se possibile."""
        try:
            return cusparse.csr_matrix(scipy_mat)
        except Exception:
            return scipy_mat

except Exception:
    cp = None

    def scipy_to_cupy(scipy_mat):
        return scipy_mat
