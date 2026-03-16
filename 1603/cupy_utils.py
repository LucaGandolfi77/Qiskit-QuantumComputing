"""
cupy_utils.py
==============
Semplice wrapper per usare CuPy quando disponibile, altrimenti NumPy.

Esporta:
 - xp: il modulo array (cupy oppure numpy)
 - asarray(x): crea array su device (cupy array se disponibile)
 - asnumpy(x): converte qualsiasi array in numpy.ndarray
 - scipy_to_gpu(scipy_sparse): prova a convertire una scipy.sparse in cupyx.sparse

Questo modulo è progettato per essere una drop-in light-weight; le
conversioni sparse sono best-effort e richiedono `cupy` e `cupyx` installati.
"""
from __future__ import annotations

try:
    import cupy as xp
    import cupyx.scipy.sparse as cusparse
    _HAS_CUPY = True
except Exception:
    import numpy as xp  # type: ignore
    cusparse = None
    _HAS_CUPY = False

import numpy as _np
from typing import Any


def asarray(x: Any):
    """Return an array on the current array module (cupy or numpy)."""
    if _HAS_CUPY:
        try:
            return xp.asarray(x)
        except Exception:
            # fallback to numpy then transfer
            return xp.asarray(_np.asarray(x))
    else:
        return _np.asarray(x)


def asnumpy(x: Any):
    """Return a numpy.ndarray for any input array-like (cupy or numpy)."""
    if _HAS_CUPY:
        try:
            if isinstance(x, xp.ndarray):
                return xp.asnumpy(x)
        except Exception:
            pass
    return _np.asarray(x)


def scipy_to_gpu(scipy_mat):
    """Try to convert a scipy.sparse matrix to a cupyx sparse matrix.

    If conversion fails or CuPy not available, returns the original matrix.
    """
    if not _HAS_CUPY or cusparse is None:
        return scipy_mat
    try:
        return cusparse.csr_matrix(scipy_mat)
    except Exception:
        return scipy_mat


def has_cupy() -> bool:
    return _HAS_CUPY
