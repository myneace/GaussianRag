"""Gaussian RAG initialization with Windows OpenBLAS thread safety."""
from __future__ import annotations

import functools
import threading
import numpy as np

_orig_eigh = np.linalg.eigh
_eigh_lock = threading.Lock()


@functools.wraps(_orig_eigh)
def _thread_safe_eigh(*args, **kwargs):
    with _eigh_lock:
        return _orig_eigh(*args, **kwargs)


np.linalg.eigh = _thread_safe_eigh

__all__: list[str] = []
