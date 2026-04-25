"""Rate evaluation using the true channel realisation.

The precoders are designed from the LMMSE estimates, but their actual
downlink throughput must be measured with the true channels via
`xi_{k,i} = sum_{l in L_i} h_{k,l}^H v_{i,l}` (eq. (9) in the paper).
"""

from __future__ import annotations

import numpy as np


def compute_rates(h_true: np.ndarray,
                  v: np.ndarray,
                  sigma2_dl: float,
                  tau_d: int,
                  tau_c: int) -> np.ndarray:
    """Return per-user downlink rates (bits/s/Hz) as `(tau_d/tau_c) log2(1+gamma)`."""
    xi = np.einsum("kln,iln->ki", h_true.conj(), v)
    sig = np.abs(np.diag(xi)) ** 2
    total = np.sum(np.abs(xi) ** 2, axis=1)
    interf = np.clip(total - sig, 0.0, None)
    sinr = sig / (interf + sigma2_dl)
    return (tau_d / tau_c) * np.log2(1.0 + sinr)


def aggregate_throughput(rates: np.ndarray) -> float:
    return float(np.sum(rates))
