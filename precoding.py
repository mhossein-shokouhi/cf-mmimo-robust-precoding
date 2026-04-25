"""Precoding algorithms.

We implement four schemes:

1. `robust_wmmse`: the proposed robust WMMSE algorithm derived in
   Section III.A of `ORAN.tex`.  It keeps the channel-estimation error
   covariance terms (`R_{k,l} = err_var_{k,l} I`) inside the update
   equations of `u_k`, `w_k`, and `v_{k,l}` (eqs. after (11) and (13)).

2. `oblivious_wmmse`: an ablation / baseline where the `R_{k,l}` terms are
   dropped. Mathematically equivalent to feeding `err_var = 0` to the
   robust solver, but treated as a separate entry point for clarity.

3. `rzf`: Local Partial Regularized Zero-Forcing (LP-RZF), the standard
   cell-free baseline of Bjornson, Demir, and Sanguinetti. Each O-RU
   inverts its local Gram matrix with regularization `K_l sigma^2 / P_max`
   and allocates equal power per served user.

4. `mrt`: conjugate beamforming with equal power per user at each O-RU.

Per-O-RU power for the WMMSE schemes is enforced through the Lagrange
multiplier `lambda_l` obtained via an eigendecomposition + bisection
search, following the TWC companion paper (eqs. 26-28 there).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from config import SimConfig

# Accelerate BLAS on macOS may emit spurious divide/overflow warnings from
# matmul when the inputs span a very wide dynamic range (e.g. 1e-14 to 1).
# The downstream computations are numerically stable so we silence them.
np.seterr(divide="ignore", over="ignore", invalid="ignore")


def _initial_precoder(h_hat: np.ndarray,
                      users_of_oru: List[List[int]],
                      P_max: float) -> np.ndarray:
    """MRT initialisation that satisfies the per-O-RU power budget."""
    K, L, N_t = h_hat.shape
    v = np.zeros((K, L, N_t), dtype=complex)
    for ell in range(L):
        users = users_of_oru[ell]
        if not users:
            continue
        p_per_user = P_max / len(users)
        for k in users:
            g = h_hat[k, ell]
            norm = np.linalg.norm(g)
            if norm > 1e-14:
                v[k, ell] = np.sqrt(p_per_user) * g / norm
    return v


def _solve_v_oru(A_l: np.ndarray,
                 B_l: np.ndarray,
                 P_max: float,
                 cfg: SimConfig) -> Tuple[np.ndarray, float]:
    """Solve `V_l = (A_l + lambda_l I)^-1 B_l` with `|V_l|_F^2 <= P_max`.

    Uses the closed-form eigendecomposition approach of the TWC paper.
    Returns the optimal `V_l` and the `lambda_l` used.
    """
    N_t = A_l.shape[0]
    try:
        V0 = np.linalg.solve(A_l + 1e-12 * np.eye(N_t), B_l)
        power0 = float(np.sum(np.abs(V0) ** 2))
        if power0 <= P_max + 1e-10:
            return V0, 0.0
    except np.linalg.LinAlgError:
        pass

    eigvals, eigvecs = np.linalg.eigh((A_l + A_l.conj().T) / 2.0)
    eigvals = np.clip(eigvals.real, 0.0, None)

    Bt = eigvecs.conj().T @ B_l
    phi = np.sum(np.abs(Bt) ** 2, axis=1).real

    def power_at(lam: float) -> float:
        denom = eigvals + lam
        denom = np.where(denom > 1e-30, denom, 1e-30)
        return float(np.sum(phi / (denom * denom)))

    total_phi = float(np.sum(phi))
    if total_phi <= 1e-30:
        return np.zeros_like(B_l), 0.0

    lam_lo = 0.0
    lam_hi = max(np.sqrt(total_phi / P_max), 1.0)
    while power_at(lam_hi) > P_max:
        lam_hi *= 2.0
        if lam_hi > cfg.lambda_max:
            break

    for _ in range(cfg.lambda_bisect_iters):
        lam_mid = 0.5 * (lam_lo + lam_hi)
        if power_at(lam_mid) > P_max:
            lam_lo = lam_mid
        else:
            lam_hi = lam_mid

    lam_star = lam_hi
    D = Bt / (eigvals[:, None] + lam_star)
    V = eigvecs @ D
    return V, float(lam_star)


def _wmmse_outer(h_hat: np.ndarray,
                 err_var: np.ndarray,
                 users_of_oru: List[List[int]],
                 cfg: SimConfig,
                 eta: Optional[np.ndarray] = None) -> np.ndarray:
    """Shared WMMSE outer loop.

    The `err_var` argument controls whether the R-terms are included.
    """
    K, L, N_t = h_hat.shape
    if eta is None:
        eta = np.ones(K)
    sigma2 = cfg.sigma2
    P_max = cfg.P_max

    v = _initial_precoder(h_hat, users_of_oru, P_max)
    hat_xi = np.einsum("kln,iln->ki", h_hat.conj(), v)

    prev_obj = -np.inf
    for outer in range(cfg.wmmse_outer_iters):
        vnorm2 = np.sum(np.abs(v) ** 2, axis=2)
        p_l = np.sum(vnorm2, axis=0)
        # Elementwise form (more numerically stable than `@` for mixed dynamic range)
        r_term = np.sum(err_var * p_l[None, :], axis=1)

        sig_pow = np.abs(np.diag(hat_xi)) ** 2
        total_sq = np.sum(np.abs(hat_xi) ** 2, axis=1)
        D = total_sq + r_term + sigma2
        D = np.clip(D, 1e-30, None)
        u = np.diag(hat_xi) / D
        e = np.clip(1.0 - sig_pow / D, 1e-12, None)
        w = 1.0 / e
        alpha = eta * w * np.abs(u) ** 2

        sinr = sig_pow / np.clip(total_sq - sig_pow + r_term + sigma2, 1e-30, None)
        rate = np.log2(1.0 + sinr) * (cfg.tau_d / cfg.tau_c)
        obj = float(np.sum(eta * rate))
        if outer > 1 and (obj - prev_obj) < cfg.wmmse_tol * max(abs(prev_obj), 1.0):
            break
        prev_obj = obj

        for _ in range(cfg.bcd_sweeps_per_outer):
            for ell in range(L):
                users = users_of_oru[ell]
                if not users:
                    continue
                H_l = h_hat[:, ell, :]
                # A_l = sum_i alpha_i (h_hat_{i,l} h_hat_{i,l}^H + err_var_{i,l} I)
                A_l = (H_l.T * alpha) @ H_l.conj()
                A_l += np.eye(N_t) * float((alpha * err_var[:, ell]).sum())

                V_old = v[users, ell, :]
                hhv = H_l.conj() @ V_old.T
                z_mat = hat_xi[:, users] - hhv

                eu = (eta[users] * w[users] * u[users])
                term1 = h_hat[users, ell, :].T * eu
                # B_l[:, j] = sum_i alpha_i hat_h_{i,l} z_{i,k_j} (no conjugate
                # on z; verified by Wirtinger gradient of the WMMSE cost).
                term2 = H_l.T @ (alpha[:, None] * z_mat)
                B_l = term1 - term2

                V_new, _ = _solve_v_oru(A_l, B_l, P_max, cfg)

                dV = V_new.T - V_old
                v[users, ell, :] = V_new.T
                hat_xi[:, users] += H_l.conj() @ dV.T

    return v


def robust_wmmse(h_hat: np.ndarray,
                 err_var: np.ndarray,
                 users_of_oru: List[List[int]],
                 cfg: SimConfig,
                 eta: Optional[np.ndarray] = None) -> np.ndarray:
    """Proposed robust WMMSE (keeps R-terms in all updates)."""
    return _wmmse_outer(h_hat, err_var, users_of_oru, cfg, eta=eta)


def oblivious_wmmse(h_hat: np.ndarray,
                    err_var: np.ndarray,  # noqa: ARG001 - kept for a uniform signature
                    users_of_oru: List[List[int]],
                    cfg: SimConfig,
                    eta: Optional[np.ndarray] = None) -> np.ndarray:
    """Oblivious WMMSE baseline (treats estimates as perfect)."""
    zero_err = np.zeros_like(err_var)
    return _wmmse_outer(h_hat, zero_err, users_of_oru, cfg, eta=eta)


def mrt(h_hat: np.ndarray,
        err_var: np.ndarray,  # noqa: ARG001
        users_of_oru: List[List[int]],
        cfg: SimConfig,
        eta: Optional[np.ndarray] = None) -> np.ndarray:
    """Maximum-ratio transmission with equal power per served user."""
    del eta
    return _initial_precoder(h_hat, users_of_oru, cfg.P_max)


def rzf(h_hat: np.ndarray,
        err_var: np.ndarray,  # noqa: ARG001
        users_of_oru: List[List[int]],
        cfg: SimConfig,
        eta: Optional[np.ndarray] = None) -> np.ndarray:
    """Local Partial Regularized Zero-Forcing (LP-RZF) per O-RU.

    For O-RU `l` with serving set `K_l` of size `K_l`:

        H_l        = [hat_h_{k,l}]_{k in K_l}    in C^{N_t x K_l}
        D_l        = H_l H_l^H + (K_l sigma^2 / P_max) I_{N_t}
        v_hat_{k,l} = D_l^{-1} hat_h_{k,l}
        v_{k,l}    = sqrt(P_max / K_l) * v_hat_{k,l} / ||v_hat_{k,l}||

    The regularization `alpha_l = K_l sigma^2 / P_max` is the canonical
    cell-free LP-RZF choice (Bjornson, Demir, Sanguinetti 2021), obtained
    from the local MMSE solution under equal per-user data power
    `p_k = P_max / K_l`. The output is per-user power-normalized so that
    `sum_{k in K_l} ||v_{k,l}||^2 = P_max`, matching MRT's normalization.
    """
    del eta
    K, L, N_t = h_hat.shape
    P_max = cfg.P_max
    sigma2 = cfg.sigma2
    v = np.zeros_like(h_hat)
    for ell in range(L):
        users = users_of_oru[ell]
        if not users:
            continue
        K_l = len(users)
        H_l = h_hat[users, ell, :].T  # (N_t, K_l)
        alpha_l = K_l * sigma2 / P_max
        D = H_l @ H_l.conj().T + alpha_l * np.eye(N_t)
        try:
            W = np.linalg.solve(D, H_l)
        except np.linalg.LinAlgError:
            W = np.linalg.lstsq(D, H_l, rcond=None)[0]
        p_per_user = P_max / K_l
        for j, k in enumerate(users):
            w = W[:, j]
            n = np.linalg.norm(w)
            if n > 1e-14:
                v[k, ell, :] = np.sqrt(p_per_user) * w / n
    return v


PRECODERS = {
    "robust": robust_wmmse,
    "oblivious": oblivious_wmmse,
    "rzf": rzf,
    "mrt": mrt,
}
