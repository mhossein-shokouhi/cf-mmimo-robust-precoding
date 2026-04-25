"""Pilot assignment algorithms.

The paper uses a multi-agent DRL algorithm guided by the priority score in
eq. (15). Per the Checklist we replace that with a heuristic: we follow the
same priority-score structure but select the new pilot index greedily.
"""

from __future__ import annotations

import numpy as np

from channel import estimation_stats, pilot_conflict_matrix
from config import SimConfig


def random_assignment(K: int, tau_p: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(low=0, high=tau_p, size=K)


def _total_conflict(pilot_idx: np.ndarray, chi: np.ndarray) -> float:
    K = pilot_idx.size
    total = 0.0
    for t in np.unique(pilot_idx):
        mask = pilot_idx == t
        if mask.sum() < 2:
            continue
        total += chi[np.ix_(mask, mask)].sum() / 2.0
    return float(total)


def _user_conflict_score(k: int, pilot_idx: np.ndarray, chi: np.ndarray) -> float:
    same = np.where(pilot_idx == pilot_idx[k])[0]
    return float(chi[k, same].sum())


def _user_csi_uncertainty(k: int,
                          pilot_idx: np.ndarray,
                          beta: np.ndarray,
                          p_ul: float,
                          sigma2_ul: float,
                          serving_oru: list) -> float:
    tau_p_eff = int(pilot_idx.max()) + 1
    L = beta.shape[1]
    denom = np.zeros((L, tau_p_eff))
    for j in range(pilot_idx.size):
        denom[:, pilot_idx[j]] += p_ul * beta[j]
    denom += sigma2_ul
    alpha_kl = (p_ul * beta[k] * beta[k]) / denom[np.arange(L), pilot_idx[k]]
    err = beta[k] - alpha_kl
    return float(err[serving_oru[k]].sum())


def greedy_priority_assignment(beta: np.ndarray,
                               serving_oru: list,
                               users_of_oru: list,
                               tau_p: int,
                               cfg: SimConfig,
                               rng: np.random.Generator) -> np.ndarray:
    """Greedy heuristic inspired by the priority score in eq. (15) of ORAN.tex.

    Steps repeated `cfg.pilot_sweep_iters` times:
      1. Compute priority score `rho_k = w_C C_k + w_U U_k` for every user
         (the QoS-deficit term is zero because minimum-rate constraints are
         disabled in this simulation).
      2. Select `k_star = argmax rho_k`.
      3. For each candidate pilot t, evaluate the reassignment cost, defined as
         the resulting `rho_{k_star}`. Pick the `t` with the minimum cost.
      4. Update pilot of `k_star`.
    Early-stops when no user changes its pilot during a full sweep.
    """
    K, L = beta.shape
    chi = pilot_conflict_matrix(beta, serving_oru, users_of_oru)

    pilot_idx = np.arange(K) % tau_p
    rng.shuffle(pilot_idx)

    w_C = cfg.priority_w_C
    w_U = cfg.priority_w_U
    chi_scale = chi.mean() + 1e-30
    u_scale = beta.mean() * cfg.L_max + 1e-30

    last_total = _total_conflict(pilot_idx, chi)
    for _ in range(cfg.pilot_sweep_iters):
        rho = np.zeros(K)
        for k in range(K):
            c_k = _user_conflict_score(k, pilot_idx, chi) / chi_scale
            u_k = _user_csi_uncertainty(k, pilot_idx, beta, cfg.p_ul, cfg.sigma2,
                                         serving_oru) / u_scale
            rho[k] = w_C * c_k + w_U * u_k
        order = np.argsort(-rho)
        changed = False
        for k_star in order:
            best_pilot = int(pilot_idx[k_star])
            best_score = np.inf
            for t in range(tau_p):
                pilot_try = pilot_idx.copy()
                pilot_try[k_star] = t
                c_k = _user_conflict_score(k_star, pilot_try, chi) / chi_scale
                u_k = _user_csi_uncertainty(k_star, pilot_try, beta, cfg.p_ul,
                                             cfg.sigma2, serving_oru) / u_scale
                score = w_C * c_k + w_U * u_k
                if score < best_score - 1e-12:
                    best_score = score
                    best_pilot = t
            if best_pilot != pilot_idx[k_star]:
                pilot_idx[k_star] = best_pilot
                changed = True
        cur_total = _total_conflict(pilot_idx, chi)
        if not changed or abs(cur_total - last_total) < 1e-9:
            break
        last_total = cur_total
    return pilot_idx


def assign(scheme: str,
           beta: np.ndarray,
           serving_oru: list,
           users_of_oru: list,
           tau_p: int,
           cfg: SimConfig,
           rng: np.random.Generator) -> np.ndarray:
    """Dispatch wrapper so that the simulator can call one function."""
    if scheme == "random":
        return random_assignment(beta.shape[0], tau_p, rng)
    if scheme == "greedy":
        return greedy_priority_assignment(beta, serving_oru, users_of_oru, tau_p, cfg, rng)
    raise ValueError(f"Unknown pilot assignment scheme: {scheme}")
