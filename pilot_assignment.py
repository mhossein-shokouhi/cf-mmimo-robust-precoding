"""Pilot assignment (PA) algorithms.

Three components are exposed here:

1. The greedy priority-score heuristic used to *initialise* the pilot
   assignment at the start of every non-RT loop (the **Proposed PA**).
2. A random-PA baseline.
3. Helpers for the per-near-RT-loop **DRL pilot refinement** described in
   Section III-B of `ORAN.tex`. They consist of:
     - the priority-score `argmax` that selects the target user `k_star`,
     - two observation builders (`proposed_observation` for the rich
       Dueling-DQN observation; `baseline_observation` for the naive
       K x tau_p one-hot pilot matrix used by Oh *et al.*'s simplified
       baseline), and
     - `apply_drl_action` which writes the new pilot back and refreshes
       the LMMSE statistics.

The DRL refinement layers cleanly on top of the heuristic: at each non-RT
loop we re-run the heuristic, then for `N_nRT` near-RT loops the DRL agent
nudges one pilot at a time. This matches the BCD decomposition of the
paper.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

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


# ---------------------------------------------------------------------------
#  DRL refinement helpers (Section III-B of ORAN.tex).
# ---------------------------------------------------------------------------
def _csi_uncertainty_per_user(pilot_idx: np.ndarray,
                              beta: np.ndarray,
                              p_ul: float,
                              sigma2_ul: float,
                              serving_oru: list) -> np.ndarray:
    """U_k = sum_{l in L_k} (beta_{k,l} - alpha_{k,l})  for every user k."""
    K, L = beta.shape
    tau_p_eff = int(pilot_idx.max()) + 1
    denom = np.zeros((L, tau_p_eff))
    for j in range(K):
        denom[:, pilot_idx[j]] += p_ul * beta[j]
    denom += sigma2_ul
    rows = np.arange(L)
    U = np.zeros(K)
    for k in range(K):
        alpha_kl = (p_ul * beta[k] * beta[k]) / denom[rows, pilot_idx[k]]
        err = beta[k] - alpha_kl
        U[k] = err[serving_oru[k]].sum()
    return U


def _per_pilot_conflict_scores(k: int,
                               pilot_idx: np.ndarray,
                               chi: np.ndarray,
                               tau_p: int) -> np.ndarray:
    """c_{k,t} = sum_{i: psi_i = phi_t, i != k} chi_{k,i}, for t = 1..tau_p."""
    out = np.zeros(tau_p)
    K = pilot_idx.size
    for i in range(K):
        if i == k:
            continue
        out[pilot_idx[i]] += chi[k, i]
    return out


def _pilot_occupancy(pilot_idx: np.ndarray, tau_p: int) -> np.ndarray:
    return np.bincount(pilot_idx, minlength=tau_p).astype(float)


def select_priority_user(beta: np.ndarray,
                         pilot_idx: np.ndarray,
                         serving_oru: list,
                         users_of_oru: list,
                         cfg: SimConfig) -> Tuple[int, np.ndarray]:
    """Return (k_star, rho) where rho is the per-user priority score.

    Implements the priority score `rho_k = w_C C_k + w_U U_k` from eq. (15)
    of `ORAN.tex` (the QoS-deficit term is omitted because mu_k = 0
    throughout the simulator).
    """
    K = beta.shape[0]
    chi = pilot_conflict_matrix(beta, serving_oru, users_of_oru)

    # C_k: pilot conflict score (sum of chi over pilot-mates).
    C = np.zeros(K)
    for k in range(K):
        same = np.where(pilot_idx == pilot_idx[k])[0]
        same = same[same != k]
        C[k] = chi[k, same].sum()
    U = _csi_uncertainty_per_user(pilot_idx, beta, cfg.p_ul, cfg.sigma2, serving_oru)

    chi_scale = chi.mean() + 1e-30
    u_scale = beta.mean() * cfg.L_max + 1e-30
    rho = cfg.priority_w_C * (C / chi_scale) + cfg.priority_w_U * (U / u_scale)
    k_star = int(np.argmax(rho))
    return k_star, rho


# ---------- observation builders -------------------------------------------
def proposed_obs_dim(tau_p: int, M: int) -> int:
    """Dimension of the rich observation passed to the proposed Dueling DQN."""
    # one-hot pilot of k_star (tau_p) + scalar rate + scalar QoS deficit
    # + scalar U_{k_star} + per-pilot conflict (tau_p) + occupancy (tau_p)
    # + M neighbours each: pilot one-hot (tau_p) + chi (1) + QoS deficit (1)
    return tau_p + 3 + tau_p + tau_p + M * (tau_p + 2)


def naive_obs_dim(K: int, tau_p: int) -> int:
    """Naive observation = vectorised K x tau_p one-hot pilot matrix.

    This is the literal stand-in for the Oh *et al.* (JSAC 2024) state.
    The agent does **not** see any channel statistics: no rate, no
    path-loss, no CSI uncertainty, no conflict scores. Identifying the
    user being updated is folded into the **action space**: the agent
    outputs a single index in `[0, K * tau_p)` which decodes to a
    `(user, pilot)` pair. There is no separate priority-based user
    selector — that is precisely the architectural difference from the
    proposed scheme.
    """
    return K * tau_p


def naive_action_space(K: int, tau_p: int) -> int:
    """The naive DRL baseline picks (user, pilot) jointly."""
    return K * tau_p


def proposed_observation(k_star: int,
                         pilot_idx: np.ndarray,
                         beta: np.ndarray,
                         serving_oru: list,
                         users_of_oru: list,
                         per_user_rate: np.ndarray,
                         qos_deficit: np.ndarray,
                         tau_p: int,
                         M: int,
                         cfg: SimConfig) -> np.ndarray:
    """Rich Dueling-DQN observation for user `k_star`.

    Includes the components that are *physically tied to the channel
    statistics* — rates, CSI uncertainty, conflict scores, neighbour pilot
    indices — so the policy can adapt when users move and the
    large-scale fading shifts.
    """
    K = pilot_idx.size
    chi = pilot_conflict_matrix(beta, serving_oru, users_of_oru)
    U = _csi_uncertainty_per_user(pilot_idx, beta, cfg.p_ul, cfg.sigma2, serving_oru)
    cks = _per_pilot_conflict_scores(k_star, pilot_idx, chi, tau_p)
    occ = _pilot_occupancy(pilot_idx, tau_p)

    chi_scale = chi.mean() + 1e-30
    u_scale = beta.mean() * cfg.L_max + 1e-30
    rate_scale = max(per_user_rate.mean() + 1e-12, 1e-3)

    pilot_oh = np.zeros(tau_p)
    pilot_oh[pilot_idx[k_star]] = 1.0

    feats = [pilot_oh,
             np.array([per_user_rate[k_star] / rate_scale]),
             np.array([qos_deficit[k_star] / rate_scale]),
             np.array([U[k_star] / u_scale]),
             cks / chi_scale,
             occ / max(K, 1)]

    chi_row = chi[k_star].copy()
    chi_row[k_star] = -np.inf
    nbr = np.argsort(-chi_row)[:M]
    for m in nbr:
        nbr_oh = np.zeros(tau_p)
        nbr_oh[pilot_idx[int(m)]] = 1.0
        feats.append(nbr_oh)
        feats.append(np.array([chi[k_star, int(m)] / chi_scale]))
        feats.append(np.array([qos_deficit[int(m)] / rate_scale]))

    return np.concatenate(feats).astype(np.float32)


def naive_observation(pilot_idx: np.ndarray, tau_p: int) -> np.ndarray:
    """Naive observation = K x tau_p one-hot pilot matrix, flattened.

    Faithful to the Oh *et al.* state matrix: the agent only sees who
    holds which pilot. It does **not** know channel quality, who moved,
    or which user is currently in trouble.
    """
    K = pilot_idx.size
    M = np.zeros((K, tau_p), dtype=np.float32)
    M[np.arange(K), pilot_idx] = 1.0
    return M.ravel()


def decode_naive_action(action: int, tau_p: int) -> Tuple[int, int]:
    """Decode the joint `(user, pilot)` action used by the naive baseline."""
    k = int(action) // int(tau_p)
    t = int(action) % int(tau_p)
    return k, t


def apply_drl_action(pilot_idx: np.ndarray,
                     user: int,
                     new_pilot: int) -> np.ndarray:
    out = pilot_idx.copy()
    out[int(user)] = int(new_pilot)
    return out
