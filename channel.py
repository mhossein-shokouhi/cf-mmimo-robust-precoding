"""Channel model and LMMSE estimator.

All equations correspond to Section II of `ORAN.tex`. We adopt the uncorrelated
Rayleigh block-fading model `h_{k,l} = sqrt(beta_{k,l}) g_{k,l}` with
`g_{k,l} ~ CN(0, I_{N_t})`. The 3GPP TR 36.814 UMi-NLOS path loss is used for
beta_{k,l}.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from config import SimConfig


@dataclass
class Topology:
    """Deployment geometry + large-scale fading."""

    user_pos: np.ndarray
    oru_pos: np.ndarray
    beta: np.ndarray
    serving_oru: list
    users_of_oru: list

    @property
    def K(self) -> int:
        return self.beta.shape[0]

    @property
    def L(self) -> int:
        return self.beta.shape[1]


def _pathloss_db_umi_nlos(distance_3d_m: np.ndarray, f_c_ghz: float) -> np.ndarray:
    """3GPP TR 36.814 UMi-NLOS path-loss model (distance in metres)."""
    d = np.maximum(distance_3d_m, 1.0)
    return 36.7 * np.log10(d) + 22.7 + 26.0 * np.log10(f_c_ghz)


def deploy(cfg: SimConfig, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Drop users and O-RUs uniformly inside the rectangular coverage area."""
    user_xy = rng.uniform(low=[0.0, 0.0], high=[cfg.area_x, cfg.area_y], size=(cfg.K, 2))
    oru_xy = rng.uniform(low=[0.0, 0.0], high=[cfg.area_x, cfg.area_y], size=(cfg.L, 2))
    user_pos = np.column_stack([user_xy, np.full(cfg.K, cfg.user_height)])
    oru_pos = np.column_stack([oru_xy, np.full(cfg.L, cfg.oru_height)])
    return user_pos, oru_pos


def large_scale_fading(user_pos: np.ndarray, oru_pos: np.ndarray, cfg: SimConfig) -> np.ndarray:
    """Compute the K x L matrix of `beta_{k,l}` in linear scale."""
    diff = user_pos[:, None, :] - oru_pos[None, :, :]
    d3d = np.sqrt(np.sum(diff * diff, axis=-1))
    d2d = np.sqrt(np.sum(diff[..., :2] * diff[..., :2], axis=-1))
    d3d = np.where(d2d < cfg.min_distance_2d,
                   np.sqrt(cfg.min_distance_2d ** 2 + (user_pos[:, None, 2] - oru_pos[None, :, 2]) ** 2),
                   d3d)
    pl_db = _pathloss_db_umi_nlos(d3d, cfg.f_c_ghz)
    return 10.0 ** (-pl_db / 10.0)


def user_centric_clusters(beta: np.ndarray, L_max: int):
    """Each user is served by its strongest `L_max` O-RUs (largest beta)."""
    K, L = beta.shape
    L_max_eff = min(L_max, L)
    serving_oru = [list(np.argsort(-beta[k])[:L_max_eff]) for k in range(K)]
    users_of_oru = [[] for _ in range(L)]
    for k, ell_list in enumerate(serving_oru):
        for ell in ell_list:
            users_of_oru[ell].append(k)
    return serving_oru, users_of_oru


def build_topology(cfg: SimConfig, rng: np.random.Generator) -> Topology:
    user_pos, oru_pos = deploy(cfg, rng)
    beta = large_scale_fading(user_pos, oru_pos, cfg)
    serving_oru, users_of_oru = user_centric_clusters(beta, cfg.L_max)
    return Topology(user_pos=user_pos,
                    oru_pos=oru_pos,
                    beta=beta,
                    serving_oru=serving_oru,
                    users_of_oru=users_of_oru)


def sample_channel(topology: Topology, N_t: int, rng: np.random.Generator) -> np.ndarray:
    """Return the true channel tensor `h[k, l, :] in C^{N_t}` for one RT loop."""
    K, L = topology.beta.shape
    g = (rng.standard_normal((K, L, N_t)) + 1j * rng.standard_normal((K, L, N_t))) / np.sqrt(2.0)
    scale = np.sqrt(topology.beta)[:, :, None]
    return scale * g


def estimation_stats(beta: np.ndarray,
                     pilot_idx: np.ndarray,
                     p_ul: float,
                     sigma2_ul: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pre-compute LMMSE scalar coefficients and error variances.

    Returns
    -------
    alpha : (K, L) array, the `alpha_{k,l}` of eq. (after (7)) of ORAN.tex.
    err_var : (K, L) array, error variance `beta_{k,l} - alpha_{k,l}` which
              equals `tr(R_{k,l}) / N_t`.
    lmmse_coef : (K, L) array, the scalar that multiplies `y_{k,l}` in the
                 LMMSE estimator of (3).
    """
    K, L = beta.shape
    pilot_idx = np.asarray(pilot_idx, dtype=int)
    denom = np.zeros((L, int(pilot_idx.max()) + 1))
    for k in range(K):
        denom[:, pilot_idx[k]] += p_ul * beta[k]
    denom += sigma2_ul
    per_user_denom = denom[np.arange(L)[None, :], pilot_idx[:, None]]
    lmmse_coef = (np.sqrt(p_ul) * beta) / per_user_denom
    alpha = (p_ul * beta * beta) / per_user_denom
    err_var = beta - alpha
    return alpha, err_var, lmmse_coef


def estimate_channels(h_true: np.ndarray,
                      pilot_idx: np.ndarray,
                      beta: np.ndarray,
                      p_ul: float,
                      sigma2_ul: float,
                      lmmse_coef: np.ndarray,
                      rng: np.random.Generator) -> np.ndarray:
    """Return LMMSE channel estimates `h_hat[k, l, :]`.

    The per-pilot received observation at O-RU l is
    `y_{p,l} = sum_{k: pilot(k)=p} sqrt(p_ul) h_{k,l} + n_{p,l}` where
    `n_{p,l} ~ CN(0, sigma2_ul I)`. `h_hat_{k,l}` is obtained by scaling
    the corresponding `y_{pilot(k), l}` by `lmmse_coef[k, l]`.
    """
    K, L, N_t = h_true.shape
    pilot_idx = np.asarray(pilot_idx, dtype=int)
    tau_p = int(pilot_idx.max()) + 1

    noise = (rng.standard_normal((tau_p, L, N_t)) + 1j * rng.standard_normal((tau_p, L, N_t)))
    noise *= np.sqrt(sigma2_ul / 2.0)

    y = noise.copy()
    sqrt_p = np.sqrt(p_ul)
    for k in range(K):
        y[pilot_idx[k]] += sqrt_p * h_true[k]

    h_hat = np.empty_like(h_true)
    for k in range(K):
        h_hat[k] = lmmse_coef[k][:, None] * y[pilot_idx[k]]
    return h_hat


def pilot_conflict_matrix(beta: np.ndarray,
                          serving_oru: list,
                          users_of_oru: list) -> np.ndarray:
    """Return `chi[k, i] = sum_{l in L_k cap L_i} beta_{k,l} beta_{i,l}` (K, K)."""
    K, L = beta.shape
    chi = np.zeros((K, K))
    for ell in range(L):
        u = users_of_oru[ell]
        if len(u) < 2:
            continue
        b = beta[u, ell]
        chi[np.ix_(u, u)] += np.outer(b, b)
    np.fill_diagonal(chi, 0.0)
    return chi
