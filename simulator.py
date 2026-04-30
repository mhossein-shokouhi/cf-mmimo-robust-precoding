"""One-scenario simulator.

Given a `SimConfig`, a pilot-assignment strategy name, and a precoder name,
evaluates the average aggregate downlink throughput over Monte-Carlo RT loops.

This module also implements the mobility-aware control loop used by the
DRL pilot-refinement experiments: the heuristic initialises the PA once per
non-RT loop, the DRL agent updates one user's pilot per near-RT loop, and
the precoder/channel evaluation runs every RT loop.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from channel import (Topology, build_topology, estimate_channels,
                     estimation_stats, sample_channel)
from config import SimConfig
from drl import DQNConfig, make_agent
from metrics import aggregate_throughput, compute_lower_bound_rates, compute_rates
from mobility import (kmh_to_mps, random_velocity_vectors, step_positions,
                      update_topology_after_motion)
from pilot_assignment import (apply_drl_action, assign, decode_naive_action,
                              naive_action_space, naive_obs_dim,
                              naive_observation, proposed_observation,
                              select_priority_user)
from precoding import PRECODERS, mrt, robust_wmmse


def _parse_scheme(name: str) -> Tuple[str, str]:
    pilot_part, precoder_part = name.split("+")
    return pilot_part, precoder_part


def evaluate_scheme(cfg: SimConfig,
                    scheme: str,
                    seed: int,
                    rt_loops: int) -> Dict[str, float]:
    """Evaluate one `{pilot}+{precoder}` scheme for one seed.

    Returns a dict with `throughput_mean`, `throughput_std`, and a couple of
    diagnostics that are useful for debugging.
    """
    pilot_scheme, precoder_name = _parse_scheme(scheme)
    if precoder_name not in PRECODERS:
        raise ValueError(f"Unknown precoder: {precoder_name}")
    precoder = PRECODERS[precoder_name]

    rng_topology = np.random.default_rng(seed)
    topology: Topology = build_topology(cfg, rng_topology)

    rng_pilot = np.random.default_rng(seed + 101)
    pilot_idx = assign(pilot_scheme,
                       topology.beta,
                       topology.serving_oru,
                       topology.users_of_oru,
                       cfg.tau_p,
                       cfg,
                       rng_pilot)

    alpha, err_var, lmmse_coef = estimation_stats(topology.beta,
                                                  pilot_idx,
                                                  cfg.p_ul,
                                                  cfg.sigma2)

    rng_channel = np.random.default_rng(seed + 5001)
    rng_noise = np.random.default_rng(seed + 9001)

    rates_history = np.zeros((rt_loops, cfg.K))
    per_loop = np.zeros(rt_loops)
    for t in range(rt_loops):
        h_true = sample_channel(topology, cfg.N_t, rng_channel)
        h_hat = estimate_channels(h_true,
                                  pilot_idx,
                                  topology.beta,
                                  cfg.p_ul,
                                  cfg.sigma2,
                                  lmmse_coef,
                                  rng_noise)
        v = precoder(h_hat, err_var, topology.users_of_oru, cfg)
        rates = compute_rates(h_true, v, cfg.sigma2, cfg.tau_d, cfg.tau_c)
        rates_history[t] = rates
        per_loop[t] = aggregate_throughput(rates)

    return {
        "throughput_mean": float(per_loop.mean()),
        "throughput_std": float(per_loop.std(ddof=1) if per_loop.size > 1 else 0.0),
        "pilot_entropy": float(-np.sum(np.bincount(pilot_idx, minlength=cfg.tau_p) / cfg.K
                                       * np.log2(np.clip(np.bincount(pilot_idx,
                                                                     minlength=cfg.tau_p) / cfg.K,
                                                         1e-12, 1.0)))),
        "avg_err_var_frac": float((err_var.sum() / max(topology.beta.sum(), 1e-30))),
        "rates_history": rates_history,
    }


def _min_rate_vector(min_rate: Union[float, np.ndarray], K: int) -> np.ndarray:
    """Return a length-K minimum-rate vector.

    The paper experiment uses a common target for every user, but accepting a
    vector keeps the dual update usable for future heterogeneous QoS tests.
    """
    r_min = np.asarray(min_rate, dtype=float)
    if r_min.ndim == 0:
        return np.full(K, float(r_min))
    if r_min.shape != (K,):
        raise ValueError(f"min_rate must be scalar or shape ({K},), got {r_min.shape}")
    return r_min.copy()


def _update_min_rate_duals(mu: np.ndarray,
                           rate_ema: Optional[np.ndarray],
                           r_min: np.ndarray,
                           rate_sample: np.ndarray,
                           cfg: SimConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Projected stochastic dual update for the ergodic min-rate constraints."""
    alpha = float(np.clip(cfg.min_rate_dual_ema_alpha, 0.0, 1.0))
    if rate_ema is None:
        new_ema = np.asarray(rate_sample, dtype=float).copy()
    else:
        new_ema = (1.0 - alpha) * rate_ema + alpha * rate_sample

    step = float(cfg.min_rate_dual_step)
    active = r_min > 0.0
    mu_next = mu.copy()
    mu_next[active] = np.clip(mu_next[active] + step * (r_min[active] - new_ema[active]),
                              0.0, float(cfg.min_rate_dual_max))
    return mu_next, new_ema


def evaluate_proposed_min_rate(cfg: SimConfig,
                               seed: int,
                               rt_loops: int,
                               min_rate: Union[float, np.ndarray]) -> Dict[str, object]:
    """Evaluate proposed PA + robust WMMSE with min-rate dual weights.

    Main simulations keep ``min_rate = 0`` and therefore ``mu = 0``. This
    entry point is used only by the proposed-only fairness CDF. For
    ``min_rate > 0``, a short independent warm-up estimates the dual weights
    before collecting the canonical per-user rate samples.
    """
    rng_topology = np.random.default_rng(seed)
    topology: Topology = build_topology(cfg, rng_topology)

    rng_pilot = np.random.default_rng(seed + 101)
    pilot_idx = assign("greedy",
                       topology.beta,
                       topology.serving_oru,
                       topology.users_of_oru,
                       cfg.tau_p,
                       cfg,
                       rng_pilot)

    alpha, err_var, lmmse_coef = estimation_stats(topology.beta,
                                                  pilot_idx,
                                                  cfg.p_ul,
                                                  cfg.sigma2)
    del alpha

    r_min = _min_rate_vector(min_rate, cfg.K)
    mu = np.zeros(cfg.K)
    rate_ema: Optional[np.ndarray] = None
    metric = str(cfg.min_rate_dual_metric).lower()
    if metric not in {"true", "lower_bound"}:
        raise ValueError("cfg.min_rate_dual_metric must be 'true' or 'lower_bound'")

    def one_rt_loop(rng_channel: np.random.Generator,
                    rng_noise: np.random.Generator,
                    update_dual: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        nonlocal mu, rate_ema
        h_true = sample_channel(topology, cfg.N_t, rng_channel)
        h_hat = estimate_channels(h_true,
                                  pilot_idx,
                                  topology.beta,
                                  cfg.p_ul,
                                  cfg.sigma2,
                                  lmmse_coef,
                                  rng_noise)
        v = robust_wmmse(h_hat, err_var, topology.users_of_oru, cfg, eta=1.0 + mu)
        lb_rates = compute_lower_bound_rates(h_hat, err_var, v,
                                             cfg.sigma2, cfg.tau_d, cfg.tau_c)
        rates = compute_rates(h_true, v, cfg.sigma2, cfg.tau_d, cfg.tau_c)
        if update_dual and np.any(r_min > 0.0):
            dual_sample = rates if metric == "true" else lb_rates
            mu, rate_ema = _update_min_rate_duals(mu, rate_ema, r_min,
                                                  dual_sample, cfg)
        return rates, lb_rates, mu.copy()

    warmup_loops = int(cfg.min_rate_warmup_rt_loops) if np.any(r_min > 0.0) else 0
    if warmup_loops > 0:
        rng_channel_warm = np.random.default_rng(seed + 15001)
        rng_noise_warm = np.random.default_rng(seed + 19001)
        for _ in range(warmup_loops):
            one_rt_loop(rng_channel_warm, rng_noise_warm, update_dual=True)

    # Keep the reported samples on the same canonical channel/noise streams as
    # the zero-min-rate CDF; warm-up uses independent streams above.
    rng_channel = np.random.default_rng(seed + 5001)
    rng_noise = np.random.default_rng(seed + 9001)
    rates_history = np.zeros((rt_loops, cfg.K))
    lb_history = np.zeros((rt_loops, cfg.K))
    mu_history = np.zeros((rt_loops + 1, cfg.K))
    mu_history[0] = mu
    for t in range(rt_loops):
        rates, lb_rates, mu_t = one_rt_loop(rng_channel, rng_noise, update_dual=True)
        rates_history[t] = rates
        lb_history[t] = lb_rates
        mu_history[t + 1] = mu_t

    per_loop = rates_history.sum(axis=1)
    return {
        "throughput_mean": float(per_loop.mean()),
        "throughput_std": float(per_loop.std(ddof=1) if per_loop.size > 1 else 0.0),
        "rates_history": rates_history,
        "lower_bound_rates_history": lb_history,
        "mu_history": mu_history,
        "final_mu": mu.copy(),
        "min_rate": r_min,
    }


def _evaluate_min_rate_worker(args):
    """Worker entry point for one (min_rate, seed) proposed-CDF evaluation."""
    cfg, min_rate, seed, rt_loops = args
    return evaluate_proposed_min_rate(cfg, int(seed), int(rt_loops), float(min_rate))


def evaluate_min_rate_cdf(cfg: SimConfig,
                          min_rates: List[float],
                          seeds: List[int],
                          rt_loops: int,
                          progress: bool = True,
                          n_workers: int = 1) -> Dict[str, np.ndarray]:
    """Evaluate proposed-only per-user CDF samples for several R_min values."""
    min_rates = [float(x) for x in min_rates]
    n_m, n_seeds, K = len(min_rates), len(seeds), cfg.K
    thr = np.zeros((n_m, n_seeds))
    rates = np.zeros((n_m, n_seeds, rt_loops, K))
    lb_rates = np.zeros((n_m, n_seeds, rt_loops, K))
    final_mu = np.zeros((n_m, n_seeds, K))
    mu_history = np.zeros((n_m, n_seeds, rt_loops + 1, K))

    jobs = [(cfg, r_min, seed, rt_loops) for r_min in min_rates for seed in seeds]
    out: Dict[Tuple[int, int], Dict[str, object]] = {}

    if n_workers <= 1:
        iterator = range(len(jobs))
        if progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, desc="min-rate CDF")
            except ImportError:
                pass
        for j in iterator:
            res = _evaluate_min_rate_worker(jobs[j])
            ri, si = divmod(j, n_seeds)
            out[(ri, si)] = res
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(_evaluate_min_rate_worker, jobs[j]): j
                    for j in range(len(jobs))}
            if progress:
                try:
                    from tqdm import tqdm
                    pbar = tqdm(total=len(jobs), desc="min-rate CDF")
                except ImportError:
                    pbar = None
            else:
                pbar = None
            for fut in as_completed(futs):
                j = futs[fut]
                res = fut.result()
                ri, si = divmod(j, n_seeds)
                out[(ri, si)] = res
                if pbar is not None:
                    pbar.update(1)
            if pbar is not None:
                pbar.close()

    for (ri, si), res in out.items():
        thr[ri, si] = res["throughput_mean"]
        rates[ri, si] = res["rates_history"]
        lb_rates[ri, si] = res["lower_bound_rates_history"]
        final_mu[ri, si] = res["final_mu"]
        mu_history[ri, si] = res["mu_history"]

    return {
        "throughput": thr,
        "rates": rates,
        "lower_bound_rates": lb_rates,
        "final_mu": final_mu,
        "mu_history": mu_history,
    }


def _run_eval_jobs(cfg: SimConfig,
                   schemes: List[str],
                   seeds: List[int],
                   rt_loops: int,
                   models_dir: Optional[str],
                   n_workers: int,
                   progress: bool):
    """Run the cartesian (scheme, seed) evaluation grid, optionally in
    parallel. Returns a dict mapping `(scheme_idx, seed_idx)` to the
    per-evaluation result dict.
    """
    jobs = [(cfg, scheme, seed, rt_loops, models_dir)
            for scheme in schemes for seed in seeds]
    n_s, n_seeds = len(schemes), len(seeds)
    out: Dict[Tuple[int, int], Dict[str, object]] = {}

    if n_workers <= 1:
        iterator = range(len(jobs))
        if progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, desc=f"eval (tau_p={cfg.tau_p}, "
                                                f"K={cfg.K}, L={cfg.L})")
            except ImportError:
                pass
        for k in iterator:
            res = _evaluate_one_worker(jobs[k])
            si, ti = divmod(k, n_seeds)
            out[(si, ti)] = res
        return out

    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(_evaluate_one_worker, jobs[k]): k for k in range(len(jobs))}
        if progress:
            try:
                from tqdm import tqdm
                pbar = tqdm(total=len(jobs), desc=f"eval (tau_p={cfg.tau_p}, "
                                                  f"K={cfg.K}, L={cfg.L})")
            except ImportError:
                pbar = None
        else:
            pbar = None
        for fut in as_completed(futs):
            k = futs[fut]
            res = fut.result()
            si, ti = divmod(k, n_seeds)
            out[(si, ti)] = res
            if pbar is not None:
                pbar.update(1)
        if pbar is not None:
            pbar.close()
    return out


def evaluate_all(cfg: SimConfig,
                 schemes: List[str],
                 seeds: List[int],
                 rt_loops: int,
                 progress: bool = True,
                 models_dir: Optional[str] = None,
                 n_workers: int = 1) -> Dict[str, np.ndarray]:
    """Evaluate many `(scheme, seed)` combinations. Returns arrays of size
    `len(schemes) x len(seeds)`.

    `naive+*` schemes use a per-seed agent (one fresh agent per
    `(K, tau_p, L, seed)`) trained on the corresponding fixed topology;
    these are pre-trained in parallel before the evaluation grid runs.
    """
    n_s, n_seeds = len(schemes), len(seeds)
    thr = np.zeros((n_s, n_seeds))
    err = np.zeros((n_s, n_seeds))

    if any(s.startswith("naive+") for s in schemes):
        _pretrain_naive_agents_parallel(cfg, seeds, models_dir, n_workers)

    out = _run_eval_jobs(cfg, schemes, seeds, rt_loops,
                         models_dir, n_workers, progress)
    for (si, ti), res in out.items():
        thr[si, ti] = res["throughput_mean"]
        err[si, ti] = res["avg_err_var_frac"]
    return {
        "throughput": thr,
        "err_var_frac": err,
    }


def evaluate_all_rates(cfg: SimConfig,
                       schemes: List[str],
                       seeds: List[int],
                       rt_loops: int,
                       progress: bool = True,
                       models_dir: Optional[str] = None,
                       n_workers: int = 1) -> Dict[str, np.ndarray]:
    """Like `evaluate_all` but additionally returns the per-user,
    per-RT-loop rate samples (needed for CDF plotting).
    """
    n_s, n_seeds = len(schemes), len(seeds)
    K = cfg.K
    thr = np.zeros((n_s, n_seeds))
    err = np.zeros((n_s, n_seeds))
    rates = np.zeros((n_s, n_seeds, rt_loops, K))

    if any(s.startswith("naive+") for s in schemes):
        _pretrain_naive_agents_parallel(cfg, seeds, models_dir, n_workers)

    out = _run_eval_jobs(cfg, schemes, seeds, rt_loops,
                         models_dir, n_workers, progress)
    for (si, ti), res in out.items():
        thr[si, ti] = res["throughput_mean"]
        err[si, ti] = res["avg_err_var_frac"]
        rates[si, ti] = res["rates_history"]
    return {
        "throughput": thr,
        "err_var_frac": err,
        "rates": rates,
    }


# ---------------------------------------------------------------------------
#  Naive-DRL training and evaluation, used by `evaluate_all*` for the
#  `naive+{precoder}` schemes.
# ---------------------------------------------------------------------------
def _naive_model_path(cfg: SimConfig, models_dir: str, seed: int) -> str:
    """Per-seed naive-agent checkpoint. Single-topology training (one
    agent per `(K, tau_p, L, seed)`) is the most generous interpretation
    of the JSAC baseline (per-deployment learning), so the cache is
    keyed by the evaluation seed too.
    """
    return os.path.join(models_dir,
                        f"naive_K{cfg.K}_taup{cfg.tau_p}_L{cfg.L}_seed{seed}.npz")


def _contamination_energy(topology: Topology,
                          pilot_idx: np.ndarray,
                          cfg: SimConfig) -> float:
    """Closed-form Eq. (14)-(15) of Oh *et al.* (JSAC 2024), specialised
    to a single O-DU (so the inter-DU message-passing trivially gives
    the agent full visibility of every relevant `p_km`).

    Recall from their Eq. (22) that
       E[|g_hat_km|^2] = beta_km + xi_km + sigma^2
    where `xi_km = sum_{k' != k : pilot_k' = pilot_k} beta_k'm` is the
    pilot-contamination term and is the *only* part of the metric the
    PA can affect. We collapse `beta_km + xi_km` to a single per-pilot
    sum, so the full metric is

       p_tilde = sum_k sum_{m in MUE_k} sum_{k' : pilot_k' = pilot_k} beta_k'm

    plus a constant `|MUE|*sigma^2` offset that does not depend on the
    PA. We drop that constant since DQN is invariant to a global reward
    offset.
    """
    K, L = topology.beta.shape
    pilot_idx = np.asarray(pilot_idx, dtype=int)
    sum_per_pilot = np.zeros((L, cfg.tau_p))
    for k in range(K):
        sum_per_pilot[:, pilot_idx[k]] += topology.beta[k]

    energy = 0.0
    for k in range(K):
        pk = pilot_idx[k]
        for m in topology.serving_oru[k]:
            energy += sum_per_pilot[m, pk]
    return float(energy)


def _contamination_norm_bounds(topology: Topology,
                               cfg: SimConfig) -> Tuple[float, float]:
    """Heuristic `[p_min, p_max]` used to scale the JSAC reward to
    `[0, 1]`. We define them per-topology from physically-meaningful
    extremes so the same scale is consistent across episodes.

    * ``p_min``: every user has a unique pilot somewhere — i.e., no
      contamination — so the energy reduces to `sum_{k, m in MUE_k}
      beta_km`.
    * ``p_max``: every user shares a pilot with every other user at
      every O-RU — the worst-case contamination — so the energy is
      `sum_{k, m in MUE_k} (sum_{k'} beta_k'm)`.
    """
    K, L = topology.beta.shape
    sum_beta_per_m = topology.beta.sum(axis=0)
    p_min = 0.0
    p_max = 0.0
    for k in range(K):
        for m in topology.serving_oru[k]:
            p_min += topology.beta[k, m]
            p_max += sum_beta_per_m[m]
    if p_max <= p_min:
        p_max = p_min + 1.0  # safety floor
    return float(p_min), float(p_max)


def _naive_forbidden_actions(pilot_idx: np.ndarray, tau_p: int) -> np.ndarray:
    """Return action indices `(k * tau_p + pilot_idx[k])` for every user
    `k`. These actions are no-ops at the current state and forbidding
    them prevents the naive agent from collapsing to a fixed-point.
    """
    pilot_idx = np.asarray(pilot_idx, dtype=int)
    K = pilot_idx.shape[0]
    return (np.arange(K) * tau_p + pilot_idx).astype(np.int64)


def _train_one_naive_episode_fast(cfg: SimConfig,
                                  agent,
                                  seed: int,
                                  num_steps: int,
                                  velocity_kmh: float = 0.0
                                  ) -> List[Tuple[np.ndarray, int, float,
                                                   np.ndarray, bool]]:
    """One JSAC-style training rollout for the naive DRL agent.

    No channel sampling, no precoding, no rate computation — the reward
    after each action is the closed-form contamination metric, mapped
    to `[0, 1]` exactly as in Eq. (16) of the paper. This is several
    orders of magnitude faster than running the full robust-WMMSE
    pipeline at every step.
    """
    rng_top = np.random.default_rng(seed)
    rng_pilot = np.random.default_rng(seed + 101)
    rng_vel = np.random.default_rng(seed + 313)

    topology = build_topology(cfg, rng_top)
    user_xy = topology.user_pos[:, :2].copy()
    speed_mps = kmh_to_mps(velocity_kmh)
    user_vel = random_velocity_vectors(cfg.K, speed_mps, rng_vel)
    dt_near_rt = cfg.T_RT_sec * cfg.n_rt_per_near_rt

    pilot_idx = assign("random",
                       topology.beta, topology.serving_oru,
                       topology.users_of_oru,
                       cfg.tau_p, cfg, rng_pilot)
    p_min_topo, p_max_topo = _contamination_norm_bounds(topology, cfg)

    transitions: List[Tuple[np.ndarray, int, float,
                             np.ndarray, bool]] = []
    for step in range(num_steps):
        if speed_mps > 0.0:
            user_xy = step_positions(user_xy, user_vel, dt_near_rt, cfg)
            topology = update_topology_after_motion(topology, user_xy, cfg)
            p_min_topo, p_max_topo = _contamination_norm_bounds(topology, cfg)

        obs = naive_observation(pilot_idx, cfg.tau_p)
        forbid = _naive_forbidden_actions(pilot_idx, cfg.tau_p)
        action = agent.select_action(obs, greedy=False, forbidden=forbid)
        k_pick, t_pick = decode_naive_action(int(action), cfg.tau_p)
        pilot_idx = apply_drl_action(pilot_idx, k_pick, t_pick)

        p_tilde = _contamination_energy(topology, pilot_idx, cfg)
        reward = (p_max_topo - p_tilde) / (p_max_topo - p_min_topo)

        next_obs = naive_observation(pilot_idx, cfg.tau_p)
        done = (step == num_steps - 1)
        transitions.append((obs, int(action), float(reward),
                            next_obs, bool(done)))
    return transitions


def train_naive_agent(cfg: SimConfig,
                      model_path: str,
                      topology_seed: int,
                      num_episodes: int = 400,
                      steps_per_episode: int = 50,
                      verbose: bool = False):
    """Per-deployment training of a naive DRL pilot-assignment agent
    on the *single fixed topology* generated from `topology_seed`,
    following Sec. III-D / Alg. 1 of Oh *et al.* (JSAC 2024) but
    specialised to one O-DU.

    The topology is held constant across all `num_episodes` episodes;
    each episode resets the pilot assignment to a fresh random PA and
    runs the agent for `steps_per_episode` near-RT actions. The reward
    is the closed-form contamination metric (Eq. 14-16, replacing the
    empirical channel-estimate-energy average by its expectation via
    Eq. (22)). This setup is the most generous reading of the JSAC
    baseline — it gives the agent the option to memorize topology-
    specific PA — and was validated empirically to reach an upper-
    envelope reward of ≈0.84 on `tau_p=4` (vs ≈0.75 for random PA;
    vs ≈0.99 for the proposed greedy heuristic).
    """
    obs_dim = naive_obs_dim(cfg.K, cfg.tau_p)
    n_actions = naive_action_space(cfg.K, cfg.tau_p)
    total_train_steps = num_episodes * steps_per_episode

    dqn_cfg = DQNConfig(
        hidden=(128, 128),
        lr=5e-4,
        gamma=0.9,
        batch_size=128,
        buffer_capacity=20000,
        target_sync_every=500,
        train_every=1,
        min_buffer_for_train=512,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay_steps=max(2000, int(0.7 * total_train_steps)),
    )
    agent = make_agent("vanilla", obs_dim, n_actions, dqn_cfg,
                       rng=np.random.default_rng(topology_seed * 31 + 1))

    if verbose:
        print(f"  [train naive] K={cfg.K}, tau_p={cfg.tau_p}, L={cfg.L}, "
              f"seed={topology_seed}, eps={num_episodes}, steps/ep={steps_per_episode}")

    # Build the fixed evaluation topology once.
    topology = build_topology(cfg, np.random.default_rng(topology_seed))
    p_min_topo, p_max_topo = _contamination_norm_bounds(topology, cfg)

    log_returns: List[float] = []
    for ep in range(num_episodes):
        rng_pilot = np.random.default_rng(topology_seed * 991 + ep + 7000)
        pilot_idx = assign("random", topology.beta, topology.serving_oru,
                           topology.users_of_oru, cfg.tau_p, cfg, rng_pilot)

        ep_return = 0.0
        for step in range(steps_per_episode):
            obs = naive_observation(pilot_idx, cfg.tau_p)
            forbid = _naive_forbidden_actions(pilot_idx, cfg.tau_p)
            action = agent.select_action(obs, greedy=False, forbidden=forbid)
            k_pick, t_pick = decode_naive_action(int(action), cfg.tau_p)
            pilot_idx = apply_drl_action(pilot_idx, k_pick, t_pick)
            p_tilde = _contamination_energy(topology, pilot_idx, cfg)
            reward = (p_max_topo - p_tilde) / (p_max_topo - p_min_topo)
            next_obs = naive_observation(pilot_idx, cfg.tau_p)
            done = (step == steps_per_episode - 1)
            agent.remember(obs, action, reward, next_obs, done)
            agent.update()
            ep_return += reward
        log_returns.append(ep_return)
        if verbose and ((ep + 1) % max(1, num_episodes // 4) == 0):
            print(f"    ep {ep + 1:4d}/{num_episodes}  "
                  f"return(25)={np.mean(log_returns[-25:]):+.2f}  "
                  f"eps={agent.epsilon():.3f}")
    agent.save(model_path)
    if verbose:
        print(f"  [train naive] saved -> {model_path}")
    return agent


# Process-local cache so within a single sweep we never retrain the same
# `(K, tau_p, L, seed)` operating point twice. Keyed by that tuple; the
# cache value is the in-memory agent (already loaded from / saved to
# disk).
_NAIVE_AGENT_CACHE: Dict[Tuple[int, int, int, int], object] = {}


def _load_or_train_naive_agent(cfg: SimConfig,
                               seed: int,
                               models_dir: Optional[str] = None,
                               num_episodes: int = 400,
                               verbose: bool = False):
    """Return a per-seed naive-DRL agent, training it on first use and
    persisting the weights to `models_dir`. Idempotent and safe to call
    after `_pretrain_naive_agents_parallel`.
    """
    key = (cfg.K, cfg.tau_p, cfg.L, int(seed))
    if key in _NAIVE_AGENT_CACHE:
        return _NAIVE_AGENT_CACHE[key]

    md = models_dir or cfg.models_dir
    os.makedirs(md, exist_ok=True)
    path = _naive_model_path(cfg, md, int(seed))

    obs_dim = naive_obs_dim(cfg.K, cfg.tau_p)
    n_actions = naive_action_space(cfg.K, cfg.tau_p)
    agent = make_agent("vanilla", obs_dim, n_actions, DQNConfig(),
                       rng=np.random.default_rng(0))
    if os.path.exists(path):
        agent.load(path)
    else:
        agent = train_naive_agent(cfg, path, topology_seed=int(seed),
                                  num_episodes=num_episodes,
                                  verbose=verbose)
    agent.cfg.eps_start = 0.0
    agent.cfg.eps_end = 0.0
    _NAIVE_AGENT_CACHE[key] = agent
    return agent


# ---------------------------------------------------------------------------
#  Parallel-friendly worker entry points (must be importable for spawn).
# ---------------------------------------------------------------------------
def _train_naive_agent_worker(args):
    """Worker entry point for parallel naive-DRL training. Imports
    `numpy` lazily so that the env-var single-thread BLAS setting in
    `run_simulations.py` actually takes effect in the child."""
    cfg, seed, models_dir, num_episodes = args
    md = models_dir or cfg.models_dir
    os.makedirs(md, exist_ok=True)
    path = _naive_model_path(cfg, md, int(seed))
    if os.path.exists(path):
        return seed, "cached", path
    train_naive_agent(cfg, path, topology_seed=int(seed),
                      num_episodes=num_episodes, verbose=False)
    return seed, "trained", path


def _pretrain_naive_agents_parallel(cfg: SimConfig,
                                    seeds: List[int],
                                    models_dir: Optional[str],
                                    n_workers: int,
                                    num_episodes: int = 400) -> None:
    """Train every missing per-seed naive agent for this op-point in
    parallel. After this call returns, every seed in `seeds` has a
    checkpoint on disk under `models_dir`."""
    md = models_dir or cfg.models_dir
    os.makedirs(md, exist_ok=True)
    todo = [s for s in seeds
            if not os.path.exists(_naive_model_path(cfg, md, int(s)))]
    if not todo:
        return
    if n_workers <= 1 or len(todo) == 1:
        for s in todo:
            _train_naive_agent_worker((cfg, s, md, num_episodes))
        return
    from concurrent.futures import ProcessPoolExecutor, as_completed
    args = [(cfg, s, md, num_episodes) for s in todo]
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(_train_naive_agent_worker, a): a[1] for a in args}
        for fut in as_completed(futs):
            fut.result()


def _evaluate_one_worker(args):
    """Worker entry point for one (scheme, seed) evaluation."""
    cfg, scheme, seed, rt_loops, models_dir = args
    if scheme.startswith("naive+"):
        agent = _load_or_train_naive_agent(cfg, seed, models_dir)
        return _evaluate_naive_drl_scheme(cfg, scheme, seed, rt_loops, agent)
    return evaluate_scheme(cfg, scheme, seed, rt_loops)


def _evaluate_naive_drl_scheme(cfg: SimConfig,
                               scheme: str,
                               seed: int,
                               rt_loops: int,
                               agent) -> Dict[str, object]:
    """Evaluate a `naive+{precoder}` scheme for one seed.

    The total RT-loop budget is matched to `rt_loops`: the agent acts
    once every RT loop for `rt_loops` steps, and we report the
    aggregate throughput averaged over the whole trajectory (i.e.
    including the early steps where the agent has not yet escaped its
    random initialisation). This is the deployment regime described in
    the JSAC baseline paper and is the closest apples-to-apples
    comparison with a static PA scheme over the same number of RT
    loops.
    """
    pilot_part, precoder_name = _parse_scheme(scheme)
    if pilot_part != "naive":
        raise ValueError(f"_evaluate_naive_drl_scheme called on {scheme!r}")
    if precoder_name not in PRECODERS:
        raise ValueError(f"Unknown precoder: {precoder_name}")
    precoder = PRECODERS[precoder_name]

    # `rt_loops` total RT loops, with an action every loop.
    eval_cfg = cfg.copy_with(
        n_near_rt_per_non_rt=int(rt_loops),
        n_rt_per_near_rt=1,
    )

    out = evaluate_mobility_episode(
        cfg=eval_cfg,
        seed=int(seed),
        velocity_kmh=0.0,
        agent_kind="naive",
        agent=agent,
        non_rt_loops=1,
        pilot_init="random",
        greedy=True,
        collect_transitions=False,
        precoder=precoder,
    )
    rates_arr = np.asarray(out["rates"])  # (n_near_rt, n_rt_per_near_rt, K)
    rates_history = rates_arr.reshape(rates_arr.shape[0]
                                      * rates_arr.shape[1], cfg.K)
    per_loop = rates_history.sum(axis=-1)
    return {
        "throughput_mean": float(per_loop.mean()),
        "throughput_std": float(per_loop.std(ddof=1) if per_loop.size > 1 else 0.0),
        "pilot_entropy": float("nan"),
        "avg_err_var_frac": float("nan"),
        "rates_history": rates_history,
    }


# ---------------------------------------------------------------------------
#  Mobility-aware control loop with optional DRL pilot refinement.
# ---------------------------------------------------------------------------
def _evaluate_rates_block(topology: Topology,
                          pilot_idx: np.ndarray,
                          cfg: SimConfig,
                          rng_channel: np.random.Generator,
                          rng_noise: np.random.Generator,
                          n_rt: int,
                          precoder=robust_wmmse) -> np.ndarray:
    """Run `n_rt` RT loops with `precoder` and return the per-RT, per-user
    rate matrix of shape (n_rt, K).

    `precoder` defaults to robust WMMSE (the proposed precoder); pass
    `mrt` for cheap reward computation during DRL training, or any
    callable in `PRECODERS` for evaluation under a different precoder.
    """
    alpha, err_var, lmmse = estimation_stats(topology.beta,
                                             pilot_idx,
                                             cfg.p_ul, cfg.sigma2)
    rates = np.zeros((n_rt, cfg.K))
    for t in range(n_rt):
        h_true = sample_channel(topology, cfg.N_t, rng_channel)
        h_hat = estimate_channels(h_true, pilot_idx, topology.beta,
                                  cfg.p_ul, cfg.sigma2, lmmse, rng_noise)
        v = precoder(h_hat, err_var, topology.users_of_oru, cfg)
        rates[t] = compute_rates(h_true, v, cfg.sigma2, cfg.tau_d, cfg.tau_c)
    return rates


def _build_proposed_obs(k_star: int,
                        pilot_idx: np.ndarray,
                        topology: Topology,
                        per_user_rate: np.ndarray,
                        cfg: SimConfig) -> np.ndarray:
    qos = np.zeros(cfg.K)  # mu_k = 0 throughout this work
    return proposed_observation(k_star=k_star,
                                pilot_idx=pilot_idx,
                                beta=topology.beta,
                                serving_oru=topology.serving_oru,
                                users_of_oru=topology.users_of_oru,
                                per_user_rate=per_user_rate,
                                qos_deficit=qos,
                                tau_p=cfg.tau_p,
                                M=cfg.drl_neighbour_M,
                                cfg=cfg)


def _initial_pilots(strategy: str,
                    topology: Topology,
                    cfg: SimConfig,
                    rng_pilot: np.random.Generator) -> np.ndarray:
    if strategy == "heuristic":
        return assign("greedy",
                      topology.beta, topology.serving_oru,
                      topology.users_of_oru,
                      cfg.tau_p, cfg, rng_pilot)
    if strategy == "random":
        return assign("random",
                      topology.beta, topology.serving_oru,
                      topology.users_of_oru,
                      cfg.tau_p, cfg, rng_pilot)
    raise ValueError(f"Unknown pilot_init strategy: {strategy}")


def evaluate_mobility_episode(cfg: SimConfig,
                              seed: int,
                              velocity_kmh: float,
                              agent_kind: str,
                              agent: Optional[object],
                              non_rt_loops: int,
                              pilot_init: str = "heuristic",
                              greedy: bool = True,
                              collect_transitions: bool = False,
                              start_pilot_idx: Optional[np.ndarray] = None,
                              precoder=robust_wmmse
                              ) -> Dict[str, object]:
    """Run one mobility episode of the BCD control loop with robust WMMSE.

    Two refinement architectures are supported through `agent_kind`:

    * ``"proposed"`` — heuristic-init at the non-RT cadence + a Dueling
      DDQN refinement at the near-RT cadence. The priority score selects
      the target user `k_star`; the agent picks a new pilot index for
      that user (action space `tau_p`).
    * ``"naive"`` — pure DRL with the simplified Oh *et al.* state. The
      single agent observes only the K x tau_p pilot matrix and emits a
      joint `(user, pilot)` action (action space `K * tau_p`). No
      heuristic, no priority selector. ``pilot_init`` defaults to
      ``"heuristic"``; for the apples-to-apples Oh-style baseline pass
      ``pilot_init="random"``.
    * ``"none"`` — no DRL refinement at all (heuristic-only or
      random-only depending on `pilot_init`).

    Each non-RT loop:
      1. (re)initialise pilots according to `pilot_init`;
      2. for `cfg.n_near_rt_per_non_rt` near-RT loops:
         a. advance user positions by `T_RT * n_rt_per_near_rt`;
         b. recompute the large-scale fading and the user-centric clusters;
         c. (DRL) build the agent's observation, sample an action, apply it;
         d. evaluate `cfg.n_rt_per_near_rt` RT loops with robust WMMSE;
         e. (training) record an `(o, a, r, o')` transition.
    """
    rng_top = np.random.default_rng(seed)
    rng_pilot = np.random.default_rng(seed + 101)
    rng_vel = np.random.default_rng(seed + 313)
    rng_channel = np.random.default_rng(seed + 5001)
    rng_noise = np.random.default_rng(seed + 9001)

    topology = build_topology(cfg, rng_top)
    user_xy = topology.user_pos[:, :2].copy()
    speed_mps = kmh_to_mps(velocity_kmh)
    user_vel = random_velocity_vectors(cfg.K, speed_mps, rng_vel)
    dt_near_rt = cfg.T_RT_sec * cfg.n_rt_per_near_rt

    if start_pilot_idx is not None:
        pilot_idx = np.asarray(start_pilot_idx, dtype=int).copy()
    else:
        pilot_idx = _initial_pilots(pilot_init, topology, cfg, rng_pilot)

    rate_log: List[np.ndarray] = []
    transitions: List[Tuple[np.ndarray, int, float, np.ndarray, bool]] = []

    # Pre-estimate of per-user rate used in the proposed agent's first
    # observation (so the rate feature is not all zeros at near-RT 0).
    rates_block = _evaluate_rates_block(topology, pilot_idx, cfg,
                                        rng_channel, rng_noise,
                                        max(1, cfg.n_rt_per_near_rt // 2),
                                        precoder=precoder)
    cur_per_user_rate = rates_block.mean(axis=0)
    prev_J = float(cur_per_user_rate.sum())

    for non_rt in range(non_rt_loops):
        if non_rt > 0:
            pilot_idx = _initial_pilots(pilot_init, topology, cfg, rng_pilot)
            rates_block = _evaluate_rates_block(topology, pilot_idx, cfg,
                                                rng_channel, rng_noise,
                                                max(1, cfg.n_rt_per_near_rt // 2),
                                                precoder=precoder)
            cur_per_user_rate = rates_block.mean(axis=0)
            prev_J = float(cur_per_user_rate.sum())

        for near_rt in range(cfg.n_near_rt_per_non_rt):
            # 1. Mobility step.
            if speed_mps > 0.0:
                user_xy = step_positions(user_xy, user_vel, dt_near_rt, cfg)
                topology = update_topology_after_motion(topology, user_xy, cfg)

            # 2 + 3. Build observation and take an action.
            obs = None
            action = None
            if agent_kind == "proposed" and agent is not None:
                k_star, _ = select_priority_user(topology.beta, pilot_idx,
                                                 topology.serving_oru,
                                                 topology.users_of_oru, cfg)
                obs = _build_proposed_obs(k_star, pilot_idx, topology,
                                          cur_per_user_rate, cfg)
                # Forbid the no-op action: the agent must actually
                # *change* the pilot of `k_star`, otherwise it learns to
                # always echo the heuristic and contributes nothing.
                action = agent.select_action(obs, greedy=greedy,
                                             forbidden=int(pilot_idx[k_star]))
                pilot_idx = apply_drl_action(pilot_idx, k_star, int(action))
            elif agent_kind == "naive" and agent is not None:
                obs = naive_observation(pilot_idx, cfg.tau_p)
                forbid = _naive_forbidden_actions(pilot_idx, cfg.tau_p)
                action = agent.select_action(obs, greedy=greedy,
                                             forbidden=forbid)
                k_pick, t_pick = decode_naive_action(int(action), cfg.tau_p)
                pilot_idx = apply_drl_action(pilot_idx, k_pick, t_pick)
            elif agent_kind == "none":
                pass
            else:
                raise ValueError(f"agent_kind={agent_kind} but no agent provided")

            # 4. Evaluate the next near-RT loop with the (possibly updated) PA.
            block = _evaluate_rates_block(topology, pilot_idx, cfg,
                                          rng_channel, rng_noise,
                                          cfg.n_rt_per_near_rt,
                                          precoder=precoder)
            rate_log.append(block)
            cur_per_user_rate = block.mean(axis=0)
            new_J = float(cur_per_user_rate.sum())
            reward = new_J - prev_J
            prev_J = new_J

            if collect_transitions and obs is not None:
                if agent_kind == "proposed":
                    next_obs = _build_proposed_obs(k_star, pilot_idx, topology,
                                                   cur_per_user_rate, cfg)
                else:  # naive
                    next_obs = naive_observation(pilot_idx, cfg.tau_p)
                done = (near_rt == cfg.n_near_rt_per_non_rt - 1) and \
                       (non_rt == non_rt_loops - 1)
                transitions.append((obs, int(action), float(reward), next_obs, bool(done)))

    rate_arr = np.stack(rate_log, axis=0)
    return {
        "rates": rate_arr,
        "throughput_mean": float(rate_arr.sum(axis=-1).mean()),
        "throughput_per_near_rt": rate_arr.sum(axis=-1).mean(axis=-1),
        "transitions": transitions,
        "final_pilot_idx": pilot_idx,
    }
