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
from typing import Dict, List, Optional, Tuple

import numpy as np

from channel import (Topology, build_topology, estimate_channels,
                     estimation_stats, sample_channel)
from config import SimConfig
from drl import DQNConfig, make_agent
from metrics import aggregate_throughput, compute_rates
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


def evaluate_all(cfg: SimConfig,
                 schemes: List[str],
                 seeds: List[int],
                 rt_loops: int,
                 progress: bool = True,
                 models_dir: Optional[str] = None) -> Dict[str, np.ndarray]:
    """Evaluate many `(scheme, seed)` combinations. Returns arrays of size
    `len(schemes) x len(seeds)`.

    Schemes whose pilot key is `naive` (e.g. `naive+oblivious`) are
    evaluated through the offline-trained naive DRL agent; the agent is
    trained once per `(K, tau_p, L)` operating point with the cheap MRT
    reward, cached under `models_dir`, and re-used across precoders.
    """
    n_s = len(schemes)
    n_seeds = len(seeds)
    thr = np.zeros((n_s, n_seeds))
    err = np.zeros((n_s, n_seeds))

    naive_agent = None
    if any(s.startswith("naive+") for s in schemes):
        naive_agent = _load_or_train_naive_agent(cfg, models_dir)

    iterator = range(n_seeds)
    if progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc=f"seeds (tau_p={cfg.tau_p}, K={cfg.K}, L={cfg.L})")
        except ImportError:
            pass
    for si, seed in zip(iterator, seeds):
        for i, scheme in enumerate(schemes):
            if scheme.startswith("naive+"):
                res = _evaluate_naive_drl_scheme(cfg, scheme, seed,
                                                 rt_loops, naive_agent)
            else:
                res = evaluate_scheme(cfg, scheme, seed, rt_loops)
            thr[i, si] = res["throughput_mean"]
            err[i, si] = res["avg_err_var_frac"]
    return {
        "throughput": thr,
        "err_var_frac": err,
    }


def evaluate_all_rates(cfg: SimConfig,
                       schemes: List[str],
                       seeds: List[int],
                       rt_loops: int,
                       progress: bool = True,
                       models_dir: Optional[str] = None) -> Dict[str, np.ndarray]:
    """Evaluate `(scheme, seed)` combinations and additionally retain the
    per-user, per-RT-loop rate samples needed to draw a CDF.

    Returns
    -------
    dict with::

        "throughput"   : (S, n_seeds)              aggregate throughput per seed
        "err_var_frac" : (S, n_seeds)              avg estimation-error fraction
        "rates"        : (S, n_seeds, rt_loops, K) per-user spectral efficiency
                                                    samples (bits/s/Hz per user)
    """
    n_s = len(schemes)
    n_seeds = len(seeds)
    K = cfg.K
    thr = np.zeros((n_s, n_seeds))
    err = np.zeros((n_s, n_seeds))
    rates = np.zeros((n_s, n_seeds, rt_loops, K))

    naive_agent = None
    if any(s.startswith("naive+") for s in schemes):
        naive_agent = _load_or_train_naive_agent(cfg, models_dir)

    iterator = range(n_seeds)
    if progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc=f"seeds (tau_p={cfg.tau_p}, K={cfg.K}, L={cfg.L})")
        except ImportError:
            pass
    for si, seed in zip(iterator, seeds):
        for i, scheme in enumerate(schemes):
            if scheme.startswith("naive+"):
                res = _evaluate_naive_drl_scheme(cfg, scheme, seed,
                                                 rt_loops, naive_agent)
            else:
                res = evaluate_scheme(cfg, scheme, seed, rt_loops)
            thr[i, si] = res["throughput_mean"]
            err[i, si] = res["avg_err_var_frac"]
            rates[i, si] = res["rates_history"]
    return {
        "throughput": thr,
        "err_var_frac": err,
        "rates": rates,
    }


# ---------------------------------------------------------------------------
#  Naive-DRL training and evaluation, used by `evaluate_all*` for the
#  `naive+{precoder}` schemes.
# ---------------------------------------------------------------------------
def _naive_model_path(cfg: SimConfig, models_dir: str) -> str:
    return os.path.join(models_dir,
                        f"naive_K{cfg.K}_taup{cfg.tau_p}_L{cfg.L}.npz")


def train_naive_agent(cfg: SimConfig,
                      model_path: str,
                      num_episodes: int = 400,
                      train_v_max: float = 0.0,
                      seed_start: int = 0,
                      verbose: bool = True):
    """Offline-train a naive DRL pilot-assignment agent at the given
    `(K, tau_p, L)` operating point.

    The reward is computed under the cheap MRT precoder so that several
    hundred training episodes finish in a couple of minutes. The same
    trained agent is then reused at evaluation time against any of the
    `PRECODERS` (the naive observation has no precoder-specific
    features, so retraining per evaluation precoder buys little — it is
    the same simplification adopted in the JSAC baseline paper).

    With action space `K * tau_p` (up to a few hundred), the DQN is
    prone to mode collapse if exploration is too narrow. We therefore
    use an extended epsilon schedule (decays over ~70 % of training and
    floors at 0.1) and a longer episode (20 near-RT loops) to give the
    replay buffer broader coverage of the joint state-action space.
    """
    obs_dim = naive_obs_dim(cfg.K, cfg.tau_p)
    n_actions = naive_action_space(cfg.K, cfg.tau_p)

    train_cfg = cfg.copy_with(
        wmmse_outer_iters=8,
        n_rt_per_near_rt=2,
        n_near_rt_per_non_rt=20,
    )
    transitions_per_ep = train_cfg.n_near_rt_per_non_rt
    total_train_steps = num_episodes * transitions_per_ep

    dqn_cfg = DQNConfig(
        hidden=(128, 128),
        lr=5e-4,
        gamma=0.5,
        batch_size=128,
        buffer_capacity=20000,
        target_sync_every=500,
        train_every=1,
        min_buffer_for_train=512,
        eps_start=1.0,
        eps_end=0.10,
        eps_decay_steps=max(2000, int(0.7 * total_train_steps)),
    )
    agent = make_agent("vanilla", obs_dim, n_actions, dqn_cfg,
                       rng=np.random.default_rng(seed_start * 7919 + 1))
    if verbose:
        print(f"  [train naive] K={cfg.K}, tau_p={cfg.tau_p}, L={cfg.L}, "
              f"obs_dim={obs_dim}, n_actions={n_actions}, "
              f"episodes={num_episodes}")
    rng_v = np.random.default_rng(seed_start + 99991)
    log_returns: List[float] = []
    for ep in range(num_episodes):
        v_ep = float(rng_v.uniform(0.0, train_v_max))
        out = evaluate_mobility_episode(
            cfg=train_cfg,
            seed=seed_start + ep,
            velocity_kmh=v_ep,
            agent_kind="naive",
            agent=agent,
            non_rt_loops=1,
            pilot_init="random",
            greedy=False,
            collect_transitions=True,
            precoder=mrt,
        )
        for (o, a, r, op, d) in out["transitions"]:
            agent.remember(o, a, r, op, d)
            agent.update()
        log_returns.append(float(np.sum([t[2] for t in out["transitions"]])))
        if verbose and ((ep + 1) % max(1, num_episodes // 8) == 0):
            print(f"    ep {ep + 1:4d}/{num_episodes}  "
                  f"return(50)={np.mean(log_returns[-50:]):+.2f}  "
                  f"thr={out['throughput_mean']:6.2f}  "
                  f"eps={agent.epsilon():.3f}")
    agent.save(model_path)
    if verbose:
        print(f"  [train naive] saved -> {model_path}")
    return agent


# Process-local cache so within a single sweep we never retrain the same
# (K, tau_p, L) operating point twice. Keyed by `(K, tau_p, L)` and the
# cache value is the in-memory agent (already loaded from / saved to
# disk).
_NAIVE_AGENT_CACHE: Dict[Tuple[int, int, int], object] = {}


def _load_or_train_naive_agent(cfg: SimConfig,
                               models_dir: Optional[str] = None,
                               num_episodes: int = 400):
    """Return a naive-DRL agent for this `(K, tau_p, L)`, training it
    on first use and persisting the weights."""
    key = (cfg.K, cfg.tau_p, cfg.L)
    if key in _NAIVE_AGENT_CACHE:
        return _NAIVE_AGENT_CACHE[key]

    md = models_dir or cfg.models_dir
    os.makedirs(md, exist_ok=True)
    path = _naive_model_path(cfg, md)

    obs_dim = naive_obs_dim(cfg.K, cfg.tau_p)
    n_actions = naive_action_space(cfg.K, cfg.tau_p)
    agent = make_agent("vanilla", obs_dim, n_actions, DQNConfig(),
                       rng=np.random.default_rng(0))
    if os.path.exists(path):
        agent.load(path)
        agent.cfg.eps_start = 0.0
        agent.cfg.eps_end = 0.0
    else:
        agent = train_naive_agent(cfg, path, num_episodes=num_episodes)
        agent.cfg.eps_start = 0.0
        agent.cfg.eps_end = 0.0
    _NAIVE_AGENT_CACHE[key] = agent
    return agent


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
                action = agent.select_action(obs, greedy=greedy)
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
