"""Mobility-sweep experiment: aggregate throughput vs. user velocity.

Three pilot-assignment architectures are compared, all evaluated with the
**robust WMMSE precoder**:

* `heuristic`     — Heuristic-only PA, re-initialised at every non-RT
  loop. No DRL refinement. This is the current "Proposed Algorithm" of
  the repository.
* `proposed_drl`  — Heuristic init at the non-RT cadence, plus a
  Dueling DDQN refinement at the near-RT cadence (Section III-B of
  `ORAN.tex`). The DDQN's observation is rich (rate, CSI uncertainty,
  conflict scores, neighbour pilots) so it can react to changes in the
  channel statistics caused by user mobility.
* `naive_drl`     — Pure DRL with the simplified Oh *et al.* (JSAC 2024)
  state. The agent's observation is **only** the K x tau_p pilot
  matrix. Action space is the joint `(user, pilot)` index. No heuristic
  init, no priority-based user selection. The agent has no signal about
  channel quality, so it cannot adapt under mobility.

Both DRL agents are trained under stationary users (`train_drl.py`) and
deployed greedily at every velocity. The expectation is:

* At v = 0 the heuristic alone is near-optimal so all three are close.
* As v grows, the rich-observation `proposed_drl` tracks the moving
  channel statistics, the heuristic re-initialisation absorbs some of the
  drift but stays one step behind, and the `naive_drl` policy collapses
  because its observation no longer matches the training distribution.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import List

import numpy as np

from config import DEFAULT_CONFIG, SimConfig
from drl import DQNConfig, make_agent
from pilot_assignment import (naive_action_space, naive_obs_dim,
                              proposed_obs_dim)
from simulator import evaluate_mobility_episode


SCHEMES_MOBILITY = ("heuristic", "proposed_drl", "naive_drl")


def _make_eval_cfg(cfg: SimConfig) -> SimConfig:
    """At eval time we extend the non-RT loop so motion can accumulate
    between heuristic re-inits, and slightly relax the WMMSE iteration
    count to keep the wall time tractable."""
    return cfg.copy_with(
        n_near_rt_per_non_rt=cfg.mobility_near_rt_per_non_rt,
        wmmse_outer_iters=15,
    )


def _load_agent(kind: str, cfg: SimConfig, model_path: str):
    dqn_cfg = DQNConfig()
    if kind == "proposed":
        obs_dim = proposed_obs_dim(cfg.tau_p, cfg.drl_neighbour_M)
        agent = make_agent("dueling", obs_dim, cfg.tau_p, dqn_cfg,
                           rng=np.random.default_rng(0))
    elif kind == "naive":
        obs_dim = naive_obs_dim(cfg.K, cfg.tau_p)
        n_actions = naive_action_space(cfg.K, cfg.tau_p)
        agent = make_agent("vanilla", obs_dim, n_actions, dqn_cfg,
                           rng=np.random.default_rng(0))
    else:
        raise ValueError(kind)
    agent.load(model_path)
    agent.cfg.eps_start = 0.0
    agent.cfg.eps_end = 0.0
    return agent


def _evaluate_one(scheme: str,
                  cfg: SimConfig,
                  velocity_kmh: float,
                  seeds: List[int],
                  models_dir: str,
                  non_rt_loops: int) -> np.ndarray:
    """Return per-seed mean aggregate throughput (length len(seeds))."""
    eval_cfg = _make_eval_cfg(cfg)

    if scheme == "heuristic":
        agent_kind, agent, pilot_init = "none", None, "heuristic"
    elif scheme == "proposed_drl":
        agent_kind = "proposed"
        agent = _load_agent("proposed", eval_cfg,
                            os.path.join(models_dir, "proposed.npz"))
        pilot_init = "heuristic"
    elif scheme == "naive_drl":
        agent_kind = "naive"
        agent = _load_agent("naive", eval_cfg,
                            os.path.join(models_dir, "naive.npz"))
        pilot_init = "random"
    else:
        raise ValueError(scheme)

    out = np.zeros(len(seeds))
    for si, seed in enumerate(seeds):
        result = evaluate_mobility_episode(
            cfg=eval_cfg,
            seed=seed + 9_000_000,
            velocity_kmh=velocity_kmh,
            agent_kind=agent_kind,
            agent=agent,
            non_rt_loops=non_rt_loops,
            pilot_init=pilot_init,
            greedy=True,
            collect_transitions=False,
        )
        out[si] = result["throughput_mean"]
    return out


def run_mobility_sweep(cfg: SimConfig,
                       velocities: List[float],
                       num_seeds: int,
                       schemes: List[str],
                       models_dir: str,
                       out_dir: str,
                       non_rt_loops: int = 1,
                       seed_start: int = 1234,
                       verbose: bool = True) -> str:
    os.makedirs(out_dir, exist_ok=True)
    seeds = list(range(seed_start, seed_start + num_seeds))
    thr = np.zeros((len(schemes), len(velocities), num_seeds))

    t0 = time.time()
    for vi, v in enumerate(velocities):
        for si, sch in enumerate(schemes):
            thr[si, vi, :] = _evaluate_one(sch, cfg, v, seeds, models_dir,
                                           non_rt_loops)
            if verbose:
                m = thr[si, vi].mean()
                s = thr[si, vi].std(ddof=1) if num_seeds > 1 else 0.0
                print(f"  v={v:5.1f} km/h  {sch:14s}  thr={m:6.2f} +- {s:5.2f}")

    elapsed = time.time() - t0
    path = os.path.join(out_dir, "mobility_sweep.npz")
    np.savez(path,
             schemes=np.array(schemes),
             velocities=np.array(velocities),
             seeds=np.array(seeds),
             throughput=thr,
             elapsed_sec=elapsed,
             K=cfg.K, L=cfg.L, tau_p=cfg.tau_p, N_t=cfg.N_t,
             non_rt_loops=non_rt_loops,
             eval_near_rt_per_non_rt=cfg.mobility_near_rt_per_non_rt)
    if verbose:
        print(f"[mobility sweep] saved -> {path} ({elapsed:.1f} s)")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=DEFAULT_CONFIG.results_dir)
    parser.add_argument("--models", default=DEFAULT_CONFIG.models_dir)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--non-rt-loops", type=int, default=1)
    parser.add_argument("--velocities", type=float, nargs="+",
                        default=list(DEFAULT_CONFIG.velocity_kmh_eval))
    parser.add_argument("--schemes", nargs="+", default=list(SCHEMES_MOBILITY))
    args = parser.parse_args()

    run_mobility_sweep(cfg=DEFAULT_CONFIG,
                       velocities=args.velocities,
                       num_seeds=args.seeds,
                       schemes=args.schemes,
                       models_dir=args.models,
                       out_dir=args.out,
                       non_rt_loops=args.non_rt_loops)


if __name__ == "__main__":
    main()
