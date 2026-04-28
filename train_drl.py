"""Offline training of the two DRL pilot-assignment agents.

Both agents are trained under **near-stationary** users — every episode
samples a velocity uniformly from [0, `train_v_max`] km/h with
`train_v_max = 3` by default, i.e., light pedestrian motion within an
episode. This matches the "static / quasi-static training" assumption
of the paper (the rApp trains in advance) while still giving the policy
some observation variability to generalise from. The mobility evaluation
applies the trained policy unchanged at much higher velocities to test
out-of-distribution generalisation.

Two agents are trained:

* `proposed.npz`  — Dueling DDQN with the rich observation that exposes
  per-user rate, CSI uncertainty, conflict scores, and neighbour pilots
  (Section III-B of `ORAN.tex`). It refines the heuristic PA by updating
  the pilot of the priority-selected user `k_star` each near-RT loop.
  Action space: `tau_p`.

* `naive.npz`     — vanilla DQN whose observation is **only** the
  K x tau_p one-hot pilot-assignment matrix and whose action is a joint
  `(user, pilot)` index in `[0, K * tau_p)` — a deliberately simplified
  re-implementation of Oh *et al.*, JSAC 2024. There is no heuristic
  init and no priority-based user selector: the agent runs the entire
  pilot-assignment loop on its own. With nothing in the observation
  about channel quality or motion, this baseline can in principle learn
  a good static PA pattern but cannot adapt when users start moving.

Usage
-----
    python train_drl.py            # ~5-7 min total at default settings
    python train_drl.py --quick    # smoke-train, ~1 min
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from config import DEFAULT_CONFIG
from drl import DQNConfig, make_agent
from pilot_assignment import (naive_action_space, naive_obs_dim,
                              proposed_obs_dim)
from simulator import evaluate_mobility_episode


def _train_agent(kind: str,
                 cfg,
                 train_cfg,
                 num_episodes: int,
                 seed_start: int,
                 dqn_cfg: DQNConfig,
                 model_path: str,
                 train_v_max: float = 3.0,
                 verbose: bool = True) -> None:
    if kind == "proposed":
        agent_kind = "proposed"
        pilot_init = "heuristic"
        obs_dim = proposed_obs_dim(cfg.tau_p, cfg.drl_neighbour_M)
        n_actions = cfg.tau_p
        agent = make_agent("dueling", obs_dim, n_actions, dqn_cfg,
                           rng=np.random.default_rng(seed_start * 7919))
    elif kind == "naive":
        agent_kind = "naive"
        pilot_init = "random"
        obs_dim = naive_obs_dim(cfg.K, cfg.tau_p)
        n_actions = naive_action_space(cfg.K, cfg.tau_p)
        agent = make_agent("vanilla", obs_dim, n_actions, dqn_cfg,
                           rng=np.random.default_rng(seed_start * 7919 + 1))
    else:
        raise ValueError(kind)

    if verbose:
        print(f"[train:{kind}] obs_dim={obs_dim}, n_actions={n_actions}, "
              f"episodes={num_episodes}, hidden={dqn_cfg.hidden}, "
              f"wmmse_iters={train_cfg.wmmse_outer_iters}, "
              f"n_rt={train_cfg.n_rt_per_near_rt}, "
              f"train_v_max={train_v_max} km/h")

    t0 = time.time()
    log_returns = []
    rng_v = np.random.default_rng(seed_start + 99991)
    for ep in range(num_episodes):
        seed = seed_start + ep
        v_ep = float(rng_v.uniform(0.0, train_v_max))
        out = evaluate_mobility_episode(
            cfg=train_cfg,
            seed=seed,
            velocity_kmh=v_ep,
            agent_kind=agent_kind,
            agent=agent,
            non_rt_loops=1,
            pilot_init=pilot_init,
            greedy=False,
            collect_transitions=True,
        )
        for (o, a, r, op, d) in out["transitions"]:
            agent.remember(o, a, r, op, d)
            agent.update()

        ep_return = float(np.sum([t[2] for t in out["transitions"]]))
        log_returns.append(ep_return)
        if verbose and ((ep + 1) % max(1, num_episodes // 10) == 0):
            avg_thr = float(out["throughput_mean"])
            print(f"  ep {ep + 1:4d}/{num_episodes}  "
                  f"return={np.mean(log_returns[-50:]):+.2f}  "
                  f"thr={avg_thr:6.2f}  eps={agent.epsilon():.3f}  "
                  f"buf={agent.buffer.size}")

    agent.save(model_path)
    if verbose:
        elapsed = time.time() - t0
        print(f"[train:{kind}] saved -> {model_path}  ({elapsed:.1f}s)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="run a tiny smoke-train (~1 min)")
    parser.add_argument("--out", default=DEFAULT_CONFIG.models_dir)
    parser.add_argument("--seed-start", type=int, default=4242)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--only", choices=("proposed", "naive", "both"),
                        default="both")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--train-v-max", type=float, default=3.0,
                        help="upper bound (km/h) of the per-episode "
                             "training velocity sampled uniformly from "
                             "[0, train_v_max]. Set to 0 to train under "
                             "strictly stationary users.")
    args = parser.parse_args()

    cfg = DEFAULT_CONFIG
    os.makedirs(args.out, exist_ok=True)

    if args.episodes is not None:
        episodes = args.episodes
    else:
        episodes = 30 if args.quick else 150

    # Speed up training: fewer WMMSE outer iters and fewer RT loops per
    # near-RT loop. The reward (delta-J) is still meaningful — the
    # precoder is the same, just less iterated. At eval time we use the
    # full-fidelity cfg.
    train_cfg = cfg.copy_with(
        wmmse_outer_iters=8,
        n_rt_per_near_rt=4,
    )

    dqn_cfg = DQNConfig(
        hidden=(64, 64),
        lr=args.lr,
        gamma=args.gamma,
        batch_size=64,
        buffer_capacity=5000,
        target_sync_every=200,
        train_every=2,
        min_buffer_for_train=128,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay_steps=max(200, episodes * cfg.n_near_rt_per_non_rt // 2),
    )

    if args.only in ("proposed", "both"):
        _train_agent("proposed", cfg, train_cfg, episodes, args.seed_start,
                     dqn_cfg, os.path.join(args.out, "proposed.npz"),
                     train_v_max=args.train_v_max)
    if args.only in ("naive", "both"):
        _train_agent("naive", cfg, train_cfg, episodes,
                     args.seed_start + 10000, dqn_cfg,
                     os.path.join(args.out, "naive.npz"),
                     train_v_max=args.train_v_max)


if __name__ == "__main__":
    main()
