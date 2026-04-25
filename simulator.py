"""One-scenario simulator.

Given a `SimConfig`, a pilot-assignment strategy name, and a precoder name,
evaluates the average aggregate downlink throughput over Monte-Carlo RT loops.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from channel import (Topology, build_topology, estimate_channels,
                     estimation_stats, sample_channel)
from config import SimConfig
from metrics import aggregate_throughput, compute_rates
from pilot_assignment import assign
from precoding import PRECODERS


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
                 progress: bool = True) -> Dict[str, np.ndarray]:
    """Evaluate many `(scheme, seed)` combinations. Returns arrays of size
    `len(schemes) x len(seeds)`."""
    n_s = len(schemes)
    n_seeds = len(seeds)
    thr = np.zeros((n_s, n_seeds))
    err = np.zeros((n_s, n_seeds))
    iterator = range(n_seeds)
    if progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc=f"seeds (tau_p={cfg.tau_p}, K={cfg.K}, L={cfg.L})")
        except ImportError:
            pass
    for si, seed in zip(iterator, seeds):
        for i, scheme in enumerate(schemes):
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
                       progress: bool = True) -> Dict[str, np.ndarray]:
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
    iterator = range(n_seeds)
    if progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc=f"seeds (tau_p={cfg.tau_p}, K={cfg.K}, L={cfg.L})")
        except ImportError:
            pass
    for si, seed in zip(iterator, seeds):
        for i, scheme in enumerate(schemes):
            res = evaluate_scheme(cfg, scheme, seed, rt_loops)
            thr[i, si] = res["throughput_mean"]
            err[i, si] = res["avg_err_var_frac"]
            rates[i, si] = res["rates_history"]
    return {
        "throughput": thr,
        "err_var_frac": err,
        "rates": rates,
    }
