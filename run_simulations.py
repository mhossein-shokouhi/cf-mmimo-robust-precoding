"""Top-level experiment driver.

Produces the three experiments needed for the GLOBECOM paper:

* `tau_p_sweep`  — aggregate throughput vs. pilot-codebook size.
* `K_sweep`      — aggregate throughput vs. number of users.
* `L_sweep`      — aggregate throughput vs. number of O-RUs.

For each configuration and each scheme, multiple seeds are averaged.
Results are saved as `.npz` archives under `results/`.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Iterable, List

import numpy as np

from config import (CDF_POINT, DEFAULT_CONFIG, K_SWEEP, L_SWEEP, SCHEMES,
                    SimConfig, TAU_P_SWEEP)
from simulator import evaluate_all, evaluate_all_rates


def _seeds(num_seeds: int, start: int = 1234) -> List[int]:
    return list(range(start, start + num_seeds))


def _ensure(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def run_tau_p_sweep(cfg: SimConfig,
                    num_seeds: int,
                    rt_loops: int,
                    schemes: Iterable[str],
                    out_dir: str) -> str:
    _ensure(out_dir)
    schemes = list(schemes)
    seeds = _seeds(num_seeds)
    tau_ps = TAU_P_SWEEP
    thr = np.zeros((len(schemes), len(tau_ps), num_seeds))
    t0 = time.time()
    for ti, tau_p in enumerate(tau_ps):
        cfg_tau = cfg.copy_with(tau_p=tau_p)
        res = evaluate_all(cfg_tau, schemes, seeds, rt_loops)
        thr[:, ti, :] = res["throughput"]
    elapsed = time.time() - t0
    path = os.path.join(out_dir, "tau_p_sweep.npz")
    np.savez(path,
             schemes=np.array(schemes),
             tau_ps=np.array(tau_ps),
             seeds=np.array(seeds),
             throughput=thr,
             elapsed_sec=elapsed,
             K=cfg.K, L=cfg.L, N_t=cfg.N_t)
    print(f"[tau_p sweep] saved -> {path} (elapsed: {elapsed:.1f} s)")
    return path


def run_K_sweep(cfg: SimConfig,
                num_seeds: int,
                rt_loops: int,
                schemes: Iterable[str],
                out_dir: str) -> str:
    _ensure(out_dir)
    schemes = list(schemes)
    seeds = _seeds(num_seeds)
    Ks = K_SWEEP
    thr = np.zeros((len(schemes), len(Ks), num_seeds))
    t0 = time.time()
    for ki, K in enumerate(Ks):
        cfg_k = cfg.copy_with(K=K)
        res = evaluate_all(cfg_k, schemes, seeds, rt_loops)
        thr[:, ki, :] = res["throughput"]
    elapsed = time.time() - t0
    path = os.path.join(out_dir, "K_sweep.npz")
    np.savez(path,
             schemes=np.array(schemes),
             Ks=np.array(Ks),
             seeds=np.array(seeds),
             throughput=thr,
             elapsed_sec=elapsed,
             tau_p=cfg.tau_p, L=cfg.L, N_t=cfg.N_t)
    print(f"[K sweep] saved -> {path} (elapsed: {elapsed:.1f} s)")
    return path


def run_L_sweep(cfg: SimConfig,
                num_seeds: int,
                rt_loops: int,
                schemes: Iterable[str],
                out_dir: str) -> str:
    _ensure(out_dir)
    schemes = list(schemes)
    seeds = _seeds(num_seeds)
    Ls = L_SWEEP
    thr = np.zeros((len(schemes), len(Ls), num_seeds))
    t0 = time.time()
    for li, L in enumerate(Ls):
        cfg_l = cfg.copy_with(L=L, L_max=min(cfg.L_max, L))
        res = evaluate_all(cfg_l, schemes, seeds, rt_loops)
        thr[:, li, :] = res["throughput"]
    elapsed = time.time() - t0
    path = os.path.join(out_dir, "L_sweep.npz")
    np.savez(path,
             schemes=np.array(schemes),
             Ls=np.array(Ls),
             seeds=np.array(seeds),
             throughput=thr,
             elapsed_sec=elapsed,
             tau_p=cfg.tau_p, K=cfg.K, N_t=cfg.N_t)
    print(f"[L sweep] saved -> {path} (elapsed: {elapsed:.1f} s)")
    return path


def run_cdf_point(cfg: SimConfig,
                  num_seeds: int,
                  rt_loops: int,
                  schemes: Iterable[str],
                  out_dir: str,
                  tau_p: int,
                  K: int,
                  L: int) -> str:
    """Evaluate a single operating point and retain per-user rate samples for
    CDF plotting. The samples have shape `(S, n_seeds, rt_loops, K)`."""
    _ensure(out_dir)
    schemes = list(schemes)
    seeds = _seeds(num_seeds)
    cfg_pt = cfg.copy_with(tau_p=tau_p, K=K, L=L, L_max=min(cfg.L_max, L))
    t0 = time.time()
    res = evaluate_all_rates(cfg_pt, schemes, seeds, rt_loops)
    elapsed = time.time() - t0
    path = os.path.join(out_dir, "cdf_point.npz")
    np.savez(path,
             schemes=np.array(schemes),
             seeds=np.array(seeds),
             throughput=res["throughput"],
             rates=res["rates"],
             err_var_frac=res["err_var_frac"],
             elapsed_sec=elapsed,
             tau_p=cfg_pt.tau_p, K=cfg_pt.K, L=cfg_pt.L,
             N_t=cfg_pt.N_t, L_max=cfg_pt.L_max,
             rt_loops=rt_loops)
    print(f"[CDF point] saved -> {path} (elapsed: {elapsed:.1f} s)")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true",
                        help="run a tiny pipeline check with few seeds / RT loops")
    parser.add_argument("--out", default=DEFAULT_CONFIG.results_dir)
    parser.add_argument("--no-tau", action="store_true")
    parser.add_argument("--no-K", action="store_true")
    parser.add_argument("--no-L", action="store_true")
    parser.add_argument("--no-cdf", action="store_true")
    parser.add_argument("--only-cdf", action="store_true",
                        help="run only the CDF point and skip the sweeps")
    args = parser.parse_args()

    cfg = DEFAULT_CONFIG
    if args.smoke:
        num_seeds = cfg.smoke_seeds
        rt_loops = cfg.smoke_rt_loops
    else:
        num_seeds = cfg.num_seeds
        rt_loops = cfg.rt_loops_per_seed

    schemes = list(SCHEMES)
    _ensure(args.out)

    if args.only_cdf:
        run_cdf_point(cfg, num_seeds, rt_loops, schemes, args.out, **CDF_POINT)
        return

    if not args.no_tau:
        run_tau_p_sweep(cfg, num_seeds, rt_loops, schemes, args.out)
    if not args.no_K:
        run_K_sweep(cfg, num_seeds, rt_loops, schemes, args.out)
    if not args.no_L:
        run_L_sweep(cfg, num_seeds, rt_loops, schemes, args.out)
    if not args.no_cdf:
        run_cdf_point(cfg, num_seeds, rt_loops, schemes, args.out, **CDF_POINT)


if __name__ == "__main__":
    main()
