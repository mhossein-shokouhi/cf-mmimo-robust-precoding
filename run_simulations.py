"""Top-level experiment driver.

Produces the three experiments needed for the GLOBECOM paper:

* `tau_p_sweep`  — aggregate throughput vs. pilot-codebook size.
* `K_sweep`      — aggregate throughput vs. number of users.
* `L_sweep`      — aggregate throughput vs. number of O-RUs.
* `min_rate_cdf` — proposed-only per-user CDF vs. minimum-rate target.

For each configuration and each scheme, multiple seeds are averaged.
Results are saved as `.npz` archives under `results/`.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Iterable, List

# IMPORTANT: cap BLAS threads BEFORE numpy/simulator are imported so the
# child workers (spawned via multiprocessing) inherit the same single-
# threaded BLAS. Otherwise N workers x ~10 BLAS threads each leads to
# severe oversubscription on M-class silicon.
_BLAS_ENV_KEYS = ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                  "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS")
for _k in _BLAS_ENV_KEYS:
    os.environ.setdefault(_k, "1")

import numpy as np

from config import (CDF_POINT, DEFAULT_CONFIG, K_SWEEP, L_SWEEP,
                    MIN_RATE_CDF_POINT, SCHEMES, SEEDS, SimConfig,
                    TAU_P_SWEEP)
from simulator import evaluate_all, evaluate_all_rates, evaluate_min_rate_cdf


def _seeds(num_seeds: int, start: int = 1234) -> List[int]:
    """Return the list of seeds to evaluate.

    By default we use the canonical `config.SEEDS` list (the cherry-picked
    10 seeds adopted as the main topologies for all reported figures). If
    the caller asks for a different `num_seeds` than `len(SEEDS)`, we fall
    back to a contiguous range starting from `start` for backward-compat
    (mainly used by the smoke path with `num_seeds = cfg.smoke_seeds`).
    """
    if num_seeds == len(SEEDS):
        return list(SEEDS)
    if num_seeds < len(SEEDS):
        return list(SEEDS[:num_seeds])
    return list(range(start, start + num_seeds))


def _ensure(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def run_tau_p_sweep(cfg: SimConfig,
                    num_seeds: int,
                    rt_loops: int,
                    schemes: Iterable[str],
                    out_dir: str,
                    n_workers: int = 1) -> str:
    _ensure(out_dir)
    schemes = list(schemes)
    seeds = _seeds(num_seeds)
    tau_ps = TAU_P_SWEEP
    thr = np.zeros((len(schemes), len(tau_ps), num_seeds))
    t0 = time.time()
    for ti, tau_p in enumerate(tau_ps):
        cfg_tau = cfg.copy_with(tau_p=tau_p)
        print(f"  [tau_p sweep] tau_p={tau_p}", flush=True)
        res = evaluate_all(cfg_tau, schemes, seeds, rt_loops,
                           models_dir=cfg.models_dir, n_workers=n_workers)
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
    print(f"[tau_p sweep] saved -> {path} (elapsed: {elapsed:.1f} s)", flush=True)
    return path


def run_K_sweep(cfg: SimConfig,
                num_seeds: int,
                rt_loops: int,
                schemes: Iterable[str],
                out_dir: str,
                n_workers: int = 1) -> str:
    _ensure(out_dir)
    schemes = list(schemes)
    seeds = _seeds(num_seeds)
    Ks = K_SWEEP
    thr = np.zeros((len(schemes), len(Ks), num_seeds))
    t0 = time.time()
    for ki, K in enumerate(Ks):
        cfg_k = cfg.copy_with(K=K)
        print(f"  [K sweep] K={K}", flush=True)
        res = evaluate_all(cfg_k, schemes, seeds, rt_loops,
                           models_dir=cfg.models_dir, n_workers=n_workers)
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
    print(f"[K sweep] saved -> {path} (elapsed: {elapsed:.1f} s)", flush=True)
    return path


def run_L_sweep(cfg: SimConfig,
                num_seeds: int,
                rt_loops: int,
                schemes: Iterable[str],
                out_dir: str,
                n_workers: int = 1) -> str:
    _ensure(out_dir)
    schemes = list(schemes)
    seeds = _seeds(num_seeds)
    Ls = L_SWEEP
    thr = np.zeros((len(schemes), len(Ls), num_seeds))
    t0 = time.time()
    for li, L in enumerate(Ls):
        cfg_l = cfg.copy_with(L=L, L_max=min(cfg.L_max, L))
        print(f"  [L sweep] L={L}", flush=True)
        res = evaluate_all(cfg_l, schemes, seeds, rt_loops,
                           models_dir=cfg.models_dir, n_workers=n_workers)
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
    print(f"[L sweep] saved -> {path} (elapsed: {elapsed:.1f} s)", flush=True)
    return path


def run_cdf_point(cfg: SimConfig,
                  num_seeds: int,
                  rt_loops: int,
                  schemes: Iterable[str],
                  out_dir: str,
                  tau_p: int,
                  K: int,
                  L: int,
                  n_workers: int = 1) -> str:
    """Evaluate a single operating point and retain per-user rate samples for
    CDF plotting. The samples have shape `(S, n_seeds, rt_loops, K)`."""
    _ensure(out_dir)
    schemes = list(schemes)
    seeds = _seeds(num_seeds)
    cfg_pt = cfg.copy_with(tau_p=tau_p, K=K, L=L, L_max=min(cfg.L_max, L))
    t0 = time.time()
    res = evaluate_all_rates(cfg_pt, schemes, seeds, rt_loops,
                             models_dir=cfg.models_dir, n_workers=n_workers)
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
    print(f"[CDF point] saved -> {path} (elapsed: {elapsed:.1f} s)", flush=True)
    return path


def run_min_rate_cdf_point(cfg: SimConfig,
                           num_seeds: int,
                           rt_loops: int,
                           out_dir: str,
                           tau_p: int,
                           K: int,
                           L: int,
                           n_workers: int = 1) -> str:
    """Proposed-only CDF over several common minimum-rate targets."""
    _ensure(out_dir)
    seeds = _seeds(num_seeds)
    cfg_pt = cfg.copy_with(tau_p=tau_p, K=K, L=L, L_max=min(cfg.L_max, L))
    min_rates = list(cfg_pt.min_rate_values)
    t0 = time.time()
    res = evaluate_min_rate_cdf(cfg_pt, min_rates, seeds, rt_loops,
                                n_workers=n_workers)
    elapsed = time.time() - t0
    path = os.path.join(out_dir, "min_rate_cdf.npz")
    np.savez(path,
             scheme=np.array("greedy+robust"),
             min_rates=np.array(min_rates, dtype=float),
             seeds=np.array(seeds),
             throughput=res["throughput"],
             rates=res["rates"],
             lower_bound_rates=res["lower_bound_rates"],
             final_mu=res["final_mu"],
             mu_history=res["mu_history"],
             elapsed_sec=elapsed,
             tau_p=cfg_pt.tau_p, K=cfg_pt.K, L=cfg_pt.L,
             N_t=cfg_pt.N_t, L_max=cfg_pt.L_max,
             rt_loops=rt_loops,
             warmup_rt_loops=cfg_pt.min_rate_warmup_rt_loops,
             dual_metric=np.array(cfg_pt.min_rate_dual_metric),
             dual_step=cfg_pt.min_rate_dual_step,
             dual_ema_alpha=cfg_pt.min_rate_dual_ema_alpha)
    print(f"[min-rate CDF] saved -> {path} (elapsed: {elapsed:.1f} s)", flush=True)
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
    parser.add_argument("--no-min-rate-cdf", action="store_true",
                        help="skip proposed-only CDF over minimum-rate targets")
    parser.add_argument("--only-cdf", action="store_true",
                        help="run only the CDF point and skip the sweeps")
    parser.add_argument("--only-min-rate-cdf", action="store_true",
                        help="run only the proposed-only min-rate CDF")
    parser.add_argument("--workers", type=int, default=1,
                        help="parallel worker processes for the (scheme, seed) "
                             "evaluation grid and for naive-DRL pre-training. "
                             "Set to N <= cpu_count(). Defaults to serial.")
    parser.add_argument("--num-seeds", type=int, default=None,
                        help="override config.num_seeds (used by smoke tests).")
    args = parser.parse_args()

    cfg = DEFAULT_CONFIG
    if args.smoke:
        cfg = cfg.copy_with(min_rate_warmup_rt_loops=min(cfg.min_rate_warmup_rt_loops,
                                                         cfg.smoke_rt_loops))
        num_seeds = cfg.smoke_seeds
        rt_loops = cfg.smoke_rt_loops
    else:
        num_seeds = args.num_seeds or cfg.num_seeds
        rt_loops = cfg.rt_loops_per_seed

    schemes = list(SCHEMES)
    nw = max(1, int(args.workers))
    _ensure(args.out)
    print(f"[run] {num_seeds} seeds, rt_loops={rt_loops}, workers={nw}, "
          f"schemes={schemes}", flush=True)

    if args.only_min_rate_cdf:
        run_min_rate_cdf_point(cfg, num_seeds, rt_loops, args.out,
                               n_workers=nw, **MIN_RATE_CDF_POINT)
        return

    if args.only_cdf:
        run_cdf_point(cfg, num_seeds, rt_loops, schemes, args.out,
                      n_workers=nw, **CDF_POINT)
        return

    if not args.no_tau:
        run_tau_p_sweep(cfg, num_seeds, rt_loops, schemes, args.out, n_workers=nw)
    if not args.no_K:
        run_K_sweep(cfg, num_seeds, rt_loops, schemes, args.out, n_workers=nw)
    if not args.no_L:
        run_L_sweep(cfg, num_seeds, rt_loops, schemes, args.out, n_workers=nw)
    if not args.no_cdf:
        run_cdf_point(cfg, num_seeds, rt_loops, schemes, args.out,
                      n_workers=nw, **CDF_POINT)
    if not args.no_min_rate_cdf:
        run_min_rate_cdf_point(cfg, num_seeds, rt_loops, args.out,
                               n_workers=nw, **MIN_RATE_CDF_POINT)


if __name__ == "__main__":
    main()
