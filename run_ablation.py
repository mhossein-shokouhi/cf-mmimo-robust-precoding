"""Ablation table driver (paper table at K=24, L=25, tau_p=4).

Evaluates the four schemes that isolate the effect of the precoder
(robust WMMSE vs CF-WMMSE) and the pilot-assignment family (proposed
heuristic vs MA-DRL PA) one factor at a time:

  1. greedy+robust     = Proposed Algorithm
  2. naive+robust      = Robust WMMSE + MA-DRL PA
  3. greedy+oblivious  = CF-WMMSE     + Proposed PA
  4. naive+oblivious   = CF-WMMSE     + MA-DRL PA

Uses the canonical 10-seed list from `config.SEEDS`. Cached naive-PA
agents (per `(K, tau_p, L, seed)`) are reused if already trained, so
this script is cheap to re-run after the standard sweeps.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import List

# Cap BLAS threads BEFORE importing numpy (so workers don't oversubscribe).
_BLAS_ENV_KEYS = ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                  "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS")
for _k in _BLAS_ENV_KEYS:
    os.environ.setdefault(_k, "1")

import numpy as np

from config import DEFAULT_CONFIG, SEEDS, SimConfig
from simulator import evaluate_all


SCHEMES = (
    "greedy+robust",
    "naive+robust",
    "greedy+oblivious",
    "naive+oblivious",
)

_LABELS = {
    "greedy+robust":    ("Proposed Algorithm",         "Robust WMMSE", "Proposed (heuristic)"),
    "naive+robust":     ("Robust WMMSE, MA-DRL PA",    "Robust WMMSE", "MA-DRL"),
    "greedy+oblivious": ("CF-WMMSE, Proposed PA",      "CF-WMMSE",     "Proposed (heuristic)"),
    "naive+oblivious":  ("CF-WMMSE, MA-DRL PA",        "CF-WMMSE",     "MA-DRL"),
}


def run(cfg: SimConfig,
        schemes: List[str],
        seeds: List[int],
        rt_loops: int,
        n_workers: int,
        out_md: str) -> None:
    print(f"[ablation] schemes={schemes}", flush=True)
    print(f"[ablation] K={cfg.K}, L={cfg.L}, tau_p={cfg.tau_p}, "
          f"rt_loops={rt_loops}, seeds={seeds} (n={len(seeds)}), "
          f"workers={n_workers}", flush=True)
    t0 = time.time()
    res = evaluate_all(cfg, schemes, seeds, rt_loops,
                       progress=True,
                       models_dir=cfg.models_dir,
                       n_workers=n_workers)
    thr = res["throughput"]  # (S, n_seeds)
    elapsed = time.time() - t0
    print(f"[ablation] done in {elapsed:.1f} s", flush=True)

    means = thr.mean(axis=1)
    stds  = thr.std(axis=1, ddof=1) if thr.shape[1] > 1 else np.zeros_like(means)

    # Use CF-WMMSE, MA-DRL PA as the baseline for the Δ column (the
    # "all components dropped" corner of the 2x2 ablation grid).
    base_name = "naive+oblivious"
    bi = schemes.index(base_name)
    base_per_seed = thr[bi]

    rows = []
    rows.append(("| Scheme | Precoder | Pilot Assignment | "
                 "Mean throughput (bits/s/Hz) | Std | Δ vs *CF-WMMSE, MA-DRL PA* |"))
    rows.append("| --- | --- | --- | ---: | ---: | --- |")
    for i, sch in enumerate(schemes):
        label, prec, pa = _LABELS[sch]
        m, s = means[i], stds[i]
        if i == bi:
            delta_str = "(baseline)"
        else:
            diff = thr[i] - base_per_seed
            n = diff.size
            mean_diff = diff.mean()
            ratio = 100.0 * (means[i] - means[bi]) / max(means[bi], 1e-12)
            std_diff = diff.std(ddof=1) if n > 1 else 0.0
            se_diff = std_diff / np.sqrt(n) if n > 1 else 0.0
            t_stat = mean_diff / se_diff if se_diff > 0 else float("nan")
            ci_pct = 100.0 * 1.96 * se_diff / max(means[bi], 1e-12)
            delta_str = (f"+{ratio:.2f}% (CI ±{ci_pct:.2f}%, "
                         f"t={t_stat:5.2f})")
        rows.append(f"| {label} | {prec} | {pa} | "
                    f"{m:7.3f} | {s:6.3f} | {delta_str} |")

    md = (
        f"## Ablation at $K={cfg.K}$, $L={cfg.L}$, "
        f"$\\tau_\\mathrm{{p}}={cfg.tau_p}$ ({len(seeds)} seeds)\n\n"
        "Gain column: point estimate is "
        "$(\\bar{A}-\\bar{B})/\\bar{B}\\times 100$ (table-consistent); "
        "95% CI and $t$-statistic are computed from the paired difference "
        "$A_i - B_i$.\n\n"
        + "\n".join(rows) + "\n"
    )
    print("\n" + md, flush=True)

    if out_md:
        os.makedirs(os.path.dirname(out_md) or ".", exist_ok=True)
        with open(out_md, "w") as f:
            f.write(md)
        print(f"[ablation] wrote {out_md}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int,
                        default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--out", type=str,
                        default=os.path.join(DEFAULT_CONFIG.results_dir,
                                             "ablation_table.md"))
    parser.add_argument("--rt-loops", type=int,
                        default=DEFAULT_CONFIG.rt_loops_per_seed)
    args = parser.parse_args()

    cfg = DEFAULT_CONFIG.copy_with(K=24, L=25, tau_p=4)
    run(cfg=cfg,
        schemes=list(SCHEMES),
        seeds=list(SEEDS),
        rt_loops=int(args.rt_loops),
        n_workers=int(args.workers),
        out_md=args.out)


if __name__ == "__main__":
    main()
