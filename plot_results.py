"""Produce paper-ready figures from the cached simulation results."""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from config import DEFAULT_CONFIG

SCHEME_PLOT_STYLES = {
    "greedy+robust":    {"label": "Proposed (greedy + robust)",      "marker": "o", "linestyle": "-",  "linewidth": 2.0, "color": "#1f77b4"},
    "greedy+oblivious": {"label": "Greedy pilot + oblivious WMMSE",  "marker": "s", "linestyle": "--", "linewidth": 1.6, "color": "#ff7f0e"},
    "random+robust":    {"label": "Random pilot + robust WMMSE",     "marker": "^", "linestyle": "-.", "linewidth": 1.6, "color": "#2ca02c"},
    "random+oblivious": {"label": "Random pilot + oblivious WMMSE",  "marker": "D", "linestyle": ":",  "linewidth": 1.6, "color": "#d62728"},
    "greedy+mrt":       {"label": "Greedy pilot + MRT",              "marker": "v", "linestyle": "-",  "linewidth": 1.4, "color": "#9467bd"},
}


_SWEEP_TITLES = {
    "tau_ps": (r"Aggregate throughput vs. $\tau_\mathrm{{p}}$ "
               r"($K={K}$, $L={L}$)"),
    "Ks":     (r"Aggregate throughput vs. $K$ "
               r"($\tau_\mathrm{{p}}={tau_p}$, $L={L}$)"),
    "Ls":     (r"Aggregate throughput vs. $L$ "
               r"($\tau_\mathrm{{p}}={tau_p}$, $K={K}$)"),
}

_ABLATION_TITLES = {
    "tau_ps": r"Ablation study ($K={K}$, $L={L}$)",
    "Ks":     r"Ablation study ($\tau_\mathrm{{p}}={tau_p}$, $L={L}$)",
    "Ls":     r"Ablation study ($\tau_\mathrm{{p}}={tau_p}$, $K={K}$)",
}


def _format_title(template: str, data) -> str:
    """Substitute the fixed-parameter placeholders from the npz metadata."""
    fields = {}
    for key in ("K", "L", "tau_p", "N_t"):
        if key in data.files:
            fields[key] = int(data[key])
    return template.format(**fields)


def _plot_sweep(path: str,
                x_key: str,
                x_label: str,
                figure_path: str) -> None:
    data = np.load(path, allow_pickle=True)
    schemes = [str(s) for s in data["schemes"]]
    x = data[x_key]
    mean = data["throughput"].mean(axis=-1)

    plt.figure(figsize=(6.5, 4.2))
    for i, sch in enumerate(schemes):
        style = SCHEME_PLOT_STYLES.get(sch, {"label": sch, "marker": "o", "linestyle": "-", "color": None})
        plt.plot(x, mean[i],
                 label=style["label"],
                 marker=style["marker"],
                 linestyle=style["linestyle"],
                 linewidth=style.get("linewidth", 1.5),
                 color=style.get("color"))
    plt.xlabel(x_label)
    plt.ylabel("Aggregate throughput (bits/s/Hz)")
    plt.title(_format_title(_SWEEP_TITLES[x_key], data))
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(figure_path)
    plt.close()
    print(f"figure saved -> {figure_path}")


def _plot_ablation(path: str,
                   x_key: str,
                   x_label: str,
                   figure_path: str) -> None:
    """Subset of schemes showing the ablation (proposed vs. dropping components)."""
    data = np.load(path, allow_pickle=True)
    schemes = [str(s) for s in data["schemes"]]
    x = data[x_key]
    mean = data["throughput"].mean(axis=-1)

    ablation_order = ["greedy+robust", "greedy+oblivious", "random+robust", "random+oblivious"]
    plt.figure(figsize=(6.5, 4.2))
    for sch in ablation_order:
        if sch not in schemes:
            continue
        i = schemes.index(sch)
        style = SCHEME_PLOT_STYLES[sch]
        plt.plot(x, mean[i],
                 label=style["label"],
                 marker=style["marker"],
                 linestyle=style["linestyle"],
                 linewidth=style.get("linewidth", 1.5),
                 color=style.get("color"))
    plt.xlabel(x_label)
    plt.ylabel("Aggregate throughput (bits/s/Hz)")
    plt.title(_format_title(_ABLATION_TITLES[x_key], data))
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(figure_path)
    plt.close()
    print(f"figure saved -> {figure_path}")


CDF_MAIN_SCHEMES = ("greedy+robust",
                    "greedy+oblivious",
                    "random+oblivious",
                    "greedy+mrt")


def _plot_cdf(path: str,
              figure_path: str,
              schemes_subset=CDF_MAIN_SCHEMES) -> None:
    """Empirical CDF of per-user spectral efficiency."""
    data = np.load(path, allow_pickle=True)
    schemes = [str(s) for s in data["schemes"]]
    rates = data["rates"]  # (S, n_seeds, rt_loops, K)
    tau_p = int(data["tau_p"]); K = int(data["K"]); L = int(data["L"])

    plt.figure(figsize=(6.5, 4.2))
    for sch in schemes_subset:
        if sch not in schemes:
            continue
        i = schemes.index(sch)
        samples = rates[i].ravel()
        samples = samples[np.isfinite(samples)]
        if samples.size == 0:
            continue
        x_sorted = np.sort(samples)
        y = (np.arange(1, x_sorted.size + 1)) / x_sorted.size
        style = SCHEME_PLOT_STYLES.get(sch, {"label": sch, "linestyle": "-", "color": None})
        p05, p50, p95 = np.percentile(samples, [5, 50, 95])
        label = f"{style['label']} (5%/50%/95%: {p05:.2f}/{p50:.2f}/{p95:.2f})"
        plt.plot(x_sorted, y,
                 label=label,
                 linestyle=style["linestyle"],
                 linewidth=style.get("linewidth", 1.8),
                 color=style.get("color"))

    plt.axhline(0.05, color="grey", linewidth=0.8, linestyle=":", alpha=0.6)
    plt.axhline(0.50, color="grey", linewidth=0.8, linestyle=":", alpha=0.6)
    plt.axhline(0.95, color="grey", linewidth=0.8, linestyle=":", alpha=0.6)
    plt.xlabel("Per-user spectral efficiency (bits/s/Hz)")
    plt.ylabel("Empirical CDF")
    plt.title(rf"Per-user SE CDF ($\tau_\mathrm{{p}}={tau_p}$, $K={K}$, $L={L}$)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(figure_path)
    plt.close()
    print(f"figure saved -> {figure_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=DEFAULT_CONFIG.results_dir)
    parser.add_argument("--out", default=DEFAULT_CONFIG.figures_dir)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    tau_path = os.path.join(args.results, "tau_p_sweep.npz")
    K_path = os.path.join(args.results, "K_sweep.npz")
    L_path = os.path.join(args.results, "L_sweep.npz")
    cdf_path = os.path.join(args.results, "cdf_point.npz")

    if os.path.exists(tau_path):
        _plot_sweep(tau_path, "tau_ps", r"Number of pilots $\tau_\mathrm{p}$",
                    os.path.join(args.out, "fig_tau_p_sweep.pdf"))
        _plot_ablation(tau_path, "tau_ps", r"Number of pilots $\tau_\mathrm{p}$",
                       os.path.join(args.out, "fig_tau_p_ablation.pdf"))
    else:
        print(f"missing {tau_path}")

    if os.path.exists(K_path):
        _plot_sweep(K_path, "Ks", r"Number of users $K$",
                    os.path.join(args.out, "fig_K_sweep.pdf"))
    else:
        print(f"missing {K_path}")

    if os.path.exists(L_path):
        _plot_sweep(L_path, "Ls", r"Number of O-RUs $L$",
                    os.path.join(args.out, "fig_L_sweep.pdf"))
    else:
        print(f"missing {L_path}")

    if os.path.exists(cdf_path):
        _plot_cdf(cdf_path, os.path.join(args.out, "fig_cdf.pdf"))
    else:
        print(f"missing {cdf_path}")


if __name__ == "__main__":
    main()
