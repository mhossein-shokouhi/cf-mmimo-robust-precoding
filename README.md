# Robust Precoding for Cell-Free Massive MIMO in O-RAN

Reference implementation and simulation code for:

> **Robust Precoding Using Channel Estimation Error Statistics for Cell-Free
> Massive MIMO in O-RAN**
> Mohammad Hossein Shokouhi and Vincent W. S. Wong
> Submitted to *IEEE GLOBECOM 2026*

---

## What this paper is about

The paper studies the downlink of a cell-free massive MIMO system deployed
through an O-RAN architecture. \(K\) single-antenna users are jointly served by
\(L\) geographically distributed open radio units (O-RUs) over a small pilot
codebook of length \(\tau_\mathrm{p}\). When \(K > \tau_\mathrm{p}\), users have
to share pilots and **pilot contamination** leaves every channel estimate
imperfect. Conventional cell-free WMMSE precoders (CF-WMMSE) simply ignore this
fact and treat the LMMSE estimate \(\hat{\mathbf{h}}_{k,l}\) as ground truth.

### Core contribution: robust precoding (Section III-A)

The central technical contribution of the paper is a **robust WMMSE precoder
that explicitly carries the per-link channel-estimation error covariance**
\(\mathbf{R}_{k,l}=(\beta_{k,l}-\alpha_{k,l})\,\mathbf{I}_{N_\mathrm{t}}\)
through every step of the WMMSE block-coordinate-descent (BCD) update:

1. A closed-form lower bound on the ergodic rate is derived (Lemma 1) so the
   precoder optimisation becomes tractable per-coherence-block.
2. The receiver MMSE weights \(u_k\), the WMMSE auxiliary variables \(w_k\),
   and the per-O-RU precoders \(\mathbf{v}_{k,l}\) are all updated **with the
   \(\mathbf{R}_{k,l}\) terms retained**.
3. The per-O-RU power constraint is enforced exactly via a closed-form
   \(\lambda_\ell\) Lagrangian solved by an eigendecomposition + bisection
   search.

The net effect is that, in the WMMSE iterations, users with reliable CSI
demand correspondingly less interference suppression effort than users with
noisy estimates — which provably outperforms the CF-WMMSE design in
pilot-contaminated regimes. In our simulations, swapping CF-WMMSE for the
proposed robust WMMSE — while keeping every other knob fixed — yields a
**paired aggregate-throughput gain of +11.77% (95% CI ±1.14%,
\(t_9 = 20.2,\ p < 10^{-3}\))** at the loaded operating point
\(\tau_\mathrm{p}=4,\ K=24,\ L=25\), and **+9.7% at the median per-user
spectral efficiency** in the CDF. Both linear baselines (LP-RZF and MRT)
fall well below either WMMSE flavor.

The paper also contains a multi-agent DRL pilot-assignment component, but the
headline gain comes from the robust precoder — and this repository focuses on
making that contribution easy to inspect, run, and reproduce.

---

## Repository layout

| File | Purpose |
| ---- | ------- |
| `config.py`            | Single dataclass holding every physical-layer / simulation parameter, plus the predefined sweeps. |
| `channel.py`           | 3GPP TR 36.814 UMi-NLOS path loss, user-centric clustering, complex-Gaussian small-scale fading, LMMSE estimator, pilot-conflict matrix. |
| `pilot_assignment.py`  | Greedy priority-score heuristic (the **Proposed PA**) and the random-assignment baseline. |
| `precoding.py`         | **Robust WMMSE (proposed)**, CF-WMMSE (ablation), Local Partial RZF, MRT (linear baselines), closed-form \(\lambda_\ell\) solver. |
| `metrics.py`           | Per-user rate / aggregate throughput evaluated against the **true** channel realisations. |
| `simulator.py`         | Glue code: turns a `(scheme, seed)` pair into a Monte-Carlo estimate (and per-user rate samples for the CDF). |
| `run_simulations.py`   | Top-level driver: \(\tau_\mathrm{p}\) sweep, \(K\) sweep, \(L\) sweep, CDF point. |
| `plot_results.py`      | Renders the paper PDFs from the cached `.npz` results. |
| `ORAN.tex`             | LaTeX source of the conference paper (cross-reference for equation numbers). |
| `TWC_Paper.pdf`        | Companion journal paper, included for the per-O-RU \(\lambda_\ell\) derivation reused here. |

The implementation is intentionally NumPy-only (no PyTorch, no SciPy). Every
algorithm in the paper fits in a few hundred lines that can be read top-to-bottom
against the published equations.

---

## Installation

```bash
git clone https://github.com/mhossein-shokouhi/cf-mmimo-robust-precoding.git
cd cf-mmimo-robust-precoding

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Dependencies (pinned in `requirements.txt`) are kept minimal:

- `numpy` ≥ 1.24
- `matplotlib` ≥ 3.7
- `tqdm` ≥ 4.65

Tested on Python 3.9 and 3.11, macOS (Apple Accelerate BLAS) and Linux.

---

## Reproducing the paper figures

Two commands:

```bash
python run_simulations.py     # tau_p, K, L sweeps + CDF point  (~30 min, single core)
python plot_results.py        # renders figures/*.pdf from cached .npz results
```

This produces:

| File | Figure |
| ---- | ------ |
| `figures/fig_tau_p_sweep.pdf`   | Aggregate throughput vs \(\tau_\mathrm{p}\) (main figure) |
| `figures/fig_tau_p_ablation.pdf`| Aggregate throughput vs \(\tau_\mathrm{p}\) (ablation) |
| `figures/fig_K_sweep.pdf`       | Aggregate throughput vs \(K\) |
| `figures/fig_L_sweep.pdf`       | Aggregate throughput vs \(L\) |
| `figures/fig_cdf.pdf`           | Per-user spectral-efficiency CDF at \(\tau_\mathrm{p}=4,\,K=24,\,L=25\) |
| `figures/fig_ablation.pdf`      | 4-cell ablation bar chart at the default operating point |
| `results/ablation_table.md`     | Markdown ablation table with paired Δ and 95% CIs |

Cached numerical results land in `results/*.npz`; you can re-render figures any
time without re-running the simulator.

### Quick smoke test

A 30-second sanity check (few seeds, few RT loops) is available with:

```bash
python run_simulations.py --smoke
```

### Running a single experiment

```bash
python run_simulations.py --no-K --no-L --no-cdf   # only tau_p sweep
python run_simulations.py --only-cdf               # only the CDF point (~50 s)
```

The same logic is exposed via `--no-tau`, `--no-K`, `--no-L`, `--no-cdf`.

---

## Schemes implemented

Display name convention: `{Precoder}, {PA}` — the precoder name comes first,
followed by the pilot-assignment (PA) policy. The combination of robust WMMSE
and the greedy priority-score PA is referred to as the **Proposed Algorithm**.

| Display name                  | Pilot assignment        | Precoder            | Internal key (in `.npz`) |
| ----------------------------- | ----------------------- | ------------------- | ------------------------ |
| **Proposed Algorithm**        | Greedy (priority score) | **Robust WMMSE**    | `greedy+robust`          |
| CF-WMMSE, Proposed PA         | Greedy (priority score) | CF-WMMSE            | `greedy+oblivious`       |
| CF-WMMSE, Random PA           | Random                  | CF-WMMSE            | `random+oblivious`       |
| RZF, Proposed PA              | Greedy (priority score) | Local Partial RZF   | `greedy+rzf`             |
| MRT, Proposed PA              | Greedy (priority score) | MRT (conjugate BF)  | `greedy+mrt`             |

The internal keys (`{pilot}+{precoder}`) are kept stable so cached
`results/*.npz` files remain readable across renames.

Adding a new precoder is a one-liner: implement a function with signature
`f(h_hat, err_var, users_of_oru, cfg) -> v` and register it in the `PRECODERS`
dict at the bottom of `precoding.py`.

---

## Default simulation parameters

(All knobs live in `config.py`.)

| Parameter | Default |
| --------- | ------- |
| Number of users \(K\)                | 24 |
| Number of O-RUs \(L\)                | 25 |
| Antennas per O-RU \(N_\mathrm{t}\)   | 4 |
| Cluster size \(L_\mathrm{max}\)      | 8 |
| Coverage area                        | 500 m × 500 m |
| Carrier frequency \(f_c\)            | 2 GHz |
| Path-loss model                      | 3GPP TR 36.814 UMi-NLOS |
| Coherence block \(\tau_\mathrm{c}\)  | 200 symbols |
| Pilot length \(\tau_\mathrm{p}\)     | 4 |
| Per-O-RU DL power \(P_\mathrm{DL}^\max\) | 30 dBm |
| User UL pilot power                  | 20 dBm |
| Noise variance                       | −114 dBm |
| Seeds × RT loops per point           | 10 × 40 |
| WMMSE outer iters / BCD sweeps       | 25 / 2 |

Sweep ranges are also defined there:

```python
TAU_P_SWEEP = (4, 8, 12, 16, 20, 24)
K_SWEEP     = (8, 12, 16, 20, 24, 28)
L_SWEEP     = (16, 25, 36, 49, 64)
CDF_POINT   = {"tau_p": 4, "K": 24, "L": 25}
```

To explore a different operating point, edit `config.py` and re-run — every
script reads the same `SimConfig`.

---

## Notes on simplifications

Per the project checklist (`Checklist.txt`), three simplifications are applied
relative to the paper:

1. **Pilot assignment.** The multi-agent DRL algorithm of Section III-B is
   replaced with a greedy priority-score heuristic (`pilot_assignment.py`),
   referred to as the **Proposed PA** in all figures. It uses the same score
   \(\rho_k = w_C C_k + w_U U_k\) (eq. (15) of the paper) but selects pilots
   greedily. The headline contribution — the robust precoder — is by design
   independent of which pilot-assignment policy sits on top, and the ablation
   table confirms this (compare *Proposed Algorithm* vs *CF-WMMSE, Random PA*).
2. **Minimum-rate constraints.** \(\eta_k = 1,\ \mu_k = 0\) throughout. The
   framework supports them but they are turned off here.
3. **Per-O-RU \(\lambda_\ell\) solver.** Computed in closed form via
   eigendecomposition + bisection following the companion TWC paper
   (`TWC_Paper.pdf`, eqs. (26)–(28)). This is correct in our setting because
   \(\lambda_\ell\) is the dual of a constraint that is invariant to the
   inclusion of \(\mathbf{R}_{k,l}\) on the diagonal of \(\mathbf{A}_\ell\).

These simplifications affect neither the correctness of the algorithms nor the
qualitative conclusions of the paper; they speed up the experiments by roughly
an order of magnitude.

---

## Reproducibility checklist

- [x] Fixed seeds (`1234, 1235, …`) — see `run_simulations.py:_seeds`.
- [x] All RNGs are derived deterministically from `(seed, role)` so topology,
      pilot, channel, and noise streams are independent.
- [x] All cached results (`results/*.npz`) and figures (`figures/*.pdf`) are
      committed.
- [x] No proprietary dependencies.
- [x] All physical-layer constants are explicit in `config.py`.

---

## Citation

```bibtex
@inproceedings{shokouhi2026robust,
  title     = {Robust Precoding Using Channel Estimation Error Statistics for
               Cell-Free Massive {MIMO} in {O-RAN}},
  author    = {Shokouhi, Mohammad Hossein and Wong, Vincent W. S.},
  booktitle = {Proc. IEEE Global Communications Conference (GLOBECOM)},
  year      = {2026},
}
```
