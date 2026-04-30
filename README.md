# Pilot Assignment and Robust Precoding for Cell-Free Massive MIMO in O-RAN

Reference implementation for the paper:

> **Pilot Assignment and Robust Precoding for Cell-Free Massive MIMO in O-RAN**
> Mohammad Hossein Shokouhi and Vincent W. S. Wong
> Submitted to *IEEE GLOBECOM 2026*

This repository simulates downlink cell-free massive MIMO under an O-RAN control
architecture. Users reuse a limited pilot codebook, so the O-RUs only have
imperfect LMMSE channel estimates. The central idea is to make the precoder
aware of that uncertainty: the proposed robust WMMSE update carries the
per-link channel-estimation error covariance through the receiver, weight, and
precoder updates instead of treating estimated CSI as perfect. The code also
implements pilot-assignment controllers, DRL baselines, mobility experiments,
minimum-rate dual weighting, and plotting scripts for reproducing the paper
figures.

The implementation is intentionally lightweight: Python + NumPy for the
simulator and DRL agents, Matplotlib for standard figures, and optional MATLAB
scripts for camera-ready plot styling.

---

## Quick Start

### 1. Create the environment

```bash
git clone https://github.com/mhossein-shokouhi/cf-mmimo-robust-precoding.git
cd cf-mmimo-robust-precoding

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Dependencies are deliberately small:

- `numpy>=1.24`
- `matplotlib>=3.7`
- `tqdm>=4.65`

### 2. Run a smoke test

```bash
python run_simulations.py --smoke --only-cdf
python plot_results.py
```

This checks the full simulation-to-figure path with a tiny number of seeds and
RT loops. It overwrites the corresponding cache in `results/` and regenerates
any available figures in `figures/`.

### 3. Reproduce the main cached experiments

```bash
python run_simulations.py --workers 4
python plot_results.py
```

`run_simulations.py` produces the numerical `.npz` archives under `results/`.
`plot_results.py` renders the paper-style PDFs under `figures/`.

The first clean run may take longer than later runs because the simplified
naive-DRL baseline trains and caches per-topology checkpoints under
`models/naive_K{K}_taup{tau_p}_L{L}_seed{seed}.npz`. Those files are generated
automatically and are ignored by git.

---

## What Gets Generated

The default simulation driver evaluates the configured schemes over the
canonical seed list in `config.SEEDS` and writes:

| Command output | Figure / purpose |
| --- | --- |
| `results/tau_p_sweep.npz` | Aggregate throughput vs. pilot length `tau_p` |
| `results/K_sweep.npz` | Aggregate throughput vs. number of users `K` |
| `results/L_sweep.npz` | Aggregate throughput vs. number of O-RUs `L` |
| `results/cdf_point.npz` | Per-user spectral-efficiency samples at `tau_p=4, K=24, L=25` |
| `results/min_rate_cdf.npz` | Proposed-only per-user average-rate CDF for several `R_min` targets |
| `results/ablation_table.md` | Paired ablation table with means, confidence intervals, and t-statistics |
| `results/mobility_sweep.npz` | Throughput vs. user velocity for heuristic/proposed-DRL/naive-DRL PA |

`plot_results.py` converts the cached results into:

| Figure | Source cache |
| --- | --- |
| `figures/fig_tau_p_sweep.pdf` | `results/tau_p_sweep.npz` |
| `figures/fig_tau_p_ablation.pdf` | `results/tau_p_sweep.npz` |
| `figures/fig_K_sweep.pdf` | `results/K_sweep.npz` |
| `figures/fig_L_sweep.pdf` | `results/L_sweep.npz` |
| `figures/fig_cdf.pdf` | `results/cdf_point.npz` |
| `figures/fig_min_rate_cdf.pdf` | `results/min_rate_cdf.npz` |
| `figures/fig_mobility.pdf` | `results/mobility_sweep.npz` |

---

## Running Specific Experiments

### Static sweeps and CDFs

The main driver supports selective runs:

```bash
python run_simulations.py --no-K --no-L --no-cdf --no-min-rate-cdf
python run_simulations.py --only-cdf
python run_simulations.py --only-min-rate-cdf
python run_simulations.py --smoke
python run_simulations.py --workers 8 --num-seeds 10
```

Useful flags:

- `--no-tau`, `--no-K`, `--no-L`, `--no-cdf`, `--no-min-rate-cdf` skip parts
  of the default pipeline.
- `--only-cdf` runs only the per-user CDF operating point.
- `--only-min-rate-cdf` runs only the proposed min-rate CDF.
- `--workers N` parallelizes the `(scheme, seed)` grid and missing naive-DRL
  checkpoint training.
- `--num-seeds N` overrides the default number of seeds. If `N` is smaller
  than the canonical seed list, the first `N` canonical seeds are used.

### Ablation table

```bash
python run_ablation.py --workers 4
```

This evaluates the four-factor ablation at `K=24, L=25, tau_p=4`:

- robust WMMSE + proposed heuristic PA
- robust WMMSE + naive/MA-DRL-style PA
- CF-WMMSE + proposed heuristic PA
- CF-WMMSE + naive/MA-DRL-style PA

The table is written to `results/ablation_table.md`.

### DRL training and mobility evaluation

Train the proposed rich-observation Dueling DDQN and the naive DQN baseline:

```bash
python train_drl.py --quick
python train_drl.py --episodes 150
```

Then evaluate throughput under moving users:

```bash
python mobility_eval.py --seeds 5
python plot_results.py
```

Additional options:

- `python train_drl.py --only proposed`
- `python train_drl.py --only naive`
- `python train_drl.py --train-v-max 3`
- `python mobility_eval.py --velocities 0 5 15 30 60 100`
- `python mobility_eval.py --schemes heuristic proposed_drl naive_drl`

The mobility experiment uses robust WMMSE for all schemes and varies only the
pilot-assignment control logic.

### Optional MATLAB figures

The `MATLAB_figures/` directory contains MATLAB scripts that load the same
NumPy `.npz` caches and export the camera-ready PDFs used for the paper:

```matlab
run MATLAB_figures/plot_tau_p_sweep_paper.m
run MATLAB_figures/plot_K_sweep_paper.m
run MATLAB_figures/plot_L_sweep_paper.m
run MATLAB_figures/plot_rate_cdf_paper.m
```

These scripts use `MATLAB_figures/npzload.m`, which requires MATLAB's Python
environment to have NumPy available.

---

## Algorithms Implemented

### Precoding

All precoders share the signature:

```python
f(h_hat, err_var, users_of_oru, cfg, eta=None) -> v
```

| Key | Display name | Description |
| --- | --- | --- |
| `robust` | Robust WMMSE | Proposed WMMSE solver. Keeps the estimation-error covariance term `R_{k,l}=(beta_{k,l}-alpha_{k,l})I` in the `u_k`, `w_k`, and `v_{k,l}` updates. |
| `oblivious` | CF-WMMSE | Conventional WMMSE baseline that drops the error terms and treats `h_hat` as perfect CSI. |
| `rzf` | LP-RZF | Local partial regularized zero-forcing at each O-RU with equal power per served user. |
| `mrt` | MRT | Conjugate beamforming with equal per-O-RU power split. |

The WMMSE precoders enforce each O-RU power constraint through the
eigendecomposition plus bisection solver in `precoding._solve_v_oru`.

### Pilot assignment and DRL

| Key / script name | Role |
| --- | --- |
| `greedy` | Priority-score heuristic. It assigns pilots by reducing pilot conflict `C_k` and CSI uncertainty `U_k`; this is the proposed PA used by the static sweeps. |
| `random` | Random pilot assignment baseline. |
| `naive` | Simplified Oh-style DQN baseline. The state is only the flattened `K x tau_p` pilot matrix, and the action is a joint `(user, pilot)` reassignment. |
| `proposed_drl` | Mobility experiment scheme: heuristic initialization plus Dueling DDQN near-RT refinement for the priority-selected user. |
| `heuristic` | Mobility experiment scheme with heuristic pilot assignment only and no DRL refinement. |

The paper describes DRL-based pilot assignment. In this reproducibility code,
the main static figures call the greedy priority heuristic as the proposed PA
because it is deterministic, fast, and isolates the robust-precoding gain. The
DRL stack remains implemented and is exercised by `train_drl.py`,
`mobility_eval.py`, the naive baseline in `run_simulations.py`, and the
ablation script.

### Minimum-rate weighting

The main sweeps use the aggregate-throughput setting with zero minimum-rate
targets, so `eta_k=1` and `mu_k=0`. The extra proposed-only min-rate CDF enables
common `R_min` targets at `tau_p=8, K=24, L=64`. It updates projected dual
weights from smoothed observed rates and passes `eta=1+mu` into robust WMMSE.

---

## Repository Map

| Path | Purpose |
| --- | --- |
| `config.py` | `SimConfig`, physical-layer constants, sweep ranges, seed list, scheme list. |
| `channel.py` | Deployment, 3GPP TR 36.814 UMi-NLOS path loss, user-centric clustering, Rayleigh fading, LMMSE estimation, pilot-conflict matrix. |
| `pilot_assignment.py` | Greedy PA, random PA, priority-user selection, DRL observation builders, action helpers. |
| `precoding.py` | Robust WMMSE, CF-WMMSE, LP-RZF, MRT, and the per-O-RU Lagrange multiplier solver. |
| `metrics.py` | True-channel downlink rate evaluation and robust lower-bound rate evaluation. |
| `mobility.py` | Constant-speed user motion with elastic boundary reflection and topology refresh. |
| `drl.py` | Pure-NumPy MLP, replay buffer, vanilla DQN, and Dueling DDQN with action masking. |
| `simulator.py` | Monte Carlo evaluation, min-rate dual loop, naive-DRL training/eval helpers, mobility-aware O-RAN control loop. |
| `run_simulations.py` | Main tau-p, K, L, CDF, and min-rate CDF driver. |
| `run_ablation.py` | Four-corner ablation table driver. |
| `train_drl.py` | Offline training for proposed and naive DRL pilot-assignment agents. |
| `mobility_eval.py` | Velocity sweep for heuristic, proposed DRL, and naive DRL pilot assignment. |
| `plot_results.py` | Matplotlib renderer for all cached Python results. |
| `MATLAB_figures/` | Optional MATLAB renderers and helpers for paper-style PDFs. |
| `results/` | Cached numerical experiment outputs. |
| `figures/` | Generated PDF figures. |
| `models/proposed.npz` | Trained proposed Dueling DDQN checkpoint used by mobility evaluation. |

---

## Default Simulation Parameters

All defaults live in `config.py`.

| Parameter | Default |
| --- | --- |
| Users `K` | 24 |
| O-RUs `L` | 25 |
| Antennas per O-RU `N_t` | 4 |
| User-centric cluster size `L_max` | 8 |
| Area | 500 m x 500 m |
| Carrier frequency | 2 GHz |
| Path loss | 3GPP TR 36.814 UMi-NLOS |
| Coherence block `tau_c` | 200 symbols |
| Pilot length `tau_p` | 4 |
| Downlink power per O-RU | 30 dBm |
| Uplink pilot power | 20 dBm |
| Noise variance | -114 dBm |
| Seeds x RT loops per point | 10 x 40 |
| WMMSE outer iterations / BCD sweeps | 25 / 2 |

Sweep defaults:

```python
TAU_P_SWEEP = (4, 8, 12, 16, 20, 24)
K_SWEEP = (8, 12, 16, 20, 24, 28)
L_SWEEP = (16, 25, 36, 49, 64)
CDF_POINT = {"tau_p": 4, "K": 24, "L": 25}
MIN_RATE_CDF_POINT = {"tau_p": 8, "K": 24, "L": 64}
```

Change an operating point by editing `SimConfig` or by using the existing
driver flags; every script reads from the same configuration object.

---

## Scheme Keys

The static result files store schemes as `{pilot}+{precoder}` keys:

| Key | Meaning |
| --- | --- |
| `greedy+robust` | Proposed Algorithm: greedy priority PA + robust WMMSE |
| `greedy+oblivious` | Proposed PA + CF-WMMSE |
| `naive+oblivious` | Naive/MA-DRL-style PA + CF-WMMSE |
| `random+oblivious` | Random PA + CF-WMMSE |
| `greedy+rzf` | Proposed PA + LP-RZF |
| `greedy+mrt` | Proposed PA + MRT |
| `naive+robust` | Naive/MA-DRL-style PA + robust WMMSE, used in `run_ablation.py` |

To add a new precoder, implement a function in `precoding.py` and register it
in `PRECODERS`. To add a new pilot-assignment policy, extend
`pilot_assignment.assign` and include the new `{pilot}+{precoder}` key in the
driver you want to evaluate.

---

## Reproducibility Notes

- All topologies and stochastic streams are derived deterministically from the
  seed and role-specific offsets.
- The canonical seed list is fixed in `config.SEEDS`.
- BLAS thread counts are capped before NumPy imports in the experiment drivers
  to avoid worker oversubscription.
- Cached `.npz` results and generated paper PDFs are included for inspection
  and fast re-plotting.
- Naive per-topology DRL checkpoints are intentionally ignored because they are
  large and regenerated automatically when missing.

---

## Citation

```bibtex
@inproceedings{shokouhi2026pilot,
  title     = {Pilot Assignment and Robust Precoding for Cell-Free Massive {MIMO} in {O-RAN}},
  author    = {Shokouhi, Mohammad Hossein and Wong, Vincent W. S.},
  booktitle = {Proc. IEEE Global Communications Conference (GLOBECOM)},
  year      = {2026}
}
```
