"""Simulation configuration.

All parameters are collected in a single dataclass so that each simulation
run can be fully described by a `SimConfig` instance. Defaults match the
scenario in Section V of `ORAN.tex`; the Checklist instructs us to explore
`tau_p in {4, 8, 16}` and to pick reasonable values for the rest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


def dbm_to_watt(x_dbm: float) -> float:
    return 10.0 ** ((x_dbm - 30.0) / 10.0)


@dataclass
class SimConfig:
    K: int = 24
    L: int = 25
    N_t: int = 4
    L_max: int = 8

    area_x: float = 500.0
    area_y: float = 500.0
    oru_height: float = 10.0
    user_height: float = 1.5
    f_c_ghz: float = 2.0
    min_distance_2d: float = 1.0

    P_max_dbm: float = 30.0
    p_ul_dbm: float = 20.0
    sigma2_dbm: float = -114.0

    tau_c: int = 200
    tau_p: int = 4

    wmmse_outer_iters: int = 25
    wmmse_tol: float = 1e-4
    bcd_sweeps_per_outer: int = 2
    lambda_max: float = 1e12
    lambda_bisect_iters: int = 60

    pilot_sweep_iters: int = 20
    priority_w_C: float = 1.0
    priority_w_U: float = 1.0

    rt_loops_per_seed: int = 40
    num_seeds: int = 10
    smoke_seeds: int = 2
    smoke_rt_loops: int = 8

    results_dir: str = "results"
    figures_dir: str = "figures"

    @property
    def P_max(self) -> float:
        return dbm_to_watt(self.P_max_dbm)

    @property
    def p_ul(self) -> float:
        return dbm_to_watt(self.p_ul_dbm)

    @property
    def sigma2(self) -> float:
        return dbm_to_watt(self.sigma2_dbm)

    @property
    def tau_d(self) -> int:
        return self.tau_c - self.tau_p

    def copy_with(self, **overrides) -> "SimConfig":
        data = self.__dict__.copy()
        data.update(overrides)
        return SimConfig(**data)


DEFAULT_CONFIG = SimConfig()

TAU_P_SWEEP: Tuple[int, ...] = (4, 8, 12, 16, 20, 24)
K_SWEEP: Tuple[int, ...] = (8, 12, 16, 20, 24, 28)
L_SWEEP: Tuple[int, ...] = (16, 25, 36, 49, 64)

CDF_POINT = {"tau_p": 4, "K": 24, "L": 25}

SCHEMES = (
    "greedy+robust",
    "greedy+oblivious",
    "random+oblivious",
    "greedy+mrt",
)

PROPOSED = "greedy+robust"
