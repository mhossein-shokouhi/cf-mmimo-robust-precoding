"""User mobility model.

Each user is given a constant speed (in m/s) and a random heading. Positions
are advanced by `dt` seconds each near-RT loop (paper Section IV uses
`T_RT = 1 ms` and `N_RT = 10` RT loops per near-RT loop, so a near-RT loop is
10 ms). Users that hit the rectangular boundary of the coverage area
elastically reflect, which keeps the deployment density and large-scale
fading distribution stationary across velocities.

The mobility model is intentionally simple: the *point* of the figure is
that observation expressiveness — not the specific motion model — is what
lets the proposed DRL agent track the channel statistics across moving
users.
"""

from __future__ import annotations

import numpy as np

from channel import (Topology, large_scale_fading, user_centric_clusters)
from config import SimConfig


def kmh_to_mps(v_kmh: float) -> float:
    return v_kmh / 3.6


def random_velocity_vectors(K: int, speed_mps: float,
                            rng: np.random.Generator) -> np.ndarray:
    """Per-user velocity (K, 2) with random heading and a constant speed."""
    if speed_mps <= 0.0:
        return np.zeros((K, 2))
    theta = rng.uniform(0.0, 2.0 * np.pi, size=K)
    return speed_mps * np.column_stack([np.cos(theta), np.sin(theta)])


def step_positions(user_pos: np.ndarray,
                   user_vel: np.ndarray,
                   dt: float,
                   cfg: SimConfig) -> np.ndarray:
    """Advance positions by `dt` and reflect on the rectangular boundary."""
    pos = user_pos.copy()
    pos[:, 0] += user_vel[:, 0] * dt
    pos[:, 1] += user_vel[:, 1] * dt

    # Elastic reflection on x.
    over_x = pos[:, 0] > cfg.area_x
    under_x = pos[:, 0] < 0.0
    pos[over_x, 0] = 2.0 * cfg.area_x - pos[over_x, 0]
    user_vel[over_x, 0] *= -1.0
    pos[under_x, 0] = -pos[under_x, 0]
    user_vel[under_x, 0] *= -1.0

    # Elastic reflection on y.
    over_y = pos[:, 1] > cfg.area_y
    under_y = pos[:, 1] < 0.0
    pos[over_y, 1] = 2.0 * cfg.area_y - pos[over_y, 1]
    user_vel[over_y, 1] *= -1.0
    pos[under_y, 1] = -pos[under_y, 1]
    user_vel[under_y, 1] *= -1.0

    return pos


def update_topology_after_motion(topology: Topology,
                                 new_user_xy: np.ndarray,
                                 cfg: SimConfig) -> Topology:
    """Recompute beta and the user-centric clusters after the users moved.

    The O-RU positions are kept fixed; only `user_pos`, `beta`, and the
    cluster sets are refreshed.
    """
    user_pos = topology.user_pos.copy()
    user_pos[:, :2] = new_user_xy
    beta = large_scale_fading(user_pos, topology.oru_pos, cfg)
    serving_oru, users_of_oru = user_centric_clusters(beta, cfg.L_max)
    return Topology(user_pos=user_pos,
                    oru_pos=topology.oru_pos,
                    beta=beta,
                    serving_oru=serving_oru,
                    users_of_oru=users_of_oru)
