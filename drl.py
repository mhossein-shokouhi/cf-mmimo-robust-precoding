"""Tiny NumPy DDQN / DQN for pilot-assignment refinement.

Two agent flavours are implemented:

* `DuelingDQNAgent`  — the **proposed** agent (Section III-B of `ORAN.tex`).
  The network has a shared MLP trunk feeding a state-value head V(o) and
  an advantage head A(o, a). The action-value is recombined as
      Q(o, a) = V(o) + A(o, a) - mean_a' A(o, a').
  Trained with **double DQN**: action-selection on the online network,
  bootstrap value on the target network.

* `VanillaDQNAgent` — a deliberately minimalist baseline inspired by Oh
  *et al.* (JSAC 2024, `A_Decentralized_Pilot_Assignment_*.pdf`). The state
  here is **only** the K x tau_p one-hot pilot-assignment matrix (no rate,
  no path-loss, no CSI uncertainty). The architecture is a plain MLP →
  tau_p logits, trained with vanilla DQN.

Both agents share the same lightweight infrastructure: a 2-layer MLP with
ReLU and Adam, an experience replay buffer, and a periodic target sync.
The implementation is intentionally NumPy-only to match the rest of the
repository.

The motivation for the side-by-side comparison is to show that the rich
observation of the proposed agent — which exposes per-user rates, conflict
scores, and CSI uncertainty — keeps the policy useful under user mobility,
whereas the naive observation collapses outside the training distribution.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np

ForbiddenSpec = Union[int, Iterable[int], None]

# Accelerate BLAS on macOS occasionally emits spurious divide/overflow
# warnings from `matmul` when inputs span a wide dynamic range. The
# downstream gradient/update computations are numerically stable.
np.seterr(divide="ignore", over="ignore", invalid="ignore")


# ---------------------------------------------------------------------------
#  Tiny NumPy MLP with Adam.
# ---------------------------------------------------------------------------
def _he_init(rng: np.random.Generator, fan_in: int, fan_out: int) -> np.ndarray:
    return (rng.standard_normal((fan_in, fan_out))
            * np.sqrt(2.0 / max(fan_in, 1))).astype(np.float32)


@dataclass
class AdamState:
    m: np.ndarray
    v: np.ndarray
    t: int = 0

    @classmethod
    def for_param(cls, p: np.ndarray) -> "AdamState":
        return cls(m=np.zeros_like(p), v=np.zeros_like(p), t=0)


@dataclass
class MLP:
    """Plain MLP: input -> hidden(ReLU) -> ... -> output (linear).

    `head` lets us keep two parallel output heads (V, A) for the dueling
    network without duplicating the trunk. When `head_dims=(out,)`, the MLP
    is a regular feed-forward network with a single linear output.
    """
    layer_sizes: Tuple[int, ...]
    head_dims: Tuple[int, ...]
    rng: np.random.Generator
    Ws: List[np.ndarray] = field(default_factory=list)
    bs: List[np.ndarray] = field(default_factory=list)
    head_Ws: List[np.ndarray] = field(default_factory=list)
    head_bs: List[np.ndarray] = field(default_factory=list)

    def __post_init__(self):
        for fi, fo in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            self.Ws.append(_he_init(self.rng, fi, fo))
            self.bs.append(np.zeros(fo, dtype=np.float32))
        last = self.layer_sizes[-1]
        for h in self.head_dims:
            self.head_Ws.append(_he_init(self.rng, last, h))
            self.head_bs.append(np.zeros(h, dtype=np.float32))

    # --- forward / backward ------------------------------------------------
    def forward(self, x: np.ndarray):
        """Return (head_outputs_list, cache).

        `cache` carries the pre-activation and post-activation tensors of
        every layer so that `backward` can be a vanilla chain-rule pass.
        """
        a = x.astype(np.float32, copy=False)
        zs, acts = [], [a]
        for W, b in zip(self.Ws, self.bs):
            z = a @ W + b
            a = np.maximum(z, 0.0)
            zs.append(z)
            acts.append(a)
        outs = [a @ Wh + bh for Wh, bh in zip(self.head_Ws, self.head_bs)]
        cache = {"zs": zs, "acts": acts, "input": x}
        return outs, cache

    def backward(self, cache, dheads: List[np.ndarray]):
        """Backprop given gradients on each head output."""
        acts = cache["acts"]
        zs = cache["zs"]
        last_act = acts[-1]
        dW_heads, db_heads = [], []
        da_trunk = np.zeros_like(last_act)
        for Wh, dh in zip(self.head_Ws, dheads):
            dW_heads.append(last_act.T @ dh)
            db_heads.append(dh.sum(axis=0))
            da_trunk = da_trunk + dh @ Wh.T

        dWs = [None] * len(self.Ws)
        dbs = [None] * len(self.bs)
        da = da_trunk
        for i in reversed(range(len(self.Ws))):
            z = zs[i]
            relu_mask = (z > 0.0).astype(np.float32)
            dz = da * relu_mask
            a_prev = acts[i]
            dWs[i] = a_prev.T @ dz
            dbs[i] = dz.sum(axis=0)
            da = dz @ self.Ws[i].T
        return dWs, dbs, dW_heads, db_heads

    # --- Adam --------------------------------------------------------------
    def make_adam_state(self) -> dict:
        return {
            "Ws": [AdamState.for_param(W) for W in self.Ws],
            "bs": [AdamState.for_param(b) for b in self.bs],
            "head_Ws": [AdamState.for_param(W) for W in self.head_Ws],
            "head_bs": [AdamState.for_param(b) for b in self.head_bs],
        }

    def adam_step(self, grads, state, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        dWs, dbs, dW_heads, db_heads = grads
        params = [(self.Ws, dWs, state["Ws"]),
                  (self.bs, dbs, state["bs"]),
                  (self.head_Ws, dW_heads, state["head_Ws"]),
                  (self.head_bs, db_heads, state["head_bs"])]
        for ps, gs, sts in params:
            for i in range(len(ps)):
                if gs[i] is None:
                    continue
                st = sts[i]
                st.t += 1
                st.m = beta1 * st.m + (1.0 - beta1) * gs[i]
                st.v = beta2 * st.v + (1.0 - beta2) * (gs[i] * gs[i])
                m_hat = st.m / (1.0 - beta1 ** st.t)
                v_hat = st.v / (1.0 - beta2 ** st.t)
                ps[i] -= lr * m_hat / (np.sqrt(v_hat) + eps)

    # --- (de)serialisation -------------------------------------------------
    def state_dict(self) -> dict:
        sd = {}
        for i, W in enumerate(self.Ws):
            sd[f"W{i}"] = W
            sd[f"b{i}"] = self.bs[i]
        for i, W in enumerate(self.head_Ws):
            sd[f"head_W{i}"] = W
            sd[f"head_b{i}"] = self.head_bs[i]
        sd["layer_sizes"] = np.array(self.layer_sizes)
        sd["head_dims"] = np.array(self.head_dims)
        return sd

    def load_state_dict(self, sd: dict) -> None:
        for i in range(len(self.Ws)):
            self.Ws[i] = sd[f"W{i}"].astype(np.float32, copy=True)
            self.bs[i] = sd[f"b{i}"].astype(np.float32, copy=True)
        for i in range(len(self.head_Ws)):
            self.head_Ws[i] = sd[f"head_W{i}"].astype(np.float32, copy=True)
            self.head_bs[i] = sd[f"head_b{i}"].astype(np.float32, copy=True)


# ---------------------------------------------------------------------------
#  Replay buffer.
# ---------------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int):
        self.cap = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.size = 0
        self.idx = 0

    def push(self, o, a, r, op, d):
        i = self.idx
        self.obs[i] = o
        self.next_obs[i] = op
        self.actions[i] = a
        self.rewards[i] = r
        self.done[i] = float(d)
        self.idx = (i + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, batch_size: int, rng: np.random.Generator):
        idx = rng.integers(0, self.size, size=batch_size)
        return (self.obs[idx], self.actions[idx], self.rewards[idx],
                self.next_obs[idx], self.done[idx])


# ---------------------------------------------------------------------------
#  Agents.
# ---------------------------------------------------------------------------
@dataclass
class DQNConfig:
    hidden: Tuple[int, ...] = (64, 64)
    lr: float = 1e-3
    gamma: float = 0.5
    batch_size: int = 64
    buffer_capacity: int = 5000
    target_sync_every: int = 100
    train_every: int = 4
    min_buffer_for_train: int = 256
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 4000
    huber_delta: float = 1.0


class _BaseAgent:
    """Common plumbing for both DQN flavours."""

    is_dueling: bool = False
    # When True, both `select_action` (via the `forbidden` argument) and
    # the TD-target during `update()` mask the just-taken action. This
    # is used by the proposed Dueling DDQN, which must change the pilot
    # of `k_star` at every near-RT step rather than degenerate to a
    # no-op.
    mask_self_action: bool = False

    def __init__(self,
                 obs_dim: int,
                 n_actions: int,
                 cfg: DQNConfig,
                 rng: Optional[np.random.Generator] = None):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.cfg = cfg
        self.rng = rng or np.random.default_rng(0)
        self.online = self._build_net()
        self.target = self._build_net()
        self._copy_to_target()
        self.adam = self.online.make_adam_state()
        self.buffer = ReplayBuffer(cfg.buffer_capacity, obs_dim)
        self.train_step = 0
        self.action_step = 0

    # subclass hooks --------------------------------------------------------
    def _build_net(self) -> MLP:
        raise NotImplementedError

    def _q_from_heads(self, heads) -> np.ndarray:
        raise NotImplementedError

    # core ------------------------------------------------------------------
    def _copy_to_target(self):
        self.target.load_state_dict(self.online.state_dict())

    def epsilon(self) -> float:
        c = self.cfg
        frac = min(1.0, self.action_step / max(c.eps_decay_steps, 1))
        return c.eps_end + (c.eps_start - c.eps_end) * (1.0 - frac)

    @staticmethod
    def _forbidden_to_array(forbidden: ForbiddenSpec) -> Optional[np.ndarray]:
        if forbidden is None:
            return None
        if np.isscalar(forbidden):
            return np.asarray([int(forbidden)], dtype=np.int64)
        arr = np.asarray(list(forbidden), dtype=np.int64)
        return arr if arr.size > 0 else None

    def select_action(self,
                      obs: np.ndarray,
                      greedy: bool = False,
                      forbidden: ForbiddenSpec = None) -> int:
        """Pick an action via eps-greedy.

        `forbidden` (if not None) excludes those action indices from both
        the random and the greedy branches. Accepts either a single int
        (proposed Dueling-DDQN refinement: force-changing the pilot of
        `k_star`) or a sequence of ints (naive DQN: forbid every no-op
        `(k, t)` whose `t` already equals `pilot_idx[k]`).
        """
        self.action_step += 1
        forbidden_arr = self._forbidden_to_array(forbidden)
        if (not greedy) and self.rng.random() < self.epsilon():
            if forbidden_arr is None:
                return int(self.rng.integers(0, self.n_actions))
            choices = np.setdiff1d(np.arange(self.n_actions), forbidden_arr)
            return int(self.rng.choice(choices))
        heads, _ = self.online.forward(obs[None, :])
        q = self._q_from_heads(heads)[0]
        if forbidden_arr is not None:
            q = q.copy()
            q[forbidden_arr] = -np.inf
        return int(np.argmax(q))

    def q_values(self, obs: np.ndarray) -> np.ndarray:
        heads, _ = self.online.forward(obs[None, :])
        return self._q_from_heads(heads)[0]

    def remember(self, o, a, r, op, done):
        self.buffer.push(o, a, r, op, done)

    def update(self) -> Optional[float]:
        if self.buffer.size < self.cfg.min_buffer_for_train:
            return None
        if (self.train_step % self.cfg.train_every) != 0:
            self.train_step += 1
            return None
        self.train_step += 1
        bs = self.cfg.batch_size
        o, a, r, op, d = self.buffer.sample(bs, self.rng)

        # --- target value (double DQN) ------------------------------------
        on_heads_next, _ = self.online.forward(op)
        q_next_online = self._q_from_heads(on_heads_next)
        if self.mask_self_action:
            # The replay buffer's `a` becomes the *current* pilot of
            # k_star at the next state, so it is the forbidden action
            # the bootstrapped policy must avoid.
            q_next_online = q_next_online.copy()
            q_next_online[np.arange(bs), a] = -np.inf
        a_star = np.argmax(q_next_online, axis=1)
        tg_heads_next, _ = self.target.forward(op)
        q_next_target = self._q_from_heads(tg_heads_next)
        if self.mask_self_action:
            q_next_target = q_next_target.copy()
            q_next_target[np.arange(bs), a] = -np.inf
        bootstrap = q_next_target[np.arange(bs), a_star]
        target_q = r + (1.0 - d) * self.cfg.gamma * bootstrap

        # --- online forward + Huber loss gradient -------------------------
        on_heads, cache = self.online.forward(o)
        q_pred = self._q_from_heads(on_heads)
        idx = np.arange(bs)
        td_error = q_pred[idx, a] - target_q
        delta = self.cfg.huber_delta
        # Huber: linear outside [-delta, delta], quadratic inside.
        clipped = np.clip(td_error, -delta, delta)
        # dL/dq[a] = clipped / bs (Huber-on-residual gradient)
        dq = np.zeros_like(q_pred)
        dq[idx, a] = clipped / bs
        dheads = self._dq_to_dheads(dq)
        grads = self.online.backward(cache, dheads)
        self.online.adam_step(grads, self.adam, lr=self.cfg.lr)

        if (self.train_step % self.cfg.target_sync_every) == 0:
            self._copy_to_target()
        return float(np.mean(np.abs(td_error)))

    def _dq_to_dheads(self, dq):
        raise NotImplementedError

    # serialisation ---------------------------------------------------------
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        sd = self.online.state_dict()
        sd["obs_dim"] = np.array(self.obs_dim)
        sd["n_actions"] = np.array(self.n_actions)
        sd["is_dueling"] = np.array(int(self.is_dueling))
        np.savez(path, **sd)

    def load(self, path: str) -> None:
        sd = dict(np.load(path, allow_pickle=False))
        self.online.load_state_dict(sd)
        self._copy_to_target()


class DuelingDQNAgent(_BaseAgent):
    """Dueling DDQN — the proposed pilot-update agent.

    Two heads on top of the shared trunk:
      V(o; theta_V)        — scalar state value
      A(o, a; theta_A)     — vector advantage of size n_actions
    Combined as Q(o, a) = V + (A - mean_a A).

    Masks self-action: the agent must change the pilot of `k_star` at
    every near-RT step rather than degenerate into a no-op.
    """
    is_dueling = True
    mask_self_action = True

    def _build_net(self) -> MLP:
        layers = (self.obs_dim, *self.cfg.hidden)
        # Heads: (V dim 1, A dim n_actions)
        return MLP(layers, (1, self.n_actions), np.random.default_rng(int(self.rng.integers(0, 2**31 - 1))))

    def _q_from_heads(self, heads) -> np.ndarray:
        V = heads[0]                 # (B, 1)
        A = heads[1]                 # (B, n_actions)
        A_centered = A - A.mean(axis=1, keepdims=True)
        return V + A_centered

    def _dq_to_dheads(self, dq):
        # Q = V + A - mean(A) -> dV = sum(dq); dA = dq - mean(dq)
        dV = dq.sum(axis=1, keepdims=True)
        dA = dq - dq.mean(axis=1, keepdims=True)
        return [dV, dA]


class VanillaDQNAgent(_BaseAgent):
    """Plain DQN with a single Q head — the baseline.

    Masks the just-taken action in the TD target. For the naive agent,
    after taking action ``a = k * tau_p + t`` the user `k` is on pilot
    `t` at the next state, so action `a` is a no-op there and forbidding
    it in bootstrap is the right constraint (the same constraint we
    enforce at action-selection time via ``forbidden``).
    """
    is_dueling = False
    mask_self_action = True

    def _build_net(self) -> MLP:
        layers = (self.obs_dim, *self.cfg.hidden)
        return MLP(layers, (self.n_actions,), np.random.default_rng(int(self.rng.integers(0, 2**31 - 1))))

    def _q_from_heads(self, heads) -> np.ndarray:
        return heads[0]

    def _dq_to_dheads(self, dq):
        return [dq]


def make_agent(kind: str,
               obs_dim: int,
               n_actions: int,
               cfg: Optional[DQNConfig] = None,
               rng: Optional[np.random.Generator] = None) -> _BaseAgent:
    cfg = cfg or DQNConfig()
    if kind == "dueling":
        return DuelingDQNAgent(obs_dim, n_actions, cfg, rng=rng)
    if kind == "vanilla":
        return VanillaDQNAgent(obs_dim, n_actions, cfg, rng=rng)
    raise ValueError(f"Unknown agent kind: {kind}")
