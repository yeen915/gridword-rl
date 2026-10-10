"""Evaluation: run any policy on a fixed list of maps, one episode per map.

All maps are played in lock-step so a neural policy can act on the whole batch with a single
forward pass (orders of magnitude faster than one predict() call per step).
"""
from __future__ import annotations

import numpy as np

from .env import GridWorldEnv, RewardConfig


class SB3Policy:
    """Adapter so a Stable-Baselines3 model has the same interface as the baselines."""

    def __init__(self, model, name: str, deterministic: bool = True):
        self.model, self.name, self.deterministic = model, name, deterministic
        # 5 input channels <=> the model was trained with the visit-count channel
        n_inputs = int(np.prod(model.observation_space.shape))
        side = int(round((n_inputs / 5) ** 0.5))
        self.visit_memory = side * side * 5 == n_inputs  # otherwise 4 channels

    def act_batch(self, envs) -> np.ndarray:
        shape = self.model.observation_space.shape  # flat (MLP) or (C, H, W) (CNN)
        obs = np.stack([e._obs().reshape(shape) for e in envs])
        actions, _ = self.model.predict(obs, deterministic=self.deterministic)
        return actions

    def act(self, env) -> int:
        return int(self.act_batch([env])[0])


def evaluate(policy, layouts, reward: RewardConfig | None = None, n_enemies: int = 0,
             seed: int = 12345, size: int = 8, visit_memory: bool = True) -> dict:
    """Metrics are computed under the same reward definition for every policy (default:
    the shaped reward described in the README) so returns are comparable across agents."""
    reward = reward or RewardConfig()
    visit_memory = getattr(policy, "visit_memory", visit_memory)
    envs = []
    for i, lay in enumerate(layouts):
        env = GridWorldEnv(size=size, n_enemies=n_enemies, reward=reward, layout_pool=[lay],
                           visit_memory=visit_memory)
        env.reset(seed=seed + i)
        envs.append(env)
    returns = np.zeros(len(envs))
    events = [None] * len(envs)
    alive = list(range(len(envs)))
    while alive:
        batch = [envs[i] for i in alive]
        if hasattr(policy, "act_batch"):
            actions = policy.act_batch(batch)
        else:
            actions = [policy.act(e) for e in batch]
        still = []
        for i, a in zip(alive, actions):
            _, r, term, trunc, info = envs[i].step(int(a))
            returns[i] += r
            if term:
                events[i] = info["event"]
            elif trunc:
                events[i] = "timeout"
            else:
                still.append(i)
        alive = still

    eps = [{"return": returns[i], "success": events[i] == "goal", "event": events[i],
            "steps": envs[i].steps, "optimal_len": envs[i].optimal_len} for i in range(len(envs))]
    succ = np.array([e["success"] for e in eps])
    steps_succ = np.array([e["steps"] for e in eps if e["success"]])
    # SPL (success weighted by path length, Anderson et al. 2018): 1.0 = always the shortest path
    spl = np.mean([e["optimal_len"] / max(e["steps"], e["optimal_len"]) if e["success"] else 0.0 for e in eps])
    ev = [e["event"] for e in eps]
    return {
        "policy": policy.name,
        "episodes": len(eps),
        "success_rate": float(succ.mean()),
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_steps_success": float(steps_succ.mean()) if len(steps_succ) else float("nan"),
        "spl": float(spl),
        "lava_rate": ev.count("lava") / len(eps),
        "enemy_rate": ev.count("enemy") / len(eps),
        "timeout_rate": ev.count("timeout") / len(eps),
        "episodes_raw": eps,
    }
