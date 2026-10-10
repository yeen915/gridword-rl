"""Non-learning reference policies. All take the env (they read its true state) and return an action."""
from __future__ import annotations

import numpy as np

from .env import ACTIONS, EMPTY, WALL, GridWorldEnv


class RandomPolicy:
    """Lower bound: uniformly random actions."""

    name = "Random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act(self, env: GridWorldEnv) -> int:
        return int(self.rng.integers(4))


class GreedyPolicy:
    """Simple heuristic: step to a safe neighbour (not wall/lava/enemy) that most reduces the
    Manhattan distance to the goal; break ties randomly. If no safe neighbour gets closer,
    take a random safe step. Has no memory and no planning, so it gets stuck behind
    U-shaped walls — exactly the case where learning (or search) should help."""

    name = "Greedy heuristic"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act(self, env: GridWorldEnv) -> int:
        r, c = env.agent
        gr, gc = env.goal
        cur = abs(r - gr) + abs(c - gc)
        safe, better = [], []
        for a, (dr, dc) in enumerate(ACTIONS):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < env.size and 0 <= nc < env.size):
                continue
            if env.grid[nr, nc] != EMPTY or (nr, nc) in env.enemies:
                continue
            safe.append(a)
            if abs(nr - gr) + abs(nc - gc) < cur:
                better.append(a)
        pool = better or safe or [0, 1, 2, 3]
        return int(self.rng.choice(pool))


class BFSOraclePolicy:
    """Upper-bound reference (not a baseline to beat): follows the true shortest path that
    avoids walls and lava. With a moving enemy it is no longer guaranteed optimal."""

    name = "BFS oracle"

    def act(self, env: GridWorldEnv) -> int:
        r, c = env.agent
        best, best_d = 0, None
        for a, (dr, dc) in enumerate(ACTIONS):
            nr, nc = r + dr, c + dc
            if 0 <= nr < env.size and 0 <= nc < env.size and env.grid[nr, nc] != WALL:
                d = env._dist[nr, nc]
                if d >= 0 and (best_d is None or d < best_d):
                    best, best_d = a, d
        return best
