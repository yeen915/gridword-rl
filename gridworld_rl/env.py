"""Procedurally generated GridWorld with walls, lava hazards and an optional moving enemy.

Gymnasium API. The agent sees an *egocentric* view of the whole map: a (2N-1)x(2N-1)
crop centred on the agent, so the observation is translation-equivariant and the
policy does not have to re-learn the same behaviour at every absolute position.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Actions: 0=up, 1=right, 2=down, 3=left
ACTIONS = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=np.int64)
ACTION_NAMES = ["up", "right", "down", "left"]

EMPTY, WALL, LAVA = 0, 1, 2


@dataclass
class RewardConfig:
    """Reward terms. `shaping` selects the potential function used for shaping."""

    goal: float = 10.0
    wall_bump: float = -2.0      # trying to move into a wall / map edge (agent stays put)
    lava: float = -5.0           # stepping into lava ends the episode
    enemy: float = -5.0          # being caught by the moving enemy ends the episode
    step: float = -0.01
    shaping: str = "bfs"         # "bfs" | "manhattan" | "none"
    shaping_scale: float = 0.1   # Phi(s) = -scale * distance(s, goal)
    gamma: float = 0.99          # must match the learner's discount for policy invariance


def bfs_distances(grid: np.ndarray, goal: tuple[int, int]) -> np.ndarray:
    """Shortest-path distance from every cell to `goal`, moving only through EMPTY cells.

    Unreachable cells get -1.
    """
    n = grid.shape[0]
    dist = np.full(grid.shape, -1, dtype=np.int64)
    dist[goal] = 0
    q = deque([goal])
    while q:
        r, c = q.popleft()
        for dr, dc in ACTIONS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n and dist[nr, nc] < 0 and grid[nr, nc] == EMPTY:
                dist[nr, nc] = dist[r, c] + 1
                q.append((nr, nc))
    return dist


def generate_layout(rng: np.random.Generator, size: int, wall_density: float, n_lava: int):
    """Sample a solvable layout: returns (grid, start, goal). Retries until start->goal is
    reachable without touching walls or lava and is at least `size//2` steps long."""
    while True:
        grid = np.zeros((size, size), dtype=np.int8)
        grid[rng.random((size, size)) < wall_density] = WALL
        free = np.argwhere(grid == EMPTY)
        if len(free) < n_lava + 2:
            continue
        idx = rng.choice(len(free), size=n_lava + 2, replace=False)
        start, goal = tuple(free[idx[0]]), tuple(free[idx[1]])
        for k in idx[2:]:
            grid[tuple(free[k])] = LAVA
        d = bfs_distances(grid, goal)
        if d[start] >= max(2, size // 2):
            return grid, start, goal


class GridWorldEnv(gym.Env):
    """Goal-seeking GridWorld.

    Observation (float32, shape (C, 2N-1, 2N-1), flattened by default):
        channel 0: wall (cells outside the map are also walls)
        channel 1: lava
        channel 2: goal
        channel 3: enemy (all zeros if dynamic obstacles are disabled)
        channel 4: episodic visit count of each cell, min(count, 3) / 3  (only if visit_memory)
    The agent is always at the centre of the crop, so it needs no channel of its own.

    Action: Discrete(4) — up, right, down, left.
    """

    metadata = {"render_modes": ["rgb_array", "ansi"], "render_fps": 6}

    def __init__(
        self,
        size: int = 8,
        wall_density: float = 0.2,
        n_lava: int = 3,
        n_enemies: int = 0,
        max_steps: int | None = None,
        reward: RewardConfig | None = None,
        layout_pool: list | None = None,
        flatten: bool = True,
        visit_memory: bool = True,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.size = size
        self.wall_density = wall_density
        self.n_lava = n_lava
        self.n_enemies = n_enemies
        self.max_steps = max_steps or 4 * size * size // 2  # 128 for an 8x8 map
        self.reward_cfg = reward or RewardConfig()
        self.layout_pool = layout_pool  # if given, episodes sample from this fixed set of maps
        self.flatten = flatten
        self.visit_memory = visit_memory
        self.render_mode = render_mode
        self.curriculum_max_dist: int | None = None  # set by the curriculum callback

        self.view = 2 * size - 1
        self.n_channels = 5 if visit_memory else 4
        shape = (self.n_channels, self.view, self.view)
        if flatten:
            self.observation_space = spaces.Box(0.0, 1.0, shape=(int(np.prod(shape)),), dtype=np.float32)
        else:
            self.observation_space = spaces.Box(0.0, 1.0, shape=shape, dtype=np.float32)
        self.action_space = spaces.Discrete(4)

    # ------------------------------------------------------------------ helpers
    def _potential(self, pos) -> float:
        cfg = self.reward_cfg
        if cfg.shaping == "none":
            return 0.0
        if cfg.shaping == "bfs":
            d = self._dist[pos]
            if d < 0:  # standing somewhere the goal can't be reached from (shouldn't happen)
                d = self.size * self.size
        else:  # manhattan
            d = abs(pos[0] - self.goal[0]) + abs(pos[1] - self.goal[1])
        return -cfg.shaping_scale * float(d)

    def _obs(self) -> np.ndarray:
        n, v = self.size, self.view
        obs = np.zeros((self.n_channels, v, v), dtype=np.float32)
        obs[0] = 1.0  # everything is wall unless covered by the map below
        r0 = (n - 1) - self.agent[0]
        c0 = (n - 1) - self.agent[1]
        obs[0, r0:r0 + n, c0:c0 + n] = (self.grid == WALL)
        obs[1, r0:r0 + n, c0:c0 + n] = (self.grid == LAVA)
        obs[2, r0 + self.goal[0], c0 + self.goal[1]] = 1.0
        for e in self.enemies:
            obs[3, r0 + e[0], c0 + e[1]] = 1.0
        if self.visit_memory:
            obs[4, r0:r0 + n, c0:c0 + n] = np.minimum(self.visits, 3) / 3.0
        return obs.reshape(-1) if self.flatten else obs

    def _move_enemies(self):
        """Enemies take a random legal step (never into walls, lava or the goal)."""
        n = self.size
        for i, (r, c) in enumerate(self.enemies):
            options = [(r, c)]
            for dr, dc in ACTIONS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < n and 0 <= nc < n and self.grid[nr, nc] == EMPTY and (nr, nc) != self.goal:
                    options.append((nr, nc))
            self.enemies[i] = options[self.np_random.integers(len(options))]

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if self.layout_pool is not None:
            grid, start, goal = self.layout_pool[self.np_random.integers(len(self.layout_pool))]
            self.grid, self.agent, self.goal = grid.copy(), tuple(start), tuple(goal)
        else:
            self.grid, self.agent, self.goal = generate_layout(
                self.np_random, self.size, self.wall_density, self.n_lava
            )
        self._dist = bfs_distances(self.grid, self.goal)
        if self.curriculum_max_dist is not None:
            # Curriculum: re-sample the start so the goal is at most k shortest-path steps away.
            k = self.curriculum_max_dist
            cand = np.argwhere((self._dist > 0) & (self._dist <= k))
            if len(cand):
                self.agent = tuple(int(x) for x in cand[self.np_random.integers(len(cand))])
        self.optimal_len = int(self._dist[self.agent])
        self.enemies = []
        if self.n_enemies:
            free = [tuple(p) for p in np.argwhere(self.grid == EMPTY)
                    if abs(p[0] - self.agent[0]) + abs(p[1] - self.agent[1]) > 2 and tuple(p) != self.goal]
            for k in self.np_random.choice(len(free), size=min(self.n_enemies, len(free)), replace=False):
                self.enemies.append(free[k])
        self.steps = 0
        self.trajectory = [self.agent]
        self.visits = np.zeros((self.size, self.size), dtype=np.float32)
        self.visits[self.agent] = 1
        return self._obs(), {"optimal_len": self.optimal_len}

    def step(self, action: int):
        cfg = self.reward_cfg
        self.steps += 1
        old = self.agent
        phi_old = self._potential(old)

        reward = cfg.step
        terminated = False
        event = "move"

        nr, nc = old[0] + ACTIONS[action][0], old[1] + ACTIONS[action][1]
        if not (0 <= nr < self.size and 0 <= nc < self.size) or self.grid[nr, nc] == WALL:
            reward += cfg.wall_bump
            event = "bump"
        else:
            self.agent = (int(nr), int(nc))

        if self.grid[self.agent] == LAVA:
            reward += cfg.lava
            terminated, event = True, "lava"
        elif self.agent == self.goal:
            reward += cfg.goal
            terminated, event = True, "goal"
        else:
            if self.enemies:
                self._move_enemies()
            if self.agent in self.enemies:
                reward += cfg.enemy
                terminated, event = True, "enemy"

        # Potential-based shaping F = gamma*Phi(s') - Phi(s), with Phi(terminal) = 0
        # (Ng, Harada & Russell, 1999) — leaves the optimal policy unchanged.
        phi_new = 0.0 if terminated else self._potential(self.agent)
        reward += cfg.gamma * phi_new - phi_old

        truncated = self.steps >= self.max_steps and not terminated
        self.trajectory.append(self.agent)
        self.visits[self.agent] += 1
        info = {"event": event, "is_success": event == "goal", "optimal_len": self.optimal_len}
        return self._obs(), float(reward), terminated, truncated, info

    # ------------------------------------------------------------------ rendering
    def render(self):
        if self.render_mode == "ansi":
            chars = {EMPTY: ".", WALL: "#", LAVA: "~"}
            rows = []
            for r in range(self.size):
                row = ""
                for c in range(self.size):
                    p = (r, c)
                    row += "A" if p == self.agent else "G" if p == self.goal else \
                        "E" if p in self.enemies else chars[int(self.grid[p])]
                rows.append(row)
            return "\n".join(rows)
        return render_frame(self)


def render_frame(env: GridWorldEnv, cell: int = 48) -> np.ndarray:
    """RGB rendering used for the demo GIF."""
    n = env.size
    img = np.full((n * cell, n * cell, 3), 245, dtype=np.uint8)
    colors = {WALL: (60, 60, 70), LAVA: (230, 110, 40)}
    for r in range(n):
        for c in range(n):
            t = int(env.grid[r, c])
            if t in colors:
                img[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = colors[t]
    # visited trail
    for (r, c) in env.trajectory[:-1]:
        img[r * cell + cell // 3:(r + 1) * cell - cell // 3, c * cell + cell // 3:(c + 1) * cell - cell // 3] = (170, 200, 240)

    def disc(pos, color, rad):
        yy, xx = np.ogrid[:cell, :cell]
        mask = (yy - cell / 2) ** 2 + (xx - cell / 2) ** 2 <= rad ** 2
        block = img[pos[0] * cell:(pos[0] + 1) * cell, pos[1] * cell:(pos[1] + 1) * cell]
        block[mask] = color

    gr, gc = env.goal
    img[gr * cell + 6:(gr + 1) * cell - 6, gc * cell + 6:(gc + 1) * cell - 6] = (60, 180, 90)
    for e in env.enemies:
        disc(e, (200, 40, 60), cell * 0.35)
    disc(env.agent, (40, 90, 220), cell * 0.32)
    # grid lines
    img[::cell, :] = 210
    img[:, ::cell] = 210
    return img


def make_layout_pool(n_maps: int, seed: int, size: int = 8, wall_density: float = 0.2, n_lava: int = 3):
    """A fixed, reproducible set of maps (used to separate train maps from unseen test maps)."""
    rng = np.random.default_rng(seed)
    return [generate_layout(rng, size, wall_density, n_lava) for _ in range(n_maps)]
