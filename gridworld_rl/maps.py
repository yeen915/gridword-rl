"""Fixed map splits. Train / validation / test maps are generated from different seeds and
de-duplicated, so 'test' maps are guaranteed never to have been seen during training."""
from __future__ import annotations

from functools import lru_cache

from .env import make_layout_pool

TRAIN_SEED, VAL_SEED, TEST_SEED, HARD_SEED = 1000, 2024, 3000, 4000
N_TRAIN_MAX, N_VAL, N_TEST = 500, 100, 500


def _key(layout):
    grid, start, goal = layout
    return grid.tobytes() + bytes(start) + bytes(goal)


@lru_cache(maxsize=None)
def train_maps(n: int = N_TRAIN_MAX):
    return make_layout_pool(n, TRAIN_SEED)


def _unseen(n, seed, **kw):
    seen = {_key(l) for l in train_maps(N_TRAIN_MAX)}
    pool = [l for l in make_layout_pool(n + 50, seed, **kw) if _key(l) not in seen]
    return pool[:n]


@lru_cache(maxsize=None)
def val_maps(size: int = 8):
    return _unseen(N_VAL, VAL_SEED, size=size)


@lru_cache(maxsize=None)
def test_maps(size: int = 8):
    """Unseen maps from the same distribution as training (20% walls, 3 lava)."""
    return _unseen(N_TEST, TEST_SEED, size=size)


@lru_cache(maxsize=None)
def hard_test_maps():
    """Out-of-distribution: denser walls (30%) and twice the lava (6)."""
    return _unseen(N_TEST, HARD_SEED, wall_density=0.3, n_lava=6)
