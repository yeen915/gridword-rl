"""Train a PPO agent on the GridWorld.

Examples
--------
python train.py --name ppo_shaped --shaping bfs --seed 0
python train.py --name ppo_sparse_curriculum --shaping none --curriculum --seed 0
"""
from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
from stable_baselines3.common.vec_env import DummyVecEnv

from gridworld_rl.env import GridWorldEnv, RewardConfig
from gridworld_rl.evaluate import SB3Policy, evaluate
from gridworld_rl.maps import train_maps, val_maps


class GridCNN(BaseFeaturesExtractor):
    """Small conv net for the (4, 15, 15) egocentric map. Weight sharing across positions gives
    a much better inductive bias for 'walls look the same everywhere' than an MLP."""

    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        c = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(c, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2), nn.ReLU(),   # 15x15 -> 7x7
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n = self.cnn(torch.zeros(1, *observation_space.shape)).shape[1]
        self.linear = nn.Sequential(nn.Linear(n, features_dim), nn.ReLU())

    def forward(self, obs):
        return self.linear(self.cnn(obs))


class TrainingLogger(BaseCallback):
    """Every `eval_every` steps: log rolling training stats and a deterministic evaluation on
    100 held-out validation maps; optionally advance the curriculum."""

    def __init__(self, path, eval_every, n_enemies, curriculum, visit_memory=True, size=8, verbose=1):
        super().__init__(verbose)
        self.size = size
        self.visit_memory = visit_memory
        self.path, self.eval_every, self.n_enemies = path, eval_every, n_enemies
        self.curriculum = curriculum
        self.k = 2 if curriculum else None
        self.next_eval = 0
        self.best, self.best_path = (-1.0, -1e9), None
        self.t0 = time.time()

    def _set_k(self):
        # VecEnv.set_attr would set the attribute on the Monitor wrapper, not the env itself,
        # so reach through to the unwrapped GridWorldEnv explicitly.
        for e in self.training_env.envs:
            e.unwrapped.curriculum_max_dist = self.k

    def _on_training_start(self):
        if self.curriculum:
            self._set_k()
        self.file = open(self.path, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(["timesteps", "train_ep_return", "train_success", "train_ep_len",
                              "val_success", "val_return", "val_spl", "curriculum_k"])

    def _on_step(self):
        if self.num_timesteps < self.next_eval:
            return True
        self.next_eval += self.eval_every
        buf = self.model.ep_info_buffer
        tr_ret = np.mean([e["r"] for e in buf]) if buf else np.nan
        tr_len = np.mean([e["l"] for e in buf]) if buf else np.nan
        tr_succ = np.mean([e.get("is_success", 0) for e in buf]) if buf else np.nan
        m = evaluate(SB3Policy(self.model, "ppo"), val_maps(self.size), n_enemies=self.n_enemies, size=self.size)
        self.writer.writerow([self.num_timesteps, tr_ret, tr_succ, tr_len, m["success_rate"],
                              m["mean_return"], m["spl"], self.k if self.k is not None else -1])
        self.file.flush()
        # Model selection on the *validation* maps (the test maps are never used for selection).
        score = (m["success_rate"], m["mean_return"])
        if self.best_path and score > self.best:
            self.best = score
            self.model.save(self.best_path)
        if self.verbose:
            print(f"[{self.num_timesteps:>8}] train_succ={tr_succ:.2f} val_succ={m['success_rate']:.2f} "
                  f"val_ret={m['mean_return']:.2f} k={self.k} ({time.time() - self.t0:.0f}s)", flush=True)
        # Adaptive curriculum: widen the start-goal distance once training episodes succeed reliably.
        if self.curriculum and self.k is not None and len(buf) >= 50 and tr_succ >= 0.8:
            self.k += 2
            if self.k > 2 * self.size + 4:  # longer than practically any shortest path -> use the real start
                self.k = None
            self._set_k()
            self.model.ep_info_buffer.clear()
        return True

    def _on_training_end(self):
        self.file.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--shaping", default="bfs", choices=["bfs", "manhattan", "none"])
    p.add_argument("--pool-size", type=int, default=500, help="number of training maps; 0 = new random map every episode")
    p.add_argument("--curriculum", action="store_true")
    p.add_argument("--enemies", type=int, default=0, help="number of randomly moving enemies (dynamic obstacles)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--eval-every", type=int, default=25_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--size", type=int, default=8, help="map side length")
    p.add_argument("--arch", default="mlp", choices=["cnn", "mlp"])
    p.add_argument("--no-visit-memory", action="store_true", help="drop the episodic visit-count channel")
    p.add_argument("--net", type=int, default=256, help="hidden width of the 2-layer MLP")
    args = p.parse_args()
    cnn = args.arch == "cnn"

    torch.set_num_threads(1)
    os.makedirs("models", exist_ok=True)
    os.makedirs("results/curves", exist_ok=True)

    assert args.pool_size == 0 or args.size == 8, "fixed map pools are only defined for 8x8"
    pool = train_maps(args.pool_size) if args.pool_size > 0 else None
    reward = RewardConfig(shaping=args.shaping)

    def make_env(rank):
        def _f():
            env = GridWorldEnv(size=args.size, reward=reward, layout_pool=pool, n_enemies=args.enemies, flatten=not cnn,
                               visit_memory=not args.no_visit_memory)
            env = Monitor(env, info_keywords=("is_success",))
            env.reset(seed=args.seed * 1000 + rank)
            return env
        return _f

    venv = DummyVecEnv([make_env(i) for i in range(args.n_envs)])
    if cnn:
        policy_kwargs = dict(features_extractor_class=GridCNN, net_arch=[128])
    else:
        policy_kwargs = dict(net_arch=[args.net, args.net])
    model = PPO(
        "MlpPolicy", venv, seed=args.seed, verbose=0, device="cpu",
        learning_rate=3e-4, n_steps=256, batch_size=512, n_epochs=4,
        gamma=reward.gamma, gae_lambda=0.95, clip_range=0.2, ent_coef=args.ent_coef,
        policy_kwargs=policy_kwargs,
    )
    tag = f"{args.name}_s{args.seed}"
    cb = TrainingLogger(f"results/curves/{tag}.csv", args.eval_every, args.enemies, args.curriculum,
                        visit_memory=not args.no_visit_memory, size=args.size)
    os.makedirs("models/final", exist_ok=True)
    cb.best_path = f"models/{tag}"            # best validation checkpoint -> used for evaluation
    model.learn(total_timesteps=args.timesteps, callback=cb)
    model.save(f"models/final/{tag}")          # last checkpoint, kept for reference
    print("saved", f"models/{tag}.zip (best on validation, val success/return={cb.best})",
          f"and models/final/{tag}.zip (last)")


if __name__ == "__main__":
    main()
