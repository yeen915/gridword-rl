"""Evaluate every saved model and the baselines on seen / unseen / harder maps.

Writes results/metrics_per_seed.csv, results/summary.csv and results/summary.md.
"""
from __future__ import annotations

import glob
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO

from gridworld_rl.baselines import BFSOraclePolicy, GreedyPolicy, RandomPolicy
from gridworld_rl.evaluate import SB3Policy, evaluate
from gridworld_rl.maps import hard_test_maps, test_maps, train_maps

# split name -> (maps, map size)
SPLITS = {
    # training-pool maps; for agents trained on a fixed pool these are maps they have seen
    # (pool50 agents are scored on their own 50 maps). Random-map agents never saw a specific map.
    "train (seen)": (lambda: train_maps(500), 8),
    "test (unseen)": (test_maps, 8),
    "hard (unseen, OOD)": (hard_test_maps, 8),
    "12x12 test (unseen)": (lambda: test_maps(12), 12),
}
METRICS = ["success_rate", "mean_return", "mean_steps_success", "spl", "lava_rate", "enemy_rate", "timeout_rate"]


def condition_of(path):
    m = re.match(r"(.+)_s(\d+)\.zip$", os.path.basename(path))
    return m.group(1), int(m.group(2))


def main():
    rows = []
    models = sorted(glob.glob("models/*.zip"))
    groups = defaultdict(list)
    for p in models:
        cond, seed = condition_of(p)
        groups[cond].append((seed, p))

    for split, (maps_fn, size) in SPLITS.items():
        maps = maps_fn()
        for n_enemies in (0, 1):
            conds = [c for c in groups if ("enemy" in c) == (n_enemies == 1) and ("12x12" in c) == (size == 12)]
            if size == 12 and (n_enemies == 1 or not conds):
                continue
            if n_enemies == 1 and not conds:
                continue
            for pol in (RandomPolicy(0), GreedyPolicy(0), BFSOraclePolicy()):
                m = evaluate(pol, maps, n_enemies=n_enemies, size=size)
                rows.append({"split": split, "enemies": n_enemies, "agent": pol.name, "seed": 0,
                             **{k: m[k] for k in METRICS}})
            for cond in conds:
                pool = re.search(r"pool(\d+)$", cond)
                cmaps = maps
                if split.startswith("train") and pool:
                    cmaps = train_maps(500)[: int(pool.group(1))]  # the maps this agent actually trained on
                for seed, path in groups[cond]:
                    model = PPO.load(path, device="cpu")
                    m = evaluate(SB3Policy(model, cond), cmaps, n_enemies=n_enemies, size=size)
                    rows.append({"split": split, "enemies": n_enemies, "agent": cond, "seed": seed,
                                 **{k: m[k] for k in METRICS}})
                    print(f"{split:20s} enemies={n_enemies} {cond:28s} s{seed} succ={m['success_rate']:.3f} "
                          f"ret={m['mean_return']:.2f} steps={m['mean_steps_success']:.2f}", flush=True)
                    if cond == "ppo_shaped" or "12x12" in cond:
                        # same network, actions sampled from the policy instead of argmax
                        torch.manual_seed(seed)
                        m = evaluate(SB3Policy(model, cond, deterministic=False), maps, size=size)
                        rows.append({"split": split, "enemies": n_enemies, "agent": f"{cond} (sampled actions)",
                                     "seed": seed, **{k: m[k] for k in METRICS}})

    df = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/metrics_per_seed.csv", index=False)

    agg = df.groupby(["split", "enemies", "agent"], sort=False)[METRICS].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg["n_seeds"] = df.groupby(["split", "enemies", "agent"], sort=False).size()
    agg = agg.reset_index()
    agg.to_csv("results/summary.csv", index=False)

    def fmt(r, k, pct=False, d=2):
        mu, sd = r[f"{k}_mean"], r[f"{k}_std"]
        if pct:
            mu, sd = 100 * mu, 100 * sd
        s = f"{mu:.1f}" if pct else f"{mu:.{d}f}"
        if r["n_seeds"] > 1 and not np.isnan(sd):
            s += f" ± {sd:.1f}" if pct else f" ± {sd:.{d}f}"
        return s

    order = ["Random", "Greedy heuristic", "BFS oracle", "ppo_shaped", "ppo_shaped (sampled actions)",
             "ppo_shaped_manhattan", "ppo_sparse", "ppo_sparse_curriculum", "ppo_shaped_nomemory",
             "ppo_shaped_pool500", "ppo_shaped_pool50", "ppo_shaped_enemy",
             "ppo_shaped_12x12", "ppo_shaped_12x12 (sampled actions)", "ppo_sparse_12x12",
             "ppo_sparse_12x12 (sampled actions)", "ppo_sparse_curriculum_12x12",
             "ppo_sparse_curriculum_12x12 (sampled actions)"]
    agg["_o"] = agg["agent"].map(lambda a: order.index(a) if a in order else len(order))
    agg = agg.sort_values(["_o"], kind="stable").drop(columns="_o")
    agg = pd.concat([agg[(agg.split == s)] for s in SPLITS])  # keep split order
    lines = []
    for (split, en), g in agg.groupby(["split", "enemies"], sort=False):
        lines.append(f"\n### {split}{' — 1 moving enemy' if en else ''} ({len(SPLITS[split][0]())} maps)\n")
        lines.append("| Agent | Seeds | Success % | Avg return | Avg steps (successes) | SPL | Lava % | Timeout % |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for _, r in g.iterrows():
            lines.append(f"| {r['agent']} | {r['n_seeds']} | {fmt(r, 'success_rate', True)} | {fmt(r, 'mean_return')} | "
                         f"{fmt(r, 'mean_steps_success')} | {fmt(r, 'spl')} | {fmt(r, 'lava_rate', True)} | "
                         f"{fmt(r, 'timeout_rate', True)} |")
    with open("results/summary.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
