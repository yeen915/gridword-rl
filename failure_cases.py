"""Find and draw failure episodes of a trained agent on the unseen test maps.

    python failure_cases.py [--model models/ppo_shaped_s0.zip]
Writes results/failure_cases.png and prints a breakdown of success by path length.
"""
from __future__ import annotations

import argparse
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from gridworld_rl.baselines import GreedyPolicy
from gridworld_rl.env import GridWorldEnv, render_frame
from gridworld_rl.evaluate import SB3Policy, evaluate
from gridworld_rl.maps import test_maps


def replay(policy, layout, seed):
    env = GridWorldEnv(layout_pool=[layout], visit_memory=policy.visit_memory)
    env.reset(seed=seed)
    done = False
    while not done:
        _, _, term, trunc, info = env.step(policy.act(env))
        done = term or trunc
    return env, (info["event"] if term else "timeout")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/ppo_shaped_s0.zip")
    p.add_argument("--n", type=int, default=4)
    args = p.parse_args()

    pol = SB3Policy(PPO.load(args.model, device="cpu"), "PPO")
    maps = test_maps()
    eps = evaluate(pol, maps)["episodes_raw"]
    greedy = evaluate(GreedyPolicy(0), maps)["episodes_raw"]
    fails = [i for i, e in enumerate(eps) if not e["success"]]
    print(f"{len(fails)} failures / {len(eps)} unseen maps:", dict(Counter(eps[i]["event"] for i in fails)))
    print("| Shortest path length | Maps | PPO success | Greedy success | PPO extra steps (successes) |")
    print("|---|---|---|---|---|")
    for lo, hi in [(1, 5), (6, 9), (10, 14), (15, 99)]:
        idx = [i for i, e in enumerate(eps) if lo <= e["optimal_len"] <= hi]
        if not idx:
            continue
        extra = [eps[i]["steps"] - eps[i]["optimal_len"] for i in idx if eps[i]["success"]]
        print(f"| {lo}-{hi if hi < 99 else ''} | {len(idx)} | {100 * np.mean([eps[i]['success'] for i in idx]):.1f}% | "
              f"{100 * np.mean([greedy[i]['success'] for i in idx]):.1f}% | {np.mean(extra):.1f} |")

    show = fails[: args.n]
    fig, axes = plt.subplots(1, len(show), figsize=(3.2 * len(show), 3.6))
    for ax, i in zip(np.atleast_1d(axes), show):
        env, ev = replay(pol, maps[i], 12345 + i)  # same seed as evaluate()
        ax.imshow(render_frame(env, cell=32))
        visits = env.visits
        ax.set_title(f"map {i}: {ev}\nshortest path {env.optimal_len}, max visits/cell {int(visits.max())}",
                     fontsize=8.5, loc="left")
        ax.axis("off")
    fig.suptitle("Failure cases on unseen maps (light-blue squares = cells visited; blue = final position)",
                 x=0.01, y=1.06, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig("results/failure_cases.png", dpi=130, bbox_inches="tight")
    print("wrote results/failure_cases.png")


if __name__ == "__main__":
    main()
