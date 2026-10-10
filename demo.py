"""Record a demo GIF of the trained agent (and the greedy baseline) on unseen maps.

    python demo.py                                   # uses models/ppo_shaped_s0.zip
    python demo.py --model models/ppo_shaped_enemy_s0.zip --enemies 1 --out results/demo_enemy.gif
"""
from __future__ import annotations

import argparse

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
from stable_baselines3 import PPO

from gridworld_rl.baselines import GreedyPolicy
from gridworld_rl.env import GridWorldEnv, render_frame
from gridworld_rl.evaluate import SB3Policy
from gridworld_rl.maps import test_maps


def caption(frame, text):
    img = Image.new("RGB", (frame.shape[1], frame.shape[0] + 40), (255, 255, 255))
    img.paste(Image.fromarray(frame), (0, 40))
    d = ImageDraw.Draw(img)
    d.text((6, 5), text, fill=(30, 30, 30))
    d.text((6, 22), "blue=agent  green=goal  orange=lava  dark=wall  red=enemy", fill=(120, 120, 120))
    return np.array(img)


def play(policy, layout, n_enemies, seed, title):
    env = GridWorldEnv(n_enemies=n_enemies, layout_pool=[layout],
                       visit_memory=getattr(policy, "visit_memory", True))
    env.reset(seed=seed)
    frames = [caption(render_frame(env), f"{title}  step 0")]
    done, info = False, {}
    while not done:
        _, _, term, trunc, info = env.step(policy.act(env))
        done = term or trunc
        if env.steps <= 30 or env.steps % 10 == 0:  # fast-forward long (looping) episodes
            frames.append(caption(render_frame(env), f"{title}  step {env.steps}" + ("  (fast-forward)" if env.steps > 30 else "")))
    result = info["event"] if term else "timeout"
    frames += [caption(render_frame(env), f"{title}  -> {result} in {env.steps} steps (optimal {env.optimal_len})")] * 6
    return frames


def side_by_side(a, b):
    n = max(len(a), len(b))
    a += [a[-1]] * (n - len(a))
    b += [b[-1]] * (n - len(b))
    gap = np.full((a[0].shape[0], 12, 3), 255, dtype=np.uint8)
    return [np.concatenate([x, gap, y], axis=1) for x, y in zip(a, b)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/ppo_shaped_s0.zip")
    p.add_argument("--enemies", type=int, default=0)
    p.add_argument("--maps", type=int, nargs="*", default=None, help="indices into the unseen test maps")
    p.add_argument("--out", default="results/demo.gif")
    args = p.parse_args()

    ppo = SB3Policy(PPO.load(args.model, device="cpu"), "PPO")
    greedy = GreedyPolicy(seed=0)
    maps = test_maps()
    idx = args.maps
    if idx is None:
        # pick maps where the greedy heuristic fails or is slow, plus a couple of ordinary ones,
        # so the demo shows what learning buys over the heuristic
        from gridworld_rl.evaluate import evaluate
        g = evaluate(greedy, maps[:120], n_enemies=args.enemies)["episodes_raw"]
        q = evaluate(ppo, maps[:120], n_enemies=args.enemies)["episodes_raw"]
        hard = [i for i in range(120) if not g[i]["success"] and q[i]["success"]][:3]
        easy = [i for i in range(120) if g[i]["success"] and q[i]["success"]][:2]
        # (failure cases are shown separately by failure_cases.py)
        idx = easy[:1] + hard + easy[1:]
    frames = []
    for k, i in enumerate(idx):
        a = play(ppo, maps[i], args.enemies, 7000 + i, f"PPO  map {i}")
        b = play(GreedyPolicy(seed=0), maps[i], args.enemies, 7000 + i, f"Greedy  map {i}")
        frames += side_by_side(a, b)
    imageio.mimsave(args.out, frames, duration=0.18, loop=0)
    print("wrote", args.out, len(frames), "frames, maps", idx)


if __name__ == "__main__":
    main()
