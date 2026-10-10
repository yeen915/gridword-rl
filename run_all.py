"""Reproduce every experiment in the README, then evaluate and plot.

    python run_all.py                 # everything (about 4 h on a 2-core CPU; much faster with more cores)
    python run_all.py --quick         # 1 seed per condition, 300k steps (~15-20 min), for a smoke test
    python run_all.py --only main     # just the main agent
    python run_all.py --jobs 4        # parallel training processes (default: CPU count, max 6)

Works the same on Windows, Linux and macOS.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

# name -> (extra train.py args, seeds)
EXPERIMENTS = {
    # main agent: shaped reward (BFS potential), visit-count memory, new random map every episode
    "main":             ("ppo_shaped",            "--shaping bfs --pool-size 0", [0, 1, 2]),
    # E1 reward ablation
    "manhattan":        ("ppo_shaped_manhattan",  "--shaping manhattan --pool-size 0", [0, 1, 2]),
    "sparse":           ("ppo_sparse",            "--shaping none --pool-size 0", [0, 1, 2]),
    # E2 curriculum on the sparse reward
    "curriculum":       ("ppo_sparse_curriculum", "--shaping none --pool-size 0 --curriculum", [0, 1, 2]),
    # E3 observation ablation: no episodic memory
    "nomemory":         ("ppo_shaped_nomemory",   "--shaping bfs --pool-size 0 --no-visit-memory", [0, 1, 2]),
    # E4 generalisation: fixed training-map pools
    "pool500":          ("ppo_shaped_pool500",    "--shaping bfs --pool-size 500", [0, 1]),
    "pool50":           ("ppo_shaped_pool50",     "--shaping bfs --pool-size 50", [0, 1]),
    # E6 bigger 12x12 maps, where exploration with a sparse reward is genuinely hard
    "big_shaped":       ("ppo_shaped_12x12",      "--shaping bfs --pool-size 0 --size 12", [0]),
    "big_sparse":       ("ppo_sparse_12x12",      "--shaping none --pool-size 0 --size 12", [0]),
    "big_curriculum":   ("ppo_sparse_curriculum_12x12", "--shaping none --pool-size 0 --size 12 --curriculum", [0]),
    # E5 dynamic obstacle
    "enemy":            ("ppo_shaped_enemy",      "--shaping bfs --pool-size 0 --enemies 1", [0]),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--only", nargs="*", default=None, choices=list(EXPERIMENTS))
    p.add_argument("--jobs", type=int, default=min(os.cpu_count() or 1, 6))
    p.add_argument("--timesteps", type=int, default=2_000_000)
    p.add_argument("--skip-train", action="store_true")
    args = p.parse_args()

    steps = 300_000 if args.quick else args.timesteps
    os.makedirs("logs", exist_ok=True)
    jobs = []
    for key in args.only or EXPERIMENTS:
        name, extra, seeds = EXPERIMENTS[key]
        for s in (seeds[:1] if args.quick else seeds):
            if os.path.exists(f"models/{name}_s{s}.zip"):
                print("skip (exists):", name, s)
                continue
            n_steps = steps if args.quick or "--size 12" not in extra else 3_000_000  # bigger maps get 3M
            cmd = [sys.executable, "train.py", "--name", name, "--seed", str(s),
                   "--timesteps", str(n_steps), *extra.split()]
            jobs.append((f"{name}_s{s}", cmd))

    def run(job):
        tag, cmd = job
        print("start", tag, flush=True)
        with open(f"logs/{tag}.log", "w") as f:
            rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
        print("done ", tag, "rc", rc, flush=True)
        return rc

    if not args.skip_train and jobs:
        with ThreadPoolExecutor(args.jobs) as ex:
            rcs = list(ex.map(run, jobs))
        if any(rcs):
            sys.exit("some training runs failed — see logs/")

    subprocess.check_call([sys.executable, "evaluate_all.py"])
    subprocess.check_call([sys.executable, "plots.py"])
    subprocess.check_call([sys.executable, "demo.py"])
    if os.path.exists("models/ppo_shaped_enemy_s0.zip"):
        subprocess.check_call([sys.executable, "demo.py", "--model", "models/ppo_shaped_enemy_s0.zip",
                               "--enemies", "1", "--out", "results/demo_enemy.gif"])
    subprocess.check_call([sys.executable, "failure_cases.py"])


if __name__ == "__main__":
    main()
