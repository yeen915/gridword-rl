"""Training curves and evaluation bar charts -> results/*.png"""
from __future__ import annotations

import glob
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Fixed colour per agent (colour follows the entity, never its rank). PPO variants take
# categorical slots in order; non-learning references are neutral greys.
COLORS = {
    "ppo_shaped": "#2a78d6",
    "ppo_shaped_manhattan": "#eb6834",
    "ppo_sparse": "#1baf7a",
    "ppo_sparse_curriculum": "#eda100",
    "ppo_shaped_nomemory": "#e87ba4",
    "ppo_shaped_pool500": "#008300",
    "ppo_shaped_pool50": "#6250d6",
    "ppo_shaped_enemy": "#e34948",
    "ppo_shaped_12x12": "#2a78d6",
    "ppo_sparse_12x12": "#1baf7a",
    "ppo_sparse_curriculum_12x12": "#eda100",
    "ppo_shaped (sampled actions)": "#86b6ef",
    "ppo_shaped_12x12 (sampled actions)": "#86b6ef",
    "ppo_sparse_12x12 (sampled actions)": "#7fd4b4",
    "ppo_sparse_curriculum_12x12 (sampled actions)": "#f5c96a",
    "Random": "#b4b2a9",
    "Greedy heuristic": "#7a7870",
    "BFS oracle": "#3d3c37",
}
LABELS = {
    "ppo_shaped": "PPO (main: BFS-shaped, memory, random maps)",
    "ppo_shaped_manhattan": "PPO, Manhattan shaping",
    "ppo_sparse": "PPO, sparse reward",
    "ppo_sparse_curriculum": "PPO, sparse + curriculum",
    "ppo_shaped_nomemory": "PPO, no visit memory",
    "ppo_shaped_pool500": "PPO, 500 fixed train maps",
    "ppo_shaped_pool50": "PPO, 50 fixed train maps",
    "ppo_shaped_enemy": "PPO, with moving enemy",
    "ppo_shaped (sampled actions)": "PPO main, sampled actions",
    "ppo_shaped_12x12 (sampled actions)": "PPO 12x12, BFS shaping, sampled",
    "ppo_sparse_12x12 (sampled actions)": "PPO 12x12, sparse, sampled",
    "ppo_sparse_curriculum_12x12 (sampled actions)": "PPO 12x12, sparse + curriculum, sampled",
    "ppo_shaped_12x12": "PPO 12x12, BFS shaping",
    "ppo_sparse_12x12": "PPO 12x12, sparse",
    "ppo_sparse_curriculum_12x12": "PPO 12x12, sparse + curriculum",
}
TEXT, MUTED, GRID = "#2b2a27", "#6b6a64", "#e6e5e0"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": TEXT, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 130, "savefig.bbox": "tight",
})


def label(a):
    return LABELS.get(a, a)


def load_curves():
    curves = {}
    for f in sorted(glob.glob("results/curves/*.csv")):
        m = re.match(r"(.+)_s(\d+)\.csv$", os.path.basename(f))
        curves.setdefault(m.group(1), []).append(pd.read_csv(f))
    return curves


def mean_band(dfs, col, smooth=3):
    n = min(len(d) for d in dfs)
    x = dfs[0]["timesteps"].values[:n]
    ys = np.stack([d[col].values[:n] for d in dfs])
    if smooth > 1:  # light moving average for readability; raw data is in the CSVs
        k = np.ones(smooth) / smooth
        ys = np.stack([np.convolve(np.pad(y, (smooth - 1, 0), mode="edge"), k, mode="valid") for y in ys])
    return x, ys.mean(0), ys.std(0), len(dfs)


def plot_curves(curves, conds, col, ylabel, title, fname, ylim=None):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for c in conds:
        if c not in curves:
            continue
        x, mu, sd, n = mean_band(curves[c], col)
        ax.plot(x / 1e6, mu, color=COLORS[c], lw=2, label=f"{label(c)}  (n={n})")
        if n > 1:
            ax.fill_between(x / 1e6, mu - sd, mu + sd, color=COLORS[c], alpha=0.15, lw=0)
    ax.set_xlabel("environment steps (millions)")
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_title(title, loc="left", color=TEXT, fontsize=11)
    ax.legend(loc="lower right", fontsize=8.5)
    fig.savefig(f"results/{fname}")
    plt.close(fig)


def bar_panel(ax, df, agents, metric, xlabel, pct=False, fmt="{:.1f}"):
    df = df.set_index("agent")
    agents = [a for a in agents if a in df.index]
    mu = df.loc[agents, f"{metric}_mean"].values * (100 if pct else 1)
    sd = np.nan_to_num(df.loc[agents, f"{metric}_std"].values) * (100 if pct else 1)
    y = np.arange(len(agents))[::-1]
    ax.barh(y, mu, xerr=sd, color=[COLORS.get(a, "#888") for a in agents], height=0.62,
            error_kw=dict(ecolor=MUTED, lw=1, capsize=2), edgecolor="white", linewidth=2)
    ax.set_yticks(y)
    ax.set_yticklabels([label(a) for a in agents])
    span = max(np.abs(mu).max(), 1e-9)
    for yi, m, s in zip(y, mu, sd):
        x = m + s + span * 0.03 if m >= 0 else 0 + span * 0.03
        ax.text(x, yi, fmt.format(m), va="center", fontsize=8.5, color=TEXT)
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)


def main():
    os.makedirs("results", exist_ok=True)
    curves = load_curves()

    plot_curves(curves, ["ppo_shaped", "ppo_shaped_manhattan", "ppo_sparse", "ppo_sparse_curriculum", "ppo_shaped_nomemory"],
                "val_success", "success rate on 100 held-out maps (greedy policy)",
                "Training curves: validation success rate (mean ± std over seeds)", "training_success.png", (0, 1.02))
    plot_curves(curves, ["ppo_shaped", "ppo_shaped_manhattan", "ppo_sparse", "ppo_sparse_curriculum", "ppo_shaped_nomemory"],
                "val_return", "episode return on held-out maps (common shaped reward)",
                "Training curves: validation return", "training_return.png")
    plot_curves(curves, ["ppo_shaped", "ppo_shaped_manhattan", "ppo_sparse", "ppo_sparse_curriculum", "ppo_shaped_nomemory"],
                "train_ep_return", "training episode return (each run's own reward)",
                "Training curves: rollout return (not comparable across reward types)", "training_rollout_return.png")
    plot_curves(curves, ["ppo_shaped", "ppo_shaped_pool500", "ppo_shaped_pool50"],
                "val_success", "success rate on 100 held-out maps",
                "Generalisation during training: number of distinct training maps", "training_generalisation.png", (0, 1.02))

    plot_curves(curves, ["ppo_shaped_12x12", "ppo_sparse_12x12", "ppo_sparse_curriculum_12x12"],
                "train_success", "rollout success rate (sampled actions)",
                "12x12 maps: training success, shaping vs sparse vs curriculum (1 seed each)",
                "training_success_12x12.png", (0, 1.02))
    plot_curves(curves, ["ppo_shaped_12x12", "ppo_sparse_12x12", "ppo_sparse_curriculum_12x12"],
                "val_success", "success on 100 held-out 12x12 maps (argmax actions)",
                "12x12 maps: validation success with argmax actions", "val_success_12x12.png", (0, 1.02))

    # sample efficiency: env steps until the (3-point smoothed) validation success first reaches a threshold
    rows = ["| Condition | Seeds | Steps to 50% | Steps to 80% | Steps to 90% | Final val. success (last 200k) |",
            "|---|---|---|---|---|---|"]
    for c, dfs in curves.items():
        firsts = {th: [] for th in (0.5, 0.8, 0.9)}
        finals = []
        for d in dfs:
            v = d["val_success"].rolling(3, min_periods=1).mean().values
            t = d["timesteps"].values
            for th in firsts:
                i = np.where(v >= th)[0]
                firsts[th].append(t[i[0]] / 1e6 if len(i) else np.nan)
            finals.append(d["val_success"].iloc[-8:].mean())

        def f(xs):
            xs = np.array(xs)
            if np.isnan(xs).all():
                return "never"
            ok = xs[~np.isnan(xs)]
            s = f"{ok.mean():.2f}M"
            if len(ok) < len(xs):
                s += f" ({len(ok)}/{len(xs)} seeds)"
            return s
        rows.append(f"| {label(c)} | {len(dfs)} | {f(firsts[0.5])} | {f(firsts[0.8])} | {f(firsts[0.9])} | "
                    f"{100 * np.mean(finals):.1f}% |")
    with open("results/sample_efficiency.md", "w") as fh:
        fh.write("\n".join(rows) + "\n")

    # curriculum schedule
    if "ppo_sparse_curriculum" in curves:
        fig, ax = plt.subplots(figsize=(7.2, 2.8))
        for d in curves["ppo_sparse_curriculum"] + curves.get("ppo_sparse_curriculum_12x12", []):
            k = d["curriculum_k"].replace(-1, np.nan)
            ax.step(d["timesteps"] / 1e6, k, where="post", color=COLORS["ppo_sparse_curriculum"], lw=1.6, alpha=0.8)
        ax.set_xlabel("environment steps (millions)")
        ax.set_ylabel("max start→goal distance k")
        ax.set_title("Curriculum schedule per seed (line ends = full task)", loc="left", color=TEXT, fontsize=11)
        fig.savefig("results/curriculum_schedule.png")
        plt.close(fig)

    summ = pd.read_csv("results/summary.csv")
    main_agents = ["Random", "Greedy heuristic", "ppo_shaped", "BFS oracle"]
    for split, fname in [("test (unseen)", "comparison_test.png"), ("hard (unseen, OOD)", "comparison_hard.png")]:
        d = summ[(summ.split == split) & (summ.enemies == 0)]
        fig, axes = plt.subplots(1, 3, figsize=(13, 2.9), sharey=True)
        bar_panel(axes[0], d, main_agents, "success_rate", "success rate (%)", pct=True)
        bar_panel(axes[1], d, main_agents, "mean_return", "average episode return", fmt="{:.2f}")
        bar_panel(axes[2], d, main_agents, "mean_steps_success", "average steps (successful episodes)", fmt="{:.1f}")
        fig.suptitle(f"Trained agent vs baselines — {split} maps", x=0.02, ha="left", color=TEXT, fontsize=11)
        fig.tight_layout()
        fig.savefig(f"results/{fname}")
        plt.close(fig)

    # all PPO variants, success on unseen + OOD maps
    variants = ["ppo_shaped", "ppo_shaped (sampled actions)", "ppo_shaped_manhattan", "ppo_sparse", "ppo_sparse_curriculum", "ppo_shaped_nomemory",
                "ppo_shaped_pool500", "ppo_shaped_pool50", "Greedy heuristic"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    for ax, split in zip(axes, ["train (seen)", "test (unseen)", "hard (unseen, OOD)"]):
        d = summ[(summ.split == split) & (summ.enemies == 0)]
        bar_panel(ax, d, variants, "success_rate", f"success % — {split}", pct=True)
        ax.set_xlim(0, 112)
    fig.suptitle("Ablations: success rate by map split (error bars = std over seeds)", x=0.02, ha="left", color=TEXT, fontsize=11)
    fig.tight_layout()
    fig.savefig("results/ablations.png")
    plt.close(fig)

    d = summ[(summ.split == "test (unseen)") & (summ.enemies == 1)]
    if len(d):
        fig, axes = plt.subplots(1, 2, figsize=(10, 2.6), sharey=True)
        ag = ["Random", "Greedy heuristic", "ppo_shaped_enemy", "BFS oracle"]
        bar_panel(axes[0], d, ag, "success_rate", "success rate (%)", pct=True)
        bar_panel(axes[1], d, ag, "enemy_rate", "caught by enemy (%)", pct=True)
        fig.suptitle("Dynamic obstacle: one randomly moving enemy (unseen maps)", x=0.02, ha="left", color=TEXT, fontsize=11)
        fig.tight_layout()
        fig.savefig("results/comparison_enemy.png")
        plt.close(fig)
    d = summ[summ.split == "12x12 test (unseen)"]
    if len(d):
        ag = ["Random", "Greedy heuristic", "ppo_shaped_12x12", "ppo_shaped_12x12 (sampled actions)",
              "ppo_sparse_12x12", "ppo_sparse_12x12 (sampled actions)", "ppo_sparse_curriculum_12x12",
              "ppo_sparse_curriculum_12x12 (sampled actions)", "BFS oracle"]
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
        bar_panel(axes[0], d, ag, "success_rate", "success rate (%)", pct=True)
        bar_panel(axes[1], d, ag, "mean_steps_success", "average steps (successful episodes)")
        fig.suptitle("12x12 unseen maps (1 seed each)", x=0.02, ha="left", color=TEXT, fontsize=11)
        fig.tight_layout()
        fig.savefig("results/comparison_12x12.png")
        plt.close(fig)
    print("plots written to results/")


if __name__ == "__main__":
    main()
