"""Step 6: compare PPO with the baselines on the SAME simulated episodes (all results SIMULATED).

1. Tune the periodic baseline's interval and pick the best PPO seed on VALIDATION episodes.
2. Evaluate every policy on 1000 fresh TEST episodes (different seeds).
3. Plots: training curves, policy comparison, sample episode timeline, example decisions table.

Outputs: models/ppo_agent.zip, reports/metrics/rl_*.json/csv, reports/figures/rl_*.png
"""
import _bootstrap  # noqa: F401

import json
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from pdm.config import load_config, path
from pdm.health import assess_health
from pdm.rl.agent import Agent, build_observation, shielded_policy
from pdm.rl.baselines import make_periodic, random_policy, rule_based, run_to_failure
from pdm.rl.env import ACTIONS, BearingMaintenanceEnv, ObservationModel
from pdm.rl.evaluate import evaluate_policies, ppo_policy, run_episode

VAL_SEED0, TEST_SEED0 = 2_000_000, 3_000_000
SIM = "SIMULATION - illustrative costs"


def main() -> None:
    cfg = load_config()["rl"]
    models_dir, metrics_dir, fig_dir = path("models_dir"), path("metrics_dir"), path("figures_dir")
    om = ObservationModel()
    seeds = [s for s in cfg["ppo"]["seeds"] if (models_dir / f"ppo_seed{s}.zip").exists()]
    ppo_models = {s: PPO.load(str(models_dir / f"ppo_seed{s}.zip"), device="cpu") for s in seeds}

    # ---------------------------------------------------------------- 1. validation
    n_val = 300
    periodic = {f"periodic_{n}": make_periodic(n) for n in (15, 20, 25, 30, 45, 60, 90)}
    val_pols = {**periodic, **{f"ppo_seed{s}": ppo_policy(m) for s, m in ppo_models.items()}}
    val, _ = evaluate_policies(val_pols, n_val, seed0=VAL_SEED0, obs_model=om)
    best_periodic = val.loc[list(periodic)].mean_reward.idxmax()
    best_seed = int(val.loc[[f"ppo_seed{s}" for s in seeds]].mean_reward.idxmax().replace("ppo_seed", ""))
    shutil.copy2(models_dir / f"ppo_seed{best_seed}.zip", models_dir / "ppo_agent.zip")
    print(f"validation: best periodic = {best_periodic}, best PPO seed = {best_seed}")
    print(val.round(1)[["mean_reward", "failures_per_episode", "availability_pct"]].to_string())

    # ---------------------------------------------------------------- 2. test
    n_test = cfg["eval_episodes"]
    test_pols = {"PPO agent + safety shield": shielded_policy(ppo_policy(ppo_models[best_seed])),
                 "PPO agent": ppo_policy(ppo_models[best_seed]), "Rule-based": rule_based,
                 f"Periodic (every {best_periodic.split('_')[1]} days)": periodic[best_periodic],
                 "Run-to-failure": run_to_failure, "Random": random_policy}
    for s in seeds:
        if s != best_seed:
            test_pols[f"PPO seed {s}"] = ppo_policy(ppo_models[s])
    summary, per_ep = evaluate_policies(test_pols, n_test, seed0=TEST_SEED0, obs_model=om)
    per_ep.to_csv(metrics_dir / "rl_test_episodes.csv", index=False)
    seed_rewards = [summary.loc["PPO agent", "mean_reward"]] + \
                   [summary.loc[f"PPO seed {s}", "mean_reward"] for s in seeds if s != best_seed]
    # paired comparisons vs rule-based on identical episodes
    def paired_vs_rule(name: str) -> dict:
        a = per_ep[per_ep.policy == name].set_index("seed").reward
        b = per_ep[per_ep.policy == "Rule-based"].set_index("seed").reward
        diff = (a - b).to_numpy()
        se = diff.std(ddof=1) / np.sqrt(len(diff))
        return {"mean_diff": float(diff.mean()), "sem_diff": float(se),
                "ci95": [float(diff.mean() - 1.96 * se), float(diff.mean() + 1.96 * se)],
                "ppo_better_episodes_pct": float(100 * np.mean(diff > 0)),
                "rule_better_episodes_pct": float(100 * np.mean(diff < 0))}
    paired = paired_vs_rule("PPO agent")
    paired_shield = paired_vs_rule("PPO agent + safety shield")
    result = {"label": SIM, "test_episodes": n_test, "best_ppo_seed": best_seed, "best_periodic": best_periodic,
              "summary": json.loads(summary.round(4).to_json(orient="index")),
              "ppo_all_seeds_mean_reward": {"mean": float(np.mean(seed_rewards)), "std": float(np.std(seed_rewards)),
                                            "values": [float(x) for x in seed_rewards]},
              "ppo_minus_rule_paired": paired, "shielded_minus_rule_paired": paired_shield, "validation": json.loads(val.round(4).to_json(orient="index"))}
    (metrics_dir / "rl_evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    # ---------------------------------------------------------------- 3a. training curves
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for s in seeds:
        c = pd.read_csv(metrics_dir / f"ppo_curve_seed{s}.csv")
        ax.plot(c.timestep, c.episode_reward.rolling(100, min_periods=10).mean(), label=f"PPO seed {s}")
    for name, color in (("Rule-based", "black"), ("Run-to-failure", "grey")):
        ax.axhline(summary.loc[name, "mean_reward"], ls="--", color=color, lw=1, label=f"{name} (test mean)")
    ax.set_xlabel("training steps (days simulated)"); ax.set_ylabel("episode reward (rolling mean of 100)")
    ax.set_title(f"PPO training curve - {SIM}", fontsize=10); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(fig_dir / "rl_training_curve.png", dpi=130); plt.close(fig)

    # ---------------------------------------------------------------- 3b. policy comparison
    main_pols = [p for p in summary.index if not p.startswith("PPO seed")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for ax, col, title in zip(axes, ["mean_reward", "failures_per_episode", "availability_pct"],
                              ["Mean reward per 180-day episode", "Breakdowns per episode", "Availability (%)"]):
        vals = summary.loc[main_pols, col]
        err = summary.loc[main_pols, "sem_reward"] * 1.96 if col == "mean_reward" else None
        ax.bar(range(len(vals)), vals, yerr=err, capsize=3,
               color=["#e76f51" if p.startswith("PPO agent") else "#457b9d" for p in main_pols])
        ax.set_xticks(range(len(vals)), [p.replace(" (", "\n(") for p in main_pols], fontsize=8)
        ax.set_title(title, fontsize=10)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.2f}" if col == "failures_per_episode" else f"{v:.0f}", ha="center",
                    va="bottom", fontsize=8)
    fig.suptitle(f"Policy comparison on the same {n_test} simulated episodes - {SIM}", fontsize=10)
    fig.tight_layout(); fig.savefig(fig_dir / "rl_policy_comparison.png", dpi=130); plt.close(fig)

    # ---------------------------------------------------------------- 3c. sample episode timeline
    env = BearingMaintenanceEnv(obs_model=om)
    # show the test episode where PPO and the rule-based policy differ the most (an informative story;
    # the overall comparison above is over all episodes, not this one)
    a = per_ep[per_ep.policy == "PPO agent + safety shield"].set_index("seed").reward
    b = per_ep[per_ep.policy == "Rule-based"].set_index("seed").reward
    ep_seed = int((a - b).abs().idxmax())
    timelines = {}
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    colors = {a: c for a, c in zip(ACTIONS, ["#2a9d8f", "#e9c46a", "#f4a261", "#8ab17d", "#e76f51"])}
    for ax, (name, pol) in zip(axes, [("PPO agent + safety shield", test_pols["PPO agent + safety shield"]),
                                      ("Rule-based", rule_based)]):
        r = run_episode(env, pol, ep_seed, record=True)
        tl = pd.DataFrame(r["timeline"])
        timelines[name] = {"reward": r["reward"], "failures": r["failures"], "timeline": r["timeline"]}
        ax.plot(tl.day, tl.degradation, color="black", lw=1.2, label="true degradation (hidden)")
        for act, g in tl.groupby("action"):
            ax.scatter(g.day, np.full(len(g), 1.08), color=colors[act], s=14, label=act, marker="s")
        ax.set_ylim(-0.05, 1.15); ax.set_ylabel("degradation")
        ax.set_title(f"{name}: reward {r['reward']:.0f}, breakdowns {r['failures']}", fontsize=9)
        ax.legend(fontsize=7, ncol=3, loc="center right")
    axes[1].set_xlabel("day")
    fig.suptitle(f"Episode where PPO and the rule-based policy differ most (seed {ep_seed}); actions as squares "
                 f"on top - {SIM}", fontsize=10)
    fig.tight_layout(); fig.savefig(fig_dir / "rl_episode_timeline.png", dpi=130); plt.close(fig)
    (metrics_dir / "rl_sample_episode.json").write_text(json.dumps({"seed": ep_seed, **timelines}, indent=1),
                                                       encoding="utf-8")

    # ---------------------------------------------------------------- 3d. example decisions
    agent = Agent(models_dir / "ppo_agent.zip")
    rules = load_config()["health"]
    examples = [("Normal", 0.95, None, 1), ("Normal", 0.62, None, 1), ("Ball", 0.80, 0.007, 1),
                ("Outer race", 0.55, 0.007, 1), ("Inner race", 0.90, 0.014, 1), ("Inner race", 0.90, 0.014, 3),
                ("Outer race", 0.90, 0.021, 0), ("Ball", 0.70, 0.028, 2)]
    rows = []
    for diag, conf, size, load in examples:
        h = assess_health(diag, conf, size, load, rules)
        obs, _ = build_observation(diag, conf, size, h.status, h.low_confidence, load)
        rec = agent.recommend(obs)
        rows.append({"diagnosis": diag, "confidence": conf, "est_size_in": size, "load_hp": load,
                     "health": h.status + (" (low conf.)" if h.low_confidence else ""),
                     "agent_choice": rec.agent_action, "agent_prob": round(rec.action_probabilities[rec.agent_action], 3),
                     "final_action (with shield)": rec.action,
                     "rule_action": rec.rule_based_action})
    ex = pd.DataFrame(rows)
    ex.to_csv(metrics_dir / "rl_example_decisions.csv", index=False)

    print(f"\nTEST ({n_test} episodes, {SIM}):")
    print(summary.round(2)[["mean_reward", "sem_reward", "failures_per_episode", "maintenances_per_episode",
                            "unnecessary_maintenance_per_episode", "inspections_per_episode", "downtime_days",
                            "availability_pct"]].to_string())
    print(f"\nPPO over seeds: {np.mean(seed_rewards):.1f} +/- {np.std(seed_rewards):.1f} {np.round(seed_rewards, 1)}")
    print(f"PPO - rule-based (paired): {paired['mean_diff']:.1f}, 95% CI {np.round(paired['ci95'], 1)}; "
          f"PPO better in {paired['ppo_better_episodes_pct']:.0f}% of episodes, rule better in "
          f"{paired['rule_better_episodes_pct']:.0f}%")
    print("\nExample decisions:\n", ex.to_string(index=False))


if __name__ == "__main__":
    main()
