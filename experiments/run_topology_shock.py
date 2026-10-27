"""Held-out DQN tuning, matched allocation, nominal controls, and abrupt fading.

DROO-Adapted follows https://arxiv.org/abs/1808.01977 Sec. IV-B/C with
nearest-edge binary actions and uniform resources, replacing the paper's
wireless-power model and convex time allocator. It is not an exact reproduction.
Decision-delay results are fixed-snapshot sensitivity analyses, not dynamic
queue or stale-channel simulations. No test trajectory selects a checkpoint.
"""
import argparse
import copy
from dataclasses import asdict, replace
from importlib.metadata import version
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import t
from we_kb_fpa.allocation import uniform_allocation
from we_kb_fpa.baselines import BaselineRunner
from we_kb_fpa.config import NetworkConfig, OptimizerConfig, CostConfig
from we_kb_fpa.dqn import DQNBaseline, DQNConfig
from we_kb_fpa.evaluation import snapshots, validation_bank, validate, fingerprint, delay_adjusted_metrics
from we_kb_fpa.knowledge_base import WirelessEdgeKnowledgeBase
from we_kb_fpa.we_kb_fpa import WEKBFPAOptimizer, OptimizationOutput
from we_kb_fpa.checkpoint import Checkpoint, namespace
from we_kb_fpa.training import train_seed


def training_config(config, slots):
    return replace(config, epsilon_decay=float(np.exp(np.log(config.epsilon_min) / max(1, .8 * slots))))


def tune_dqn(tuning_slots, validation_slots, output, checkpoint_root, checkpoint_every=1000, resume=True,
             selection_metric="validation_log_cost"):
    candidates = [DQNConfig(learning_rate=.001, gamma=1.),
                  DQNConfig(learning_rate=.0003, gamma=1.),
                  DQNConfig(learning_rate=.0001, gamma=1.)]
    diagnostics = []
    for split in range(2):
        bank = validation_bank(2000000 + split, validation_slots)
        for candidate, config in enumerate(candidates):
            checkpoint = Checkpoint(checkpoint_root, f"tuning-{split}-{candidate}",
                                    dict(split=split, candidate=candidate, config=asdict(config)), resume)
            state = checkpoint.load()
            if state and state.get("complete"):
                diagnostics.append(state["diagnostic"])
                continue
            if state is None:
                state = dict(complete=False, agent=DQNBaseline(seed=1000000 + split, config=training_config(config, tuning_slots)),
                             stream=snapshots(1000000 + split, tuning_slots))
            else:
                print(f"Resuming tuning candidate {candidate}, split {split}, slot {state['stream'].next_slot}", flush=True)
            agent = state["agent"]
            last_saved = time.perf_counter()
            for slot, args, _, _ in state["stream"]:
                agent.run(*args, training=True)
                if (slot+1) % checkpoint_every == 0 or slot+1 == tuning_slots or time.perf_counter()-last_saved >= 60:
                    checkpoint.save(state)
                    last_saved = time.perf_counter()
            diagnostic = dict(candidate=candidate, split=split, train_seed=1000000 + split,
                              validation_seed=2000000 + split, **asdict(training_config(config, tuning_slots)), **validate(agent, bank))
            diagnostics.append(diagnostic)
            checkpoint.save(dict(complete=True, diagnostic=diagnostic), compact=True)
            print(f"Validated DQN candidate {candidate}, development split {split}", flush=True)
    df = pd.DataFrame(diagnostics)
    df.to_csv(output / "hyperparameter_validation.csv", index=False)
    selected = int(df.groupby("candidate")[selection_metric].mean().idxmin())
    return candidates[selected]


def greedy_uniform(args, runner):
    start = time.perf_counter()
    raw = runner.run_greedy_latency(*args)
    tasks, users, edges, cloud, sinr, distances = args
    allocation = uniform_allocation(raw.best_targets, tasks, users, edges, cloud, distances)
    evaluation = runner.cost_engine.evaluate_allocation(tasks, users, edges, cloud, *allocation, sinr, distances)
    return OptimizationOutput(evaluation.total_cost, *allocation, [evaluation.total_cost],
                              time.perf_counter() - start, 1, evaluation, "Greedy-Uniform")


def evaluate_checkpoint(dqn, droo, quality, seed, budget, slots, fading_values, population, iterations,
                        checkpoint_root=None, checkpoint_every=1000, resume=True):
    records = []
    net, opt = NetworkConfig(num_users=8, num_edge_servers=3), OptimizerConfig(population_size=population, max_iterations=iterations)
    for fade in [0.] + list(fading_values):
        checkpoint = Checkpoint(checkpoint_root, f"evaluation-{seed}-{budget}-{fade:g}",
                                dict(seed=seed, budget=budget, fade=fade, slots=slots,
                                     population=population, iterations=iterations), resume) if checkpoint_root else None
        state = checkpoint.load() if checkpoint else None
        if state and state["complete"]:
            records.extend(state["records"])
            continue
        if state is None:
            agents = {"DQN-Frozen": copy.deepcopy(dqn), "DQN-Online": copy.deepcopy(dqn),
                      "DROO-Adapted-Online": copy.deepcopy(droo)}
            agents["DQN-Frozen"].memory.clear()
            agents["DQN-Frozen"].losses.clear()
            state = dict(complete=False, agents=agents,
                         knowledge={name: WirelessEdgeKnowledgeBase(net_config=net) for name in ("WE-KB-FPA-Uniform", "WE-KB-FPA-Joint")},
                         runner=BaselineRunner(net_config=net, seed=seed), records=[],
                         stream=snapshots(seed, slots, population, iterations, fade, slots//2))
        else:
            print(f"Resuming evaluation seed {seed}, budget {budget}, fade {fade:g}, slot {state['stream'].next_slot}", flush=True)
        agents, knowledge, runner = state["agents"], state["knowledge"], state["runner"]
        for slot, args, variance, mean_db in state["stream"]:
            snapshot_hash = fingerprint(args)
            for name in ["DQN-Frozen", "DQN-Online", "DROO-Adapted-Online", "Greedy-Uniform",
                         "WE-KB-FPA-Uniform", "WE-KB-FPA-Joint"]:
                if name in agents:
                    agent = agents[name]
                    out = agent.run(*args, training=name != "DQN-Frozen")
                    decision_ms = agent.last_decision_seconds * 1000
                    learning_ms = agent.last_learning_seconds * 1000
                    updates = agent.updates
                elif name == "Greedy-Uniform":
                    out = greedy_uniform(args, runner)
                    decision_ms, learning_ms, updates = out.wall_clock_time_sec * 1000, 0., 0
                else:
                    optimizer = WEKBFPAOptimizer(opt_config=opt, net_config=net, seed=seed + slot)
                    out = optimizer.optimize(*args, channel_variance=variance, knowledge_base=knowledge[name],
                                             allocation_mode="uniform" if name.endswith("Uniform") else "joint")
                    knowledge[name].add_experience(args[4], args[0], args[5], out.best_targets, out.best_cpu, out.best_bw, out.best_cost)
                    decision_ms, learning_ms, updates = out.wall_clock_time_sec * 1000, 0., 0
                if fingerprint(args) != snapshot_hash:
                    raise RuntimeError(f"{name} mutated the shared simulation snapshot")
                evaluation = out.evaluation_result
                row = dict(seed=seed, training_budget=budget, condition="nominal" if fade == 0 else "shock", fading_db=fade,
                           slot=slot, phase="pre" if slot < slots // 2 else "post", algorithm=name,
                           allocation_scope="joint" if name.endswith("Joint") else "binary_nearest_edge" if name.startswith("DROO") else "matched_uniform",
                           cost=out.best_cost, latency_ms=evaluation.average_latency * 1000,
                           violation_pct=evaluation.deadline_violation_rate, energy_j=evaluation.average_energy,
                           runtime_ms=out.wall_clock_time_sec * 1000, decision_ms=decision_ms, learning_ms=learning_ms,
                           exceeds_slot=out.wall_clock_time_sec > .1, mean_sinr_db=mean_db, snapshot_hash=snapshot_hash,
                           gradient_updates=updates, offload_pct=100 * float(np.mean(out.best_targets != 0)),
                           nominal_quality_pass=quality["dqn" if name.startswith("DQN") else "droo"]["nominal_quality_pass"] if name in agents else None)
                row.update(delay_adjusted_metrics(out, CostConfig()))
                state["records"].append(row)
            if checkpoint and ((slot+1) % min(10, checkpoint_every) == 0 or slot+1 == slots):
                checkpoint.save(state)
        records.extend(state["records"])
        if checkpoint:
            checkpoint.save(dict(complete=True, records=state["records"]), compact=True)
    return records


def adaptation_records(df):
    nominal = df[df.condition == "nominal"][["seed", "training_budget", "slot", "algorithm", "cost"]].rename(columns={"cost": "nominal_control_cost"})
    paired = df[df.condition == "shock"].merge(nominal, on=["seed", "training_budget", "slot", "algorithm"], validate="many_to_one")
    paired["cost_ratio_to_control"] = paired.cost / paired.nominal_control_cost.clip(lower=1e-12)
    reports = []
    for keys, group in paired.groupby(["seed", "training_budget", "fading_db", "algorithm"]):
        post = group[group.phase == "post"].sort_values("slot")
        pre = group[group.phase == "pre"].cost.mean()
        we = paired[(paired.seed == keys[0]) & (paired.training_budget == keys[1]) &
                    (paired.fading_db == keys[2]) & (paired.algorithm == "WE-KB-FPA-Uniform") & (paired.phase == "pre")].cost.mean()
        ratio = float(pre / max(we, 1e-12))
        comparable = 1/1.2 <= ratio <= 1.2
        qualified = bool(group.nominal_quality_pass.iloc[0]) if pd.notna(group.nominal_quality_pass.iloc[0]) else None
        recovered = (post.cost <= 1.1 * pre).to_numpy()
        lag = next((i for i in range(len(recovered)-2) if recovered[i:i+3].all()), None)
        reports.append(dict(seed=int(keys[0]), training_budget=int(keys[1]), fading_db=float(keys[2]), algorithm=keys[3],
                            recovery_slots=lag, first_shock_ratio_to_control=float(post.iloc[0].cost_ratio_to_control),
                            mean_post_ratio_to_control=float(post.cost_ratio_to_control.mean()),
                            excess_cost_area=float(np.maximum(0, post.cost - post.nominal_control_cost).sum()),
                            nominal_quality_pass=qualified, pre_cost_ratio_to_we_uniform=ratio,
                            nominal_performance_comparable=comparable,
                            adaptation_claim_eligible=bool(qualified and comparable and keys[3].startswith("DQN"))))
    return paired, reports


def paired_effects(df):
    means = df.groupby(["seed", "training_budget", "fading_db", "condition", "phase", "algorithm"])[["cost", "delay_adjusted_cost"]].mean().reset_index()
    proposed = means[means.algorithm == "WE-KB-FPA-Uniform"].drop(columns="algorithm")
    keys = ["seed", "training_budget", "fading_db", "condition", "phase"]
    comparison = means[means.algorithm.isin(["DQN-Frozen", "DQN-Online", "DROO-Adapted-Online", "Greedy-Uniform"])].merge(proposed, on=keys, suffixes=("_baseline", "_we"))
    comparison["cost_difference"] = comparison.cost_baseline - comparison.cost_we
    comparison["log_cost_difference"] = np.log1p(comparison.cost_baseline) - np.log1p(comparison.cost_we)
    comparison["delay_adjusted_difference"] = comparison.delay_adjusted_cost_baseline - comparison.delay_adjusted_cost_we
    rows = []
    for group_keys, group in comparison.groupby(["training_budget", "fading_db", "condition", "phase", "algorithm"]):
        for metric in ("cost_difference", "log_cost_difference", "delay_adjusted_difference"):
            n, std = len(group), group[metric].std()
            margin = float(t.ppf(.975, n-1) * std / np.sqrt(n)) if n > 1 else None
            rows.append(dict(zip(["training_budget", "fading_db", "condition", "phase", "baseline"], group_keys),
                             metric=metric, mean=float(group[metric].mean()), ci95_margin=margin, independent_seeds=n))
    return comparison, pd.DataFrame(rows)


def run_topology_shock(slots=40, training_slots=1000, seeds=(42, 43, 44), fading_db=35.,
                       output_dir="output/topology_shock", population=20, iterations=30,
                       training_budgets=None, tuning_slots=None, validation_slots=50, validation_interval=None,
                       fading_values=None, require_nominal_quality=False, checkpoint_every=1000, resume=True):
    if slots < 6 or slots % 2 or training_slots < 1 or not seeds or validation_slots < 1:
        raise ValueError("Use even slots >= 6, positive training/validation slots and nonempty seeds")
    if len(set(seeds)) != len(seeds) or any(not 0 <= seed < 100000 for seed in seeds):
        raise ValueError("Evaluation seeds must be unique integers in [0, 100000)")
    budgets = sorted(set(training_budgets or [max(1, training_slots//4), max(1, training_slots//2), training_slots]))
    if budgets[0] < 1 or budgets[-1] != training_slots:
        raise ValueError("Training budgets must be positive and end at training_slots")
    fades = sorted(set(fading_values if fading_values is not None else [fading_db]))
    if not fades or any(not np.isfinite(fade) or fade <= 0 for fade in fades):
        raise ValueError("Fading severities must be finite and positive")
    if population < 2 or iterations < 1:
        raise ValueError("Use population >= 2 and iterations >= 1")
    interval = validation_interval if validation_interval is not None else max(1, training_slots//10)
    tune_slots = tuning_slots if tuning_slots is not None else min(2000, training_slots)
    if interval < 1 or tune_slots < 1 or checkpoint_every < 1:
        raise ValueError("Validation interval and tuning slots must be positive")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = dict(slots=slots, training_slots=training_slots, training_budgets=budgets, seeds=list(seeds),
                  fading_db=max(fades), fading_values=fades, affected_edges="all", population=population,
                  iterations=iterations, validation_slots=validation_slots, validation_interval=interval, tuning_slots=tune_slots,
                  require_nominal_quality=require_nominal_quality,
                  seed_partitions=dict(tuning_train=[1000000, 1000001], tuning_validation=[2000000, 2000001],
                                       evaluation_train_offset=100000, checkpoint_validation_offset=300000, test=list(seeds)),
                  network=asdict(NetworkConfig(num_users=8, num_edge_servers=3)), cost=asdict(CostConfig()),
                  droo_reference="https://arxiv.org/abs/1808.01977",
                  droo_settings=dict(hidden=64, batch_size=32, replay_capacity=1024, learning_rate=.001, candidate_count=9),
                  checkpoint_format="NPZ inference exports plus atomic full-state resume checkpoints (agents, replay, optimizer, RNG, simulation and records)",
                  droo_adaptations="Binary local/nearest-edge; full slot inputs; penalized cost; uniform shares; no wireless power; fixed K=N+1",
                  delay_model="Serialized decision+learning delay added to fixed snapshot latency; no queue evolution or decision staleness",
                  recovery_definition="First of 3 post-shock slots <= 110% of own pre-shock mean; null if absent",
                  quality_definition="Held-out mean cost <= 120% of min(greedy-uniform, all-local) held-out mean; not an acceptance guarantee",
                  packages={name: version(name) for name in ("numpy", "scipy", "pandas", "matplotlib", "scikit-learn")})
    checkpoint_root = namespace(output / "checkpoints", config)
    selected = tune_dqn(tune_slots, validation_slots, output, checkpoint_root, checkpoint_every, resume)
    config["selected_dqn"] = asdict(training_config(selected, training_slots))
    config["dqn_observation"] = "Current task, observable load-dependent per-action latency/energy estimates, full slot context and partial assignment; no target masking or heuristic policy overrides"
    config["checkpoint_every"] = checkpoint_every
    (output / "config.json").write_text(json.dumps(config, indent=2))
    records, validations, timings, qualities, logs = [], [], [], [], []
    for seed in seeds:
        state, log_path = train_seed(seed, config, training_config(selected, training_slots), checkpoint_root,
                                     output, evaluate_checkpoint, checkpoint_every, resume)
        records.extend(state["records"])
        validations.extend(state["validations"])
        qualities.extend(state["qualities"])
        timings.append(state["timing"])
        logs.append(log_path)
        pd.DataFrame(records).to_csv(output / "slots.csv", index=False)
        pd.DataFrame(validations).to_csv(output / "validation.csv", index=False)
        (output / "quality_gate.json").write_text(json.dumps(qualities, indent=2))
    import shutil
    with (output / "training.csv").open("wb") as aggregate:
        for index, log_path in enumerate(logs):
            with log_path.open("rb") as log:
                if index:
                    log.readline()
                shutil.copyfileobj(log, aggregate)
    df = pd.DataFrame(records)
    pd.DataFrame(timings).to_csv(output / "timing.csv", index=False)
    df.groupby(["training_budget", "condition", "fading_db", "algorithm", "phase"])[
        ["cost", "delay_adjusted_cost", "latency_ms", "violation_pct", "delay_adjusted_violation_pct", "runtime_ms"]].mean().to_csv(output / "summary.csv")
    paired, recovery = adaptation_records(df)
    paired.to_csv(output / "normalized_adaptation.csv", index=False)
    (output / "recovery.json").write_text(json.dumps(recovery, indent=2))
    differences, effects = paired_effects(df)
    differences.to_csv(output / "paired_seed_differences.csv", index=False)
    effects.to_csv(output / "paired_summary.csv", index=False)
    plot_adaptation(df, slots, max(fades), output)
    plot_learning(pd.DataFrame(validations), df, output)
    return df


def plot_adaptation(df, slots, fading_db, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    selected = df[(df.training_budget == df.training_budget.max()) & (df.condition == "shock") & (df.fading_db == fading_db)]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4))
    for ax, metric, label in zip(axes, ["cost", "delay_adjusted_cost", "delay_adjusted_violation_pct", "runtime_ms"],
                                ["System cost", "Cost with decision delay", "Violations with decision delay (%)", "Decision + learning time (ms)"]):
        for name, group in selected.groupby("algorithm"):
            stats = group.groupby("slot")[metric].agg(["mean", "std", "count"])
            critical = t.ppf(.975, np.maximum(1, stats["count"] - 1))
            error = critical * stats["std"].fillna(0) / np.sqrt(stats["count"])
            ax.plot(stats.index, stats["mean"], label=name)
            lower = np.maximum(1e-12 if "cost" in metric else 0., stats["mean"] - error)
            upper = np.minimum(100, stats["mean"] + error) if "violation" in metric else stats["mean"] + error
            ax.fill_between(stats.index, lower, upper, alpha=.12)
        ax.axvline(slots//2-.5, color="black", linestyle="--")
        ax.set(xlabel="Slot", ylabel=label)
        if "cost" in metric:
            ax.set_yscale("log")
        if "violation" in metric:
            ax.set_ylim(0, 100)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"adaptation.{suffix}", dpi=300)
    plt.close(fig)
    paired, _ = adaptation_records(df)
    selected = paired[(paired.training_budget == paired.training_budget.max()) & (paired.fading_db == fading_db)]
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, group in selected.groupby("algorithm"):
        mean = group.groupby("slot").cost_ratio_to_control.mean()
        ax.plot(mean.index, mean, label=name)
    ax.axvline(slots//2-.5, color="black", linestyle="--")
    ax.set(xlabel="Slot", ylabel="Cost / paired no-shock control cost", yscale="log")
    ax.legend(fontsize=6)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"normalized_adaptation.{suffix}", dpi=300)
    plt.close(fig)


def plot_learning(validation, slots, output):
    output = Path(output)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for name, group in validation.groupby("algorithm"):
        stats = group.groupby("slot").validation_log_cost.mean()
        axes[0].plot(stats.index, stats, label=name)
        quality = group.groupby("slot").nominal_cost_ratio.mean()
        axes[1].plot(quality.index, quality, label=name)
    axes[0].set(xlabel="Training slots", ylabel="Held-out mean log(1 + cost / users)")
    axes[1].axhline(1.2, color="black", linestyle="--", label="Quality threshold")
    axes[1].set(xlabel="Training slots", ylabel="Held-out cost / baseline cost", yscale="log")
    for ax in axes:
        ax.legend(fontsize=7)
        ax.grid(alpha=.2)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"learning_curves.{suffix}", dpi=300)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, group in slots[slots.condition == "nominal"].groupby("algorithm"):
        per_seed = group.groupby(["training_budget", "seed"]).cost.mean()
        stats = per_seed.groupby("training_budget").agg(["mean", "std", "count"])
        margin = t.ppf(.975, np.maximum(1, stats["count"]-1)) * stats["std"].fillna(0) / np.sqrt(stats["count"])
        error = np.vstack((np.minimum(stats["mean"] * .999, margin), margin))
        ax.errorbar(stats.index, stats["mean"], yerr=error, label=name, marker="o")
    ax.set(xlabel="DQN / DROO training budget (slots)", ylabel="No-shock control mean cost", yscale="log")
    ax.legend(fontsize=6)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"training_budget.{suffix}", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slots", type=int, default=40)
    parser.add_argument("--training-slots", type=int, default=1000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--training-budgets", type=int, nargs="+")
    parser.add_argument("--fading-values", type=float, nargs="+", default=[35.])
    parser.add_argument("--tuning-slots", type=int)
    parser.add_argument("--validation-slots", type=int, default=50)
    parser.add_argument("--validation-interval", type=int)
    parser.add_argument("--require-nominal-quality", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--fresh", action="store_true", help="Ignore saved progress and start again")
    parser.add_argument("--output-dir", default="output/topology_shock")
    arguments = vars(parser.parse_args())
    arguments["resume"] = not arguments.pop("fresh")
    run_topology_shock(**arguments)
