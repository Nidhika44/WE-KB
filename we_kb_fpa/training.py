"""Resume learning at committed slot boundaries, including replay and RNG state."""
import copy
import csv
import json
import os
from pathlib import Path
import time
import numpy as np
from we_kb_fpa.checkpoint import Checkpoint
from we_kb_fpa.dqn import DQNBaseline
from we_kb_fpa.droo import DROOAdapted
from we_kb_fpa.learner_state import save_learner_state
from we_kb_fpa.config import NetworkConfig
from we_kb_fpa.evaluation import snapshots, validation_bank, validate, save_agent


TRAINING_FIELDS = ["seed", "slot", "algorithm", "cost", "epsilon", "updates", "mean_loss"]


def train_seed(seed, config, selected_config, checkpoint_root, output, evaluate, checkpoint_every=1000, resume=True):
    identity = dict(seed=seed, config={key: value for key, value in config.items() if key != "checkpoint_every"},
                    selected_config=selected_config)
    checkpoint = Checkpoint(checkpoint_root, f"seed-{seed}-training", identity, resume)
    state = checkpoint.load()
    log_path = Path(checkpoint_root) / f"seed-{seed}-training-{checkpoint.identity[:24]}.csv"
    if state and state["complete"]:
        print(f"Resuming completed seed {seed}: training and evaluation already saved", flush=True)
        return state, log_path
    if state is None:
        state = dict(complete=False, stream=snapshots(100000+seed, config["training_slots"]),
                     agents=dict(dqn=DQNBaseline(seed=100000+seed, config=selected_config,net_config=NetworkConfig(num_users=8,num_edge_servers=3)),
                                 droo=DROOAdapted(seed=100000+seed,net_config=NetworkConfig(num_users=8,num_edge_servers=3))),
                     best={}, best_scores=dict(dqn=float("inf"), droo=float("inf")), best_quality={},
                     records=[], validations=[], qualities=[], completed_budgets=[], pending_budget=None,
                     training_seconds=0., elapsed_seconds=0., log_offset=0)
        with log_path.open("w", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=TRAINING_FIELDS).writeheader()
            stream.flush()
            os.fsync(stream.fileno())
            state["log_offset"] = stream.tell()
    else:
        print(f"Resuming seed {seed} at training slot {state['stream'].next_slot}/{config['training_slots']}", flush=True)
    if not log_path.exists() or log_path.stat().st_size < state["log_offset"]:
        raise RuntimeError(f"Training log missing or truncated: {log_path}; restore it or start with --fresh")
    bank = validation_bank(300000+seed, config["validation_slots"])
    last_accounted = time.perf_counter()
    last_saved = last_accounted
    with log_path.open("r+", newline="", encoding="utf-8") as training_file:
        training_file.seek(state["log_offset"])
        training_file.truncate()
        writer = csv.DictWriter(training_file, fieldnames=TRAINING_FIELDS)

        def commit():
            nonlocal last_accounted, last_saved
            training_file.flush()
            os.fsync(training_file.fileno())
            state["log_offset"] = training_file.tell()
            now = time.perf_counter()
            state["elapsed_seconds"] += now - last_accounted
            last_accounted = now
            checkpoint.save(state)
            last_saved = time.perf_counter()

        commit()
        while True:
            budget = state["pending_budget"]
            if budget is not None:
                best, quality = state["best"], state["best_quality"]
                print(f"Seed {seed}, budget {budget}: held-out DQN / reference cost = "
                      f"{quality['dqn']['nominal_cost_ratio']:.3f}; quality pass = {quality['dqn']['nominal_quality_pass']}", flush=True)
                if config["require_nominal_quality"] and not quality["dqn"]["nominal_quality_pass"]:
                    Path(output, "quality_gate.json").write_text(json.dumps(state["qualities"], indent=2))
                    raise RuntimeError("DQN failed held-out nominal quality gate; inspect diagnostics and tune training before testing")
                for name in best:
                    save_agent(best[name], Path(output) / f"{name}_seed_{seed}_budget_{budget}.npz")
                    save_learner_state(best[name], Path(output) / f"{name}_seed_{seed}_budget_{budget}.agent.pkl",
                                       dict(seed=seed,training_budget=budget,algorithm=name,
                                            quality=quality[name],selected_config=selected_config,
                                            training_seed=100000+seed,validation_seed=300000+seed))
                state["records"].extend(evaluate(best["dqn"], best["droo"], quality, seed, budget,
                    config["slots"], config["fading_values"], config["population"], config["iterations"],
                    checkpoint_root=checkpoint_root, checkpoint_every=checkpoint_every, resume=resume))
                state["completed_budgets"].append(budget)
                state["pending_budget"] = None
                commit()
                print(f"Completed seed {seed}, training budget {budget}", flush=True)
            if state["stream"].next_slot >= config["training_slots"]:
                break
            slot, args, _, _ = next(state["stream"])
            started = time.perf_counter()
            for name, agent in state["agents"].items():
                out = agent.run(*args, training=True)
                writer.writerow(dict(seed=seed, slot=slot+1, algorithm=name, cost=out.best_cost,
                                     epsilon=getattr(agent, "epsilon", 0.), updates=agent.updates,
                                     mean_loss=float(np.mean(agent.losses[-100:])) if agent.losses else ""))
            state["training_seconds"] += time.perf_counter()-started
            step = slot+1
            if step % config["validation_interval"] == 0 or step in config["training_budgets"]:
                for name, agent in state["agents"].items():
                    quality = validate(agent, bank)
                    state["validations"].append(dict(seed=seed, slot=step, algorithm=name, **quality))
                    selection_metric = config.get("selection_metric", "validation_log_cost")
                    if quality[selection_metric] < state["best_scores"][name]:
                        state["best"][name] = copy.deepcopy(agent)
                        state["best_quality"][name] = dict(quality, checkpoint_slot=step)
                        state["best_scores"][name] = quality[selection_metric]
            if step in config["training_budgets"]:
                state["qualities"].append(dict(seed=seed, training_budget=step, **state["best_quality"]))
                state["pending_budget"] = step
            if step % checkpoint_every == 0 or state["pending_budget"] is not None or time.perf_counter()-last_saved >= 60:
                commit()
        commit()
    completed = dict(complete=True, records=state["records"], validations=state["validations"], qualities=state["qualities"],
                     log_offset=state["log_offset"], timing=dict(seed=seed, training_seconds=state["training_seconds"],
                                                               total_seconds=state["elapsed_seconds"]))
    checkpoint.save(completed, compact=True)
    return completed, log_path
