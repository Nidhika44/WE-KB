"""Held-out learning diagnostics and snapshot decision-delay sensitivity."""
import copy
import hashlib
import numpy as np
from we_kb_fpa.allocation import uniform_allocation, binary_greedy_allocation
from we_kb_fpa.simulation_executor import SimulationExecutor
from we_kb_fpa.config import NetworkConfig, OptimizerConfig


class SnapshotStream:
    def __init__(self, seed, count, population=20, iterations=30, fading_db=0., midpoint=None):
        self.sim = SimulationExecutor(net_config=NetworkConfig(num_users=8, num_edge_servers=3),
                                      opt_config=OptimizerConfig(population_size=population, max_iterations=iterations), seed=seed)
        self.count, self.next_slot = count, 0
        self.fading_db, self.midpoint = fading_db, midpoint

    def __iter__(self):
        return self

    def __next__(self):
        slot = self.next_slot
        if slot >= self.count:
            raise StopIteration
        if slot:
            self.sim.env.update_server_queues(self.sim.sim_cfg.time_step)
        fade = self.fading_db if self.midpoint is not None and slot >= self.midpoint else 0.
        tasks, sinr, distances, variance, mean_db = self.sim.sample_slot(slot, fading_db=fade)
        args = (tasks, self.sim.env.users, self.sim.env.edge_servers, self.sim.env.cloud, sinr, distances)
        self.next_slot += 1
        return slot, args, variance, mean_db


def snapshots(seed, count, population=20, iterations=30, fading_db=0., midpoint=None):
    return SnapshotStream(seed, count, population, iterations, fading_db, midpoint)


def fingerprint(args):
    tasks, users, edges, cloud, sinr, distances = args
    digest = hashlib.sha256()
    for value in ([t.data_size_bits for t in tasks], [t.total_cycles for t in tasks],
                  [t.deadline_sec for t in tasks], [u.f_local for u in users],
                  [u.p_tx for u in users], [e.f_total for e in edges],
                  [e.bandwidth_total for e in edges], [e.queue_delay_current for e in edges],
                  [cloud.f_total, cloud.queue_delay_current], sinr, distances):
        digest.update(np.asarray(value, dtype=np.float64).tobytes())
    return digest.hexdigest()


def validation_bank(seed, count):
    from we_kb_fpa.baselines import BaselineRunner
    runner = BaselineRunner(seed=seed)
    bank = []
    for _, live_args, _, _ in snapshots(seed, count):
        args = copy.deepcopy(live_args)
        tasks, users, edges, cloud, sinr, distances = args
        greedy = runner.run_greedy_latency(*args)
        allocation = uniform_allocation(greedy.best_targets, tasks, users, edges, cloud, distances)
        greedy_cost = runner.cost_engine.evaluate_allocation(tasks, users, edges, cloud, *allocation, sinr, distances).total_cost
        allocation = uniform_allocation(np.zeros(len(tasks), dtype=int), tasks, users, edges, cloud, distances)
        local_cost = runner.cost_engine.evaluate_allocation(tasks, users, edges, cloud, *allocation, sinr, distances).total_cost
        bank.append((args, greedy_cost, local_cost))
    return bank


def validate(agent, bank):
    from we_kb_fpa.droo import DROOAdapted
    costs, scores, greedy, local = [], [], [], []
    for args, greedy_cost, local_cost in bank:
        if isinstance(agent,DROOAdapted):
            tasks,users,edges,cloud,sinr,distances=args
            allocation=binary_greedy_allocation(*args,agent.engine.net_cfg)
            greedy_cost=agent.engine.evaluate_allocation(tasks,users,edges,cloud,*allocation,sinr,distances).total_cost
        cost = agent.run(*args, training=False).best_cost
        costs.append(cost)
        scores.append(np.log1p(cost / len(args[0])))
        greedy.append(greedy_cost)
        local.append(local_cost)
    mean = float(np.mean(costs))
    reference = min(float(np.mean(greedy)), float(np.mean(local)))
    return dict(reference_action_space='binary_nearest_edge' if isinstance(agent,DROOAdapted) else 'local_all_edges_cloud',
                validation_log_cost=float(np.mean(scores)), validation_cost=mean,
                validation_greedy_cost=float(np.mean(greedy)), validation_local_cost=float(np.mean(local)),
                nominal_cost_ratio=mean / max(reference, 1e-12), nominal_quality_pass=mean <= 1.2 * reference)


def save_agent(agent, path):
    data = {f"weight_{i}": w for i, w in enumerate(agent.weights)}
    if hasattr(agent, "target"):
        data.update({f"target_{i}": w for i, w in enumerate(agent.target)})
    np.savez(path, **data, updates=agent.updates, epsilon=getattr(agent, "epsilon", 0.))


def delay_adjusted_metrics(output, cost_config, elapsed_seconds=None):
    """Conservative serialized decision+learning delay, without queue evolution.

This sensitivity calculation holds the channel and allocation fixed. It is
not a continuous-time deployment model and does not model stale decisions.
"""
    delay = output.wall_clock_time_sec if elapsed_seconds is None else elapsed_seconds
    if not np.isfinite(delay) or delay < 0:
        raise ValueError("Decision delay must be finite and nonnegative")
    metrics = output.evaluation_result.task_metrics
    cost = output.best_cost
    violations = 0
    for task in metrics:
        before = task.latency_total / task.deadline_sec
        after = (task.latency_total + delay) / task.deadline_sec
        cost += cost_config.alpha * delay / task.deadline_sec
        cost += cost_config.penalty_deadline_multiplier * (
            (after ** 2 if after > 1 else 0.) - (before ** 2 if before > 1 else 0.))
        violations += after > 1
    return dict(delay_adjusted_cost=float(cost),
                delay_adjusted_latency_ms=output.evaluation_result.average_latency * 1000 + delay * 1000,
                delay_adjusted_violation_pct=100 * violations / len(metrics))
