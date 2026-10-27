"""Seeded repeated trials on shared simulation snapshots."""

import time
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd

from we_kb_fpa.config import (
    NetworkConfig,
    ChannelConfig,
    TaskConfig,
    CostConfig,
    OptimizerConfig,
    KnowledgeBaseConfig,
    SimulationConfig,
)
from we_kb_fpa.simulation_executor import SimulationExecutor, SlotResult
from we_kb_fpa.metrics_collector import MetricsCollector
from we_kb_fpa.exact_solver import ExactMINLPSolver
from we_kb_fpa.checkpoint import Checkpoint


def _run_single_trial_worker(
    trial_seed: int,
    slot_idx: int,
    net_cfg: NetworkConfig,
    ch_cfg: ChannelConfig,
    task_cfg: TaskConfig,
    cost_cfg: CostConfig,
    opt_cfg: OptimizerConfig,
    kb_cfg: KnowledgeBaseConfig,
    algorithms: List[str],
    run_exact: bool = False,
    param_value: Optional[float] = None,
    scenario_name: str = "default",
) -> List[Dict[str, Any]]:
    executor = SimulationExecutor(
        net_config=net_cfg,
        channel_config=ch_cfg,
        task_config=task_cfg,
        cost_config=cost_cfg,
        opt_config=opt_cfg,
        kb_config=kb_cfg,
        seed=trial_seed,
    )

    slot_res = executor.run_slot(
        slot_idx=slot_idx, algorithms=algorithms, run_exact=run_exact
    )

    records = []
    exact_cost = slot_res.exact_output.optimal_cost if slot_res.exact_output else None
    exact_runtime_ms = (
        slot_res.exact_output.solve_time_sec * 1000.0 if slot_res.exact_output else None
    )

    for algo_name, out in slot_res.algorithm_outputs.items():
        eval_res = out.evaluation_result

        lat_tx = sum(m.latency_tx for m in eval_res.task_metrics) / len(eval_res.task_metrics)
        lat_q = sum(m.latency_queue for m in eval_res.task_metrics) / len(eval_res.task_metrics)
        lat_c = sum(m.latency_comp for m in eval_res.task_metrics) / len(eval_res.task_metrics)
        lat_p = sum(m.latency_prop for m in eval_res.task_metrics) / len(eval_res.task_metrics)
        lat_dl = sum(m.latency_dl for m in eval_res.task_metrics) / len(eval_res.task_metrics)

        e_comm = sum(m.energy_comm for m in eval_res.task_metrics) / len(eval_res.task_metrics)
        e_comp = sum(m.energy_comp for m in eval_res.task_metrics) / len(eval_res.task_metrics)

        opt_gap = None
        reference_difference = None
        if exact_cost is not None and exact_cost > 0:
            reference_difference = ((eval_res.total_cost - exact_cost) / exact_cost) * 100.0

        records.append({
            "scenario": scenario_name,
            "run_id": trial_seed,
            "slot_id": slot_idx,
            "algorithm": algo_name,
            "cost_j": eval_res.total_cost,
            "latency_ms": eval_res.average_latency * 1000.0,
            "latency_tx_ms": lat_tx * 1000.0,
            "latency_queue_ms": lat_q * 1000.0,
            "latency_comp_ms": lat_c * 1000.0,
            "latency_prop_ms": lat_p * 1000.0,
            "latency_dl_ms": lat_dl * 1000.0,
            "energy_j": eval_res.average_energy,
            "energy_comm_j": e_comm,
            "energy_comp_j": e_comp,
            "violation_rate_pct": eval_res.deadline_violation_rate,
            "jains_fairness": eval_res.jains_fairness_index,
            "runtime_ms": out.wall_clock_time_sec * 1000.0,
            "iter_to_95": out.iterations_to_95_percent,
            "exact_cost": exact_cost,
            "exact_runtime_ms": exact_runtime_ms,
            "optimality_gap_pct": opt_gap,
            "reference_difference_pct": reference_difference,
            "reference_scope": slot_res.exact_output.optimality_scope if slot_res.exact_output else None,
            "reference_certified": slot_res.exact_output.is_optimal if slot_res.exact_output else None,
            "parameter_value": param_value,
            "convergence_curve": out.convergence_curve,
        })


    if slot_res.exact_output:
        ex = slot_res.exact_output
        records.append({
            "scenario": scenario_name,
            "run_id": trial_seed,
            "slot_id": slot_idx,
            "algorithm": ex.solver_name,
            "cost_j": ex.optimal_cost,
            "latency_ms": ex.evaluation_result.average_latency * 1000.0 if ex.evaluation_result else 0.0,
            "latency_tx_ms": 0.0,
            "latency_queue_ms": 0.0,
            "latency_comp_ms": 0.0,
            "latency_prop_ms": 0.0,
            "latency_dl_ms": 0.0,
            "energy_j": ex.evaluation_result.average_energy if ex.evaluation_result else 0.0,
            "energy_comm_j": 0.0,
            "energy_comp_j": 0.0,
            "violation_rate_pct": ex.evaluation_result.deadline_violation_rate,
            "jains_fairness": ex.evaluation_result.jains_fairness_index,
            "runtime_ms": ex.solve_time_sec * 1000.0,
            "iter_to_95": 1,
            "exact_cost": ex.optimal_cost,
            "exact_runtime_ms": ex.solve_time_sec * 1000.0,
            "optimality_gap_pct": None,
            "reference_difference_pct": 0.,
            "reference_scope": ex.optimality_scope,
            "reference_certified": ex.is_optimal,
            "reference_status": ex.termination_status,
            "reference_lower_bound": ex.lower_bound,
            "reference_relative_gap": ex.relative_gap,
            "reference_assignments_evaluated": ex.assignments_evaluated,
            "reference_assignments_total": ex.assignments_total,
            "parameter_value": param_value,
            "convergence_curve": [ex.optimal_cost],
        })

    return records


def _checkpointed_trial(kwargs, checkpoint_dir, resume, workers=1):
    checkpoint = Checkpoint(checkpoint_dir, "trial", dict(trial=kwargs, execution_workers=workers), resume) if checkpoint_dir else None
    records = checkpoint.load() if checkpoint else None
    if records is None:
        records = _run_single_trial_worker(**kwargs)
        if checkpoint:
            checkpoint.save(records, compact=True)
    return records


@contextmanager
def _trial_pool(workers):
    pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        yield pool
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)


class ExperimentRunner:

    def __init__(
        self,
        base_seed: int = 42,
        num_workers: Optional[int] = None,
        checkpoint_dir=None,
        resume=True,
    ):
        self.base_seed = base_seed
        self.num_workers = num_workers or 1
        self.checkpoint_dir = checkpoint_dir
        self.resume = resume
        if self.num_workers < 1:
            raise ValueError("num_workers must be positive")

    def run_sweep(
        self,
        scenario_name: str,
        param_values: List[float],
        param_updater_fn,
        algorithms: List[str],
        num_trials: int = 20,
        run_exact: bool = False,
    ) -> Tuple[pd.DataFrame, Dict[str, List[List[float]]]]:
        all_records = []
        convergence_map: Dict[str, List[List[float]]] = {a: [] for a in algorithms}

        print(f"=== Starting Experiment: {scenario_name} ({len(param_values)} points, {num_trials} trials/point) ===")

        with _trial_pool(self.num_workers) as pool:
            for val in param_values:
                print(f"  --> Simulating parameter point: {val} ...")

                net_cfg = NetworkConfig()
                ch_cfg = ChannelConfig()
                task_cfg = TaskConfig()
                cost_cfg = CostConfig()
                opt_cfg = OptimizerConfig(population_size=30, max_iterations=50)
                kb_cfg = KnowledgeBaseConfig()


                param_updater_fn(val, net_cfg, ch_cfg, task_cfg, cost_cfg, opt_cfg)

                pending = []
                for trial in range(num_trials):
                    trial_seed = self.base_seed + trial * 100 + int(abs(val) * 7)
                    kwargs = dict(
                        trial_seed=trial_seed,
                        slot_idx=trial,
                        net_cfg=net_cfg,
                        ch_cfg=ch_cfg,
                        task_cfg=task_cfg,
                        cost_cfg=cost_cfg,
                        opt_cfg=opt_cfg,
                        kb_cfg=kb_cfg,
                        algorithms=algorithms,
                        run_exact=run_exact,
                        param_value=val,
                        scenario_name=scenario_name,
                    )
                    if pool is None:
                        pending.append(_checkpointed_trial(kwargs, self.checkpoint_dir, self.resume, self.num_workers))
                    else:
                        pending.append(pool.submit(_checkpointed_trial, kwargs, self.checkpoint_dir, self.resume, self.num_workers))
                for item in pending:
                    recs = item.result() if pool is not None else item
                    for original in recs:
                        r = original.copy()
                        algo = r["algorithm"]
                        curve = r.pop("convergence_curve", None)
                        if curve and algo in convergence_map:
                            convergence_map[algo].append(curve)
                        all_records.append(r)

        df = pd.DataFrame(all_records)
        return df, convergence_map
