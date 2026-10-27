"""Certified assignment enumeration restricted to the shared uniform allocator.

This is not a global reference for joint continuous resource optimization.
The legacy class name is retained for callers; the invalid MINLP formulation
is intentionally not used. Every incumbent is scored by the simulator.
"""
from dataclasses import dataclass
from itertools import product
import time
import numpy as np
from we_kb_fpa.allocation import uniform_allocation
from we_kb_fpa.baselines import BaselineRunner
from we_kb_fpa.config import CostConfig, NetworkConfig
from we_kb_fpa.cost_engine import CostEngine, SystemEvaluationResult

try:
    import gurobipy
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False


@dataclass
class ExactSolverResult:
    optimal_cost: float
    offloading_targets: np.ndarray
    cpu_allocations: np.ndarray
    bandwidth_allocations: np.ndarray
    solve_time_sec: float
    is_optimal: bool
    solver_name: str
    evaluation_result: SystemEvaluationResult | None = None
    optimality_scope: str = "uniform_allocator_only"
    termination_status: str = "UNKNOWN"
    assignments_evaluated: int = 0
    assignments_total: int = 0
    lower_bound: float = 0.
    relative_gap: float | None = None


class ExactMINLPSolver:
    def __init__(self, cost_config=None, net_config=None, time_limit_sec=60.):
        if not np.isfinite(time_limit_sec) or time_limit_sec <= 0:
            raise ValueError("Reference time limit must be finite and positive")
        self.cost_cfg = cost_config or CostConfig()
        self.net_cfg = net_config or NetworkConfig()
        self.time_limit = time_limit_sec
        self.cost_engine = CostEngine(self.cost_cfg, self.net_cfg)

    def solve(self, tasks, users, edge_servers, cloud, sinr_matrix, distances):
        start = time.perf_counter()
        args = tasks, users, edge_servers, cloud, sinr_matrix, distances
        n, choices = len(tasks), len(edge_servers)+2
        if not n:
            raise ValueError("Reference requires at least one task")
        incumbent = BaselineRunner(cost_config=self.cost_cfg, net_config=self.net_cfg).run_greedy_latency(*args)
        allocation = (incumbent.best_targets, incumbent.best_cpu, incumbent.best_bw)
        evaluation = incumbent.evaluation_result
        evaluated, total = 0, choices**n
        for targets in product(range(choices), repeat=n):
            if time.perf_counter()-start >= self.time_limit:
                break
            candidate = uniform_allocation(np.array(targets,dtype=np.int32), tasks, users, edge_servers, cloud, distances)
            result = self.cost_engine.evaluate_allocation(tasks,users,edge_servers,cloud,*candidate,sinr_matrix,distances)
            evaluated += 1
            if result.total_cost < evaluation.total_cost:
                allocation, evaluation = candidate, result
        complete = evaluated == total
        return ExactSolverResult(evaluation.total_cost,*allocation,time.perf_counter()-start,complete,
                                 "Uniform-assignment enumeration",evaluation,
                                 termination_status="EXHAUSTED" if complete else "TIME_LIMIT",
                                 assignments_evaluated=evaluated,assignments_total=total,
                                 lower_bound=evaluation.total_cost if complete else 0.,
                                 relative_gap=0. if complete else None)
