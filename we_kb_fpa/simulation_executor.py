"""Time-slotted mobility, workload, channel, and allocation simulation."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import numpy as np

from we_kb_fpa.config import (
    SimulationConfig,
    NetworkConfig,
    ChannelConfig,
    TaskConfig,
    CostConfig,
    OptimizerConfig,
    KnowledgeBaseConfig,
)
from we_kb_fpa.environment_model import NetworkEnvironment
from we_kb_fpa.channel_model import WirelessChannelSimulator
from we_kb_fpa.task_model import TaskWorkloadGenerator, Task
from we_kb_fpa.knowledge_base import WirelessEdgeKnowledgeBase
from we_kb_fpa.cost_engine import CostEngine, SystemEvaluationResult
from we_kb_fpa.we_kb_fpa import WEKBFPAOptimizer, OptimizationOutput
from we_kb_fpa.baselines import BaselineRunner
from we_kb_fpa.dqn import DQNBaseline
from we_kb_fpa.droo import DROOAdapted
from we_kb_fpa.exact_solver import ExactMINLPSolver, ExactSolverResult


@dataclass
class SlotResult:
    slot_index: int
    timestamp: float
    channel_variance: float
    mean_sinr_db: float
    num_tasks: int
    algorithm_outputs: Dict[str, OptimizationOutput] = field(default_factory=dict)
    exact_output: Optional[ExactSolverResult] = None


class SimulationExecutor:

    def __init__(
        self,
        sim_config: Optional[SimulationConfig] = None,
        net_config: Optional[NetworkConfig] = None,
        channel_config: Optional[ChannelConfig] = None,
        task_config: Optional[TaskConfig] = None,
        cost_config: Optional[CostConfig] = None,
        opt_config: Optional[OptimizerConfig] = None,
        kb_config: Optional[KnowledgeBaseConfig] = None,
        seed: int = 42,
    ):
        self.sim_cfg = sim_config or SimulationConfig()
        self.net_cfg = net_config or NetworkConfig()
        self.ch_cfg = channel_config or ChannelConfig()
        self.task_cfg = task_config or TaskConfig()
        self.cost_cfg = cost_config or CostConfig()
        self.opt_cfg = opt_config or OptimizerConfig()
        self.kb_cfg = kb_config or KnowledgeBaseConfig()
        self.seed = seed
        self.dqn = DQNBaseline(self.opt_cfg, self.cost_cfg, self.net_cfg, seed=seed)
        self.droo = DROOAdapted(self.cost_cfg, self.net_cfg, seed=seed)


        self.env = NetworkEnvironment(self.net_cfg, seed=seed)
        self.channel = WirelessChannelSimulator(self.ch_cfg, self.net_cfg, seed=seed)
        self.workload_gen = TaskWorkloadGenerator(self.task_cfg, seed=seed)
        self.knowledge_base = WirelessEdgeKnowledgeBase(self.kb_cfg, self.net_cfg)
        self.baseline_runner = BaselineRunner(self.opt_cfg, self.cost_cfg, self.net_cfg, seed=seed)
        self.exact_solver = ExactMINLPSolver(self.cost_cfg, self.net_cfg, time_limit_sec=30.0)


        self._warmup_knowledge_base()

    def _warmup_knowledge_base(self):
        warmup_tasks = self.workload_gen.generate_slot_tasks(self.net_cfg.num_users, 0.0)
        dist_mat = self.env.get_distance_matrix()
        comp_gains, _ = self.channel.update_channel_states(dist_mat)
        p_tx = np.array([u.p_tx for u in self.env.users])
        sinr = self.channel.compute_sinr_matrix(comp_gains, p_tx)


        greedy_res = self.baseline_runner.run_greedy_latency(
            warmup_tasks, self.env.users, self.env.edge_servers, self.env.cloud, sinr, dist_mat
        )
        self.knowledge_base.add_experience(
            sinr,
            warmup_tasks,
            dist_mat,
            greedy_res.best_targets,
            greedy_res.best_cpu,
            greedy_res.best_bw,
            greedy_res.best_cost,
        )

    def sample_slot(self, slot_idx, fading_db=0., faded_edges=None):
        dt = self.sim_cfg.time_step
        current_time = slot_idx * dt


        self.env.step_mobility(dt)
        dist_mat = self.env.get_distance_matrix()


        comp_gains, fading = self.channel.update_channel_states(dist_mat)
        p_tx = np.array([u.p_tx for u in self.env.users])
        sinr = self.channel.compute_sinr_matrix(comp_gains, p_tx)
        # Apply an exogenous link shock before any algorithm sees the snapshot.
        if not np.isfinite(fading_db) or fading_db < 0:
            raise ValueError("fading_db must be finite and nonnegative")
        affected = list(range(sinr.shape[1])) if faded_edges is None else faded_edges
        if any(i < 0 or i >= sinr.shape[1] for i in affected):
            raise ValueError("faded_edges contains an invalid zero-based edge index")
        sinr[:, affected] *= 10.0 ** (-fading_db / 10.0)
        ch_variance = self.channel.get_channel_variance()
        mean_sinr_db = float(10.0 * np.log10(np.mean(sinr)))


        tasks = self.workload_gen.generate_slot_tasks(self.net_cfg.num_users, current_time)


        return tasks, sinr, dist_mat, ch_variance, mean_sinr_db

    def run_slot(
        self,
        slot_idx: int,
        algorithms: Optional[List[str]] = None,
        run_exact: bool = False,
        fading_db: float = 0.0,
        faded_edges: Optional[List[int]] = None,
        dqn_training: bool = False,
        droo_training: bool = False,
    ) -> SlotResult:
        dt = self.sim_cfg.time_step
        current_time = slot_idx * dt
        tasks, sinr, dist_mat, ch_variance, mean_sinr_db = self.sample_slot(slot_idx, fading_db, faded_edges)

        algos = algorithms or [
            "Random",
            "Greedy-Latency",
            "Greedy-Energy",
            "PSO",
            "GA",
            "FPA-Baseline",
            "KB-FPA",
            "WE-FPA",
            "WE-KB-FPA",
        ]

        slot_res = SlotResult(
            slot_index=slot_idx,
            timestamp=current_time,
            channel_variance=ch_variance,
            mean_sinr_db=mean_sinr_db,
            num_tasks=len(tasks),
        )

        for algo in algos:
            if algo == "DQN":
                out = self.dqn.run(tasks, self.env.users, self.env.edge_servers,
                                   self.env.cloud, sinr, dist_mat, training=dqn_training)
            elif algo == "DROO-Adapted":
                out = self.droo.run(tasks, self.env.users, self.env.edge_servers,
                                    self.env.cloud, sinr, dist_mat, training=droo_training)
            elif algo == "Random":
                out = self.baseline_runner.run_random(
                    tasks, self.env.users, self.env.edge_servers, self.env.cloud, sinr, dist_mat
                )
            elif algo == "Greedy-Latency":
                out = self.baseline_runner.run_greedy_latency(
                    tasks, self.env.users, self.env.edge_servers, self.env.cloud, sinr, dist_mat
                )
            elif algo == "Greedy-Energy":
                out = self.baseline_runner.run_greedy_energy(
                    tasks, self.env.users, self.env.edge_servers, self.env.cloud, sinr, dist_mat
                )
            elif algo == "PSO":
                out = self.baseline_runner.run_pso(
                    tasks, self.env.users, self.env.edge_servers, self.env.cloud, sinr, dist_mat
                )
            elif algo == "GA":
                out = self.baseline_runner.run_ga(
                    tasks, self.env.users, self.env.edge_servers, self.env.cloud, sinr, dist_mat
                )
            elif algo == "FPA-Baseline":
                fpa = WEKBFPAOptimizer(
                    self.opt_cfg,
                    self.cost_cfg,
                    self.net_cfg,
                    use_kb=False,
                    use_channel_adaptive=False,
                    seed=self.seed + slot_idx,
                )
                out = fpa.optimize(
                    tasks,
                    self.env.users,
                    self.env.edge_servers,
                    self.env.cloud,
                    sinr,
                    dist_mat,
                    ch_variance,
                )
            elif algo == "KB-FPA":
                kb_fpa = WEKBFPAOptimizer(
                    self.opt_cfg,
                    self.cost_cfg,
                    self.net_cfg,
                    use_kb=True,
                    use_channel_adaptive=False,
                    seed=self.seed + slot_idx,
                )
                out = kb_fpa.optimize(
                    tasks,
                    self.env.users,
                    self.env.edge_servers,
                    self.env.cloud,
                    sinr,
                    dist_mat,
                    ch_variance,
                    self.knowledge_base,
                )
            elif algo == "WE-FPA":
                we_fpa = WEKBFPAOptimizer(
                    self.opt_cfg,
                    self.cost_cfg,
                    self.net_cfg,
                    use_kb=False,
                    use_channel_adaptive=True,
                    seed=self.seed + slot_idx,
                )
                out = we_fpa.optimize(
                    tasks,
                    self.env.users,
                    self.env.edge_servers,
                    self.env.cloud,
                    sinr,
                    dist_mat,
                    ch_variance,
                )
            elif algo in ("WE-KB-FPA", "WE-KB-FPA-Uniform"):
                we_kb_fpa = WEKBFPAOptimizer(
                    self.opt_cfg,
                    self.cost_cfg,
                    self.net_cfg,
                    use_kb=True,
                    use_channel_adaptive=True,
                    seed=self.seed + slot_idx,
                )
                out = we_kb_fpa.optimize(
                    tasks,
                    self.env.users,
                    self.env.edge_servers,
                    self.env.cloud,
                    sinr,
                    dist_mat,
                    ch_variance,
                    self.knowledge_base,
                    allocation_mode="uniform" if algo.endswith("Uniform") else "joint",
                )
            else:
                raise ValueError(f"Unknown algorithm: {algo}")

            slot_res.algorithm_outputs[algo] = out


        if run_exact and len(tasks) <= 8 and len(self.env.edge_servers) <= 3:
            exact_res = self.exact_solver.solve(
                tasks, self.env.users, self.env.edge_servers, self.env.cloud, sinr, dist_mat
            )
            slot_res.exact_output = exact_res


        best_algo = min(
            slot_res.algorithm_outputs.keys(),
            key=lambda a: slot_res.algorithm_outputs[a].best_cost,
        )
        best_out = slot_res.algorithm_outputs[best_algo]
        self.knowledge_base.add_experience(
            sinr,
            tasks,
            dist_mat,
            best_out.best_targets,
            best_out.best_cpu,
            best_out.best_bw,
            best_out.best_cost,
        )


        self.env.update_server_queues(dt)

        return slot_res
