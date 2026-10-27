"""Channel-adaptive flower pollination with knowledge-base initialization."""

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import numpy as np
from scipy.special import gamma as gamma_fn

from we_kb_fpa.config import OptimizerConfig, CostConfig, NetworkConfig
from we_kb_fpa.environment_model import UserDevice, EdgeServer, CloudServer
from we_kb_fpa.task_model import Task
from we_kb_fpa.cost_engine import CostEngine, SystemEvaluationResult
from we_kb_fpa.knowledge_base import WirelessEdgeKnowledgeBase
from we_kb_fpa.allocation import uniform_allocation


@dataclass
class OptimizationOutput:
    best_cost: float
    best_targets: np.ndarray
    best_cpu: np.ndarray
    best_bw: np.ndarray
    convergence_curve: List[float]
    wall_clock_time_sec: float
    iterations_to_95_percent: int
    evaluation_result: SystemEvaluationResult
    algorithm_name: str


class WEKBFPAOptimizer:

    def __init__(
        self,
        opt_config: Optional[OptimizerConfig] = None,
        cost_config: Optional[CostConfig] = None,
        net_config: Optional[NetworkConfig] = None,
        use_kb: bool = True,
        use_channel_adaptive: bool = True,
        seed: int = 42,
    ):
        self.opt_cfg = opt_config or OptimizerConfig()
        self.cost_cfg = cost_config or CostConfig()
        self.net_cfg = net_config or NetworkConfig()
        self.use_kb = use_kb
        self.use_channel_adaptive = use_channel_adaptive
        self.rng = np.random.default_rng(seed)
        self.cost_engine = CostEngine(self.cost_cfg, self.net_cfg)

    def _sample_mantegna_levy(self, size: int) -> np.ndarray:
        lamb = self.opt_cfg.levy_exponent

        num = gamma_fn(1.0 + lamb) * np.sin(np.pi * lamb / 2.0)
        den = gamma_fn((1.0 + lamb) / 2.0) * lamb * (2.0 ** ((lamb - 1.0) / 2.0))
        sigma_u = (num / den) ** (1.0 / lamb)

        u = self.rng.normal(0.0, sigma_u, size=size)
        v = self.rng.normal(0.0, 1.0, size=size)
        step = u / (np.abs(v) ** (1.0 / lamb))
        return step

    def _compute_dynamic_switch_prob(self, channel_variance: float) -> float:
        if not self.use_channel_adaptive:
            return self.opt_cfg.base_switch_prob

        theta = self.opt_cfg.csi_variance_theta
        # p_s(t) = p_base * (1 - exp(-Var(H_t) / theta))

        factor = 1.0 - np.exp(-channel_variance / max(1e-4, theta))
        p_s = self.opt_cfg.base_switch_prob * factor

        return float(np.clip(p_s, 0.20, 0.90))

    def _compute_channel_guided_step_sizes(
        self, sinr_matrix: np.ndarray, current_targets: np.ndarray
    ) -> np.ndarray:
        """Scales Levy flight step size gamma_i by instantaneous SINR degradation.

        gamma_i = gamma_0 * (1 + beta * (SINR_ref / max(SINR_i, epsilon)))
        """
        N = len(current_targets)
        if not self.use_channel_adaptive:
            return np.full(N, self.opt_cfg.gamma_base)

        gamma_0 = self.opt_cfg.gamma_base
        beta = self.opt_cfg.sinr_scaling_beta
        sinr_ref_lin = 10.0 ** (self.opt_cfg.sinr_ref_db / 10.0)

        step_scales = np.zeros(N, dtype=np.float64)
        for i in range(N):
            tgt = int(current_targets[i])
            if 1 <= tgt <= sinr_matrix.shape[1]:

                sinr_val = sinr_matrix[i, tgt - 1]
            else:

                sinr_val = sinr_ref_lin

            degradation_ratio = sinr_ref_lin / max(sinr_val, 1e-4)

            amp = 1.0 + beta * np.clip(degradation_ratio, 0.1, 5.0)
            step_scales[i] = gamma_0 * amp

        return step_scales

    def decode_chromosome(
        self,
        chromosome: np.ndarray,
        tasks: List[Task],
        users: List[UserDevice],
        edge_servers: List[EdgeServer],
        cloud: CloudServer,
        pruning_mask: Optional[np.ndarray] = None,
        distances: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Decodes 3N continuous vector into (targets, cpu_allocations, bw_allocations).

        Chromosome layout:
        [0 : N] -> Target node continuous indicator in [0, M+1]
        [N : 2N] -> CPU fraction indicator in [0, 1]
        [2N : 3N] -> Bandwidth fraction indicator in [0, 1]
        """
        N = len(tasks)
        M = len(edge_servers)
        num_targets = M + 2

        raw_targets = chromosome[0:N]
        raw_cpu = chromosome[N : 2 * N]
        raw_bw = chromosome[2 * N : 3 * N]


        targets = np.round(raw_targets).astype(np.int32)
        targets = np.clip(targets, 0, num_targets - 1)


        if self.use_kb and pruning_mask is not None:
            for i in range(N):
                if pruning_mask[i, targets[i]] == 0:

                    feasible_indices = np.where(pruning_mask[i] == 1)[0]
                    if len(feasible_indices) > 0:
                        closest_idx = feasible_indices[
                            np.argmin(np.abs(feasible_indices - targets[i]))
                        ]
                        targets[i] = closest_idx


        if distances is None:
            distances = getattr(self, "current_distances", None)
        if distances is None:
            distances = np.array([[np.linalg.norm([u.x-e.x, u.y-e.y, u.z-e.z]) for e in edge_servers] for u in users])
        if getattr(self, "allocation_mode", "joint") == "uniform":
            return uniform_allocation(targets, tasks, users, edge_servers, cloud, distances)

        cpu_allocs = np.zeros(N, dtype=np.float64)
        for i in range(N):
            if targets[i] == 0:
                cpu_allocs[i] = users[i].f_local


        for m in range(1, M + 1):
            m_users = [i for i in range(N) if targets[i] == m]
            if m_users:
                edge_f_total = edge_servers[m - 1].f_total
                shares = np.maximum(1e-4, raw_cpu[m_users])
                norm_shares = shares / np.sum(shares)
                for u_idx, i in enumerate(m_users):
                    cpu_allocs[i] = norm_shares[u_idx] * edge_f_total


        cloud_users = [i for i in range(N) if targets[i] == (M + 1)]
        if cloud_users:
            shares = np.maximum(1e-4, raw_cpu[cloud_users])
            norm_shares = shares / np.sum(shares)
            for u_idx, i in enumerate(cloud_users):
                cpu_allocs[i] = norm_shares[u_idx] * cloud.f_total


        bw_allocs = np.zeros(N, dtype=np.float64)
        access = targets - 1
        cloud_mask = targets == M + 1
        access[cloud_mask] = np.argmin(distances[cloud_mask], axis=1)
        for m in range(M):
            selected = np.where((targets != 0) & (access == m))[0]
            if selected.size:
                shares = np.maximum(1e-4, raw_bw[selected])
                bw_allocs[selected] = shares / shares.sum()

        return targets, cpu_allocs, bw_allocs

    def optimize(
        self,
        tasks: List[Task],
        users: List[UserDevice],
        edge_servers: List[EdgeServer],
        cloud: CloudServer,
        sinr_matrix: np.ndarray,
        distances: np.ndarray,
        channel_variance: float = 0.5,
        knowledge_base: Optional[WirelessEdgeKnowledgeBase] = None,
        allocation_mode: str = "joint",
    ) -> OptimizationOutput:
        if allocation_mode not in ("joint", "uniform"):
            raise ValueError("Unknown allocation_mode")
        self.allocation_mode = allocation_mode
        self.current_distances = distances
        start_time = time.perf_counter()
        N = len(tasks)
        M = len(edge_servers)
        num_targets = M + 2
        pop_size = self.opt_cfg.population_size
        max_iters = self.opt_cfg.max_iterations


        pruning_mask = None
        if self.use_kb and knowledge_base is not None:
            pruning_mask = knowledge_base.generate_pruning_mask(
                tasks, users, edge_servers, cloud, sinr_matrix, distances
            )


        chrom_len = 3 * N
        population = np.zeros((pop_size, chrom_len), dtype=np.float64)
        kb_seeded_count = 0


        if self.use_kb and knowledge_base is not None and len(knowledge_base.records) > 0:
            target_seed_count = int(pop_size * self.opt_cfg.kb_seeding_percentage)
            nearest = knowledge_base.query_nearest_solutions(sinr_matrix, tasks, distances, k=target_seed_count)
            for rec, _ in nearest:
                if kb_seeded_count >= target_seed_count:
                    break

                rec_targets = rec.solution_targets
                if len(rec_targets) == N:
                    population[kb_seeded_count, 0:N] = rec_targets.astype(np.float64)
                else:
                    population[kb_seeded_count, 0:N] = self.rng.choice(num_targets, size=N)

                population[kb_seeded_count, N : 2 * N] = self.rng.uniform(0.1, 1.0, size=N)
                population[kb_seeded_count, 2 * N : 3 * N] = self.rng.uniform(0.1, 1.0, size=N)
                kb_seeded_count += 1


        for p in range(kb_seeded_count, pop_size):
            if self.use_kb and pruning_mask is not None:

                for i in range(N):
                    feas = np.where(pruning_mask[i] == 1)[0]
                    population[p, i] = float(self.rng.choice(feas)) if len(feas) > 0 else 0.0
            else:
                population[p, 0:N] = self.rng.uniform(0.0, num_targets - 1, size=N)

            population[p, N : 2 * N] = self.rng.uniform(0.1, 1.0, size=N)
            population[p, 2 * N : 3 * N] = self.rng.uniform(0.1, 1.0, size=N)


        fitness = np.zeros(pop_size, dtype=np.float64)
        eval_cache: List[SystemEvaluationResult] = []

        for p in range(pop_size):
            tgt, cpu, bw = self.decode_chromosome(
                population[p], tasks, users, edge_servers, cloud, pruning_mask
            )
            res = self.cost_engine.evaluate_allocation(
                tasks, users, edge_servers, cloud, tgt, cpu, bw, sinr_matrix, distances
            )
            fitness[p] = res.total_cost
            eval_cache.append(res)

        best_idx = int(np.argmin(fitness))
        global_best_chrom = population[best_idx].copy()
        global_best_cost = fitness[best_idx]
        global_best_eval = eval_cache[best_idx]

        convergence_curve: List[float] = [global_best_cost]


        p_s = self._compute_dynamic_switch_prob(channel_variance)


        for iteration in range(max_iters):
            for i in range(pop_size):
                curr_targets = population[i, 0:N]

                if self.rng.uniform(0.0, 1.0) < p_s:

                    levy_step = self._sample_mantegna_levy(chrom_len)

                    step_scales_target = self._compute_channel_guided_step_sizes(
                        sinr_matrix, curr_targets
                    )
                    full_step_scales = np.concatenate([
                        step_scales_target,
                        np.full(2 * N, self.opt_cfg.gamma_base),
                    ])

                    # P_i^(t+1) = P_i^t + gamma * L * (g_best - P_i^t)
                    candidate = population[i] + full_step_scales * levy_step * (
                        global_best_chrom - population[i]
                    )
                else:

                    j, k = self.rng.choice(pop_size, size=2, replace=False)
                    epsilon = self.rng.uniform(0.0, 1.0)
                    # P_i^(t+1) = P_i^t + epsilon * (P_j^t - P_k^t)
                    candidate = population[i] + epsilon * (population[j] - population[k])


                candidate[0:N] = np.clip(candidate[0:N], 0.0, num_targets - 1)
                candidate[N : 3 * N] = np.clip(candidate[N : 3 * N], 0.01, 1.0)


                c_tgt, c_cpu, c_bw = self.decode_chromosome(
                    candidate, tasks, users, edge_servers, cloud, pruning_mask
                )
                c_res = self.cost_engine.evaluate_allocation(
                    tasks, users, edge_servers, cloud, c_tgt, c_cpu, c_bw, sinr_matrix, distances
                )


                if c_res.total_cost < fitness[i]:
                    population[i] = candidate
                    fitness[i] = c_res.total_cost
                    eval_cache[i] = c_res

                    if c_res.total_cost < global_best_cost:
                        global_best_cost = c_res.total_cost
                        global_best_chrom = candidate.copy()
                        global_best_eval = c_res

            convergence_curve.append(global_best_cost)

        duration = time.perf_counter() - start_time


        final_cost = global_best_cost
        init_cost = convergence_curve[0]
        cost_delta = init_cost - final_cost
        target_95 = init_cost - 0.95 * cost_delta
        iter_95 = max_iters
        for it, cost_val in enumerate(convergence_curve):
            if cost_val <= target_95:
                iter_95 = it
                break


        best_tgt, best_cpu, best_bw = self.decode_chromosome(
            global_best_chrom, tasks, users, edge_servers, cloud, pruning_mask
        )

        algo_name = "WE-KB-FPA"
        if not self.use_kb and not self.use_channel_adaptive:
            algo_name = "FPA-Baseline"
        elif self.use_kb and not self.use_channel_adaptive:
            algo_name = "KB-FPA"
        elif not self.use_kb and self.use_channel_adaptive:
            algo_name = "WE-FPA"

        return OptimizationOutput(
            best_cost=global_best_cost,
            best_targets=best_tgt,
            best_cpu=best_cpu,
            best_bw=best_bw,
            convergence_curve=convergence_curve,
            wall_clock_time_sec=duration,
            iterations_to_95_percent=iter_95,
            evaluation_result=global_best_eval,
            algorithm_name=algo_name,
        )
