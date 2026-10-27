"""Heuristic, PSO, GA, and FPA comparison algorithms."""

import time
from typing import List, Optional, Tuple
import numpy as np

from we_kb_fpa.config import OptimizerConfig, CostConfig, NetworkConfig
from we_kb_fpa.environment_model import UserDevice, EdgeServer, CloudServer
from we_kb_fpa.task_model import Task
from we_kb_fpa.cost_engine import CostEngine, SystemEvaluationResult
from we_kb_fpa.we_kb_fpa import OptimizationOutput, WEKBFPAOptimizer


class BaselineRunner:

    def __init__(
        self,
        opt_config: Optional[OptimizerConfig] = None,
        cost_config: Optional[CostConfig] = None,
        net_config: Optional[NetworkConfig] = None,
        seed: int = 42,
    ):
        self.opt_cfg = opt_config or OptimizerConfig()
        self.cost_cfg = cost_config or CostConfig()
        self.net_cfg = net_config or NetworkConfig()
        self.cost_engine = CostEngine(self.cost_cfg, self.net_cfg)
        self.rng = np.random.default_rng(seed)

    def run_random(
        self,
        tasks: List[Task],
        users: List[UserDevice],
        edge_servers: List[EdgeServer],
        cloud: CloudServer,
        sinr_matrix: np.ndarray,
        distances: np.ndarray,
    ) -> OptimizationOutput:
        start_time = time.perf_counter()
        N = len(tasks)
        M = len(edge_servers)
        num_targets = M + 2

        targets = self.rng.integers(0, num_targets, size=N)
        raw_cpu = self.rng.uniform(0.1, 1.0, size=N)
        raw_bw = self.rng.uniform(0.1, 1.0, size=N)


        fpa_helper = WEKBFPAOptimizer(
            self.opt_cfg, self.cost_cfg, self.net_cfg, use_kb=False, use_channel_adaptive=False
        )
        chrom = np.concatenate([targets.astype(np.float64), raw_cpu, raw_bw])
        best_tgt, best_cpu, best_bw = fpa_helper.decode_chromosome(
            chrom, tasks, users, edge_servers, cloud, distances=distances
        )

        res = self.cost_engine.evaluate_allocation(
            tasks, users, edge_servers, cloud, best_tgt, best_cpu, best_bw, sinr_matrix, distances
        )
        duration = time.perf_counter() - start_time

        return OptimizationOutput(
            best_cost=res.total_cost,
            best_targets=best_tgt,
            best_cpu=best_cpu,
            best_bw=best_bw,
            convergence_curve=[res.total_cost],
            wall_clock_time_sec=duration,
            iterations_to_95_percent=1,
            evaluation_result=res,
            algorithm_name="Random",
        )

    def run_greedy_latency(
        self,
        tasks: List[Task],
        users: List[UserDevice],
        edge_servers: List[EdgeServer],
        cloud: CloudServer,
        sinr_matrix: np.ndarray,
        distances: np.ndarray,
    ) -> OptimizationOutput:
        start_time = time.perf_counter()
        N = len(tasks)
        M = len(edge_servers)
        num_targets = M + 2

        targets = np.zeros(N, dtype=np.int32)
        nominal_bw = 1.0 / max(1, N // max(1, M))
        nominal_f = 2.0e9

        for i in range(N):
            task = tasks[i]
            user = users[i]
            latencies = np.zeros(num_targets)


            latencies[0] = task.total_cycles / user.f_local


            for m in range(1, M + 1):
                m_idx = m - 1
                edge = edge_servers[m_idx]
                r = nominal_bw * edge.bandwidth_total * np.log2(1.0 + sinr_matrix[i, m_idx])
                tau_tx = task.data_size_bits / max(1e3, r)
                tau_c = task.total_cycles / min(edge.f_total, nominal_f)
                tau_p = distances[i, m_idx] / 3.0e8
                latencies[m] = tau_tx + tau_c + tau_p + edge.queue_delay_current + self.net_cfg.d_dl_constant


            best_e = int(np.argmin(distances[i]))
            r_c = nominal_bw * edge_servers[best_e].bandwidth_total * np.log2(1.0 + sinr_matrix[i, best_e])
            tau_tx_c = (task.data_size_bits / max(1e3, r_c)) + (task.data_size_bits / cloud.backhaul_bandwidth)
            tau_c_cloud = task.total_cycles / cloud.f_total
            latencies[M + 1] = (
                tau_tx_c
                + tau_c_cloud
                + cloud.propagation_delay
                + cloud.queue_delay_current
                + self.net_cfg.d_dl_constant
            )

            targets[i] = int(np.argmin(latencies))

        fpa_helper = WEKBFPAOptimizer(
            self.opt_cfg, self.cost_cfg, self.net_cfg, use_kb=False, use_channel_adaptive=False
        )
        chrom = np.concatenate([targets.astype(np.float64), np.ones(N), np.ones(N)])
        best_tgt, best_cpu, best_bw = fpa_helper.decode_chromosome(
            chrom, tasks, users, edge_servers, cloud, distances=distances
        )
        res = self.cost_engine.evaluate_allocation(
            tasks, users, edge_servers, cloud, best_tgt, best_cpu, best_bw, sinr_matrix, distances
        )
        duration = time.perf_counter() - start_time

        return OptimizationOutput(
            best_cost=res.total_cost,
            best_targets=best_tgt,
            best_cpu=best_cpu,
            best_bw=best_bw,
            convergence_curve=[res.total_cost],
            wall_clock_time_sec=duration,
            iterations_to_95_percent=1,
            evaluation_result=res,
            algorithm_name="Greedy-Latency",
        )

    def run_greedy_energy(
        self,
        tasks: List[Task],
        users: List[UserDevice],
        edge_servers: List[EdgeServer],
        cloud: CloudServer,
        sinr_matrix: np.ndarray,
        distances: np.ndarray,
    ) -> OptimizationOutput:
        start_time = time.perf_counter()
        N = len(tasks)
        M = len(edge_servers)
        num_targets = M + 2

        targets = np.zeros(N, dtype=np.int32)
        nominal_bw = 1.0 / max(1, N // max(1, M))

        for i in range(N):
            task = tasks[i]
            user = users[i]
            energies = np.zeros(num_targets)


            energies[0] = self.net_cfg.kappa_local * (user.f_local**2) * task.total_cycles


            for m in range(1, M + 1):
                m_idx = m - 1
                edge = edge_servers[m_idx]
                r = nominal_bw * edge.bandwidth_total * np.log2(1.0 + sinr_matrix[i, m_idx])
                tau_tx = task.data_size_bits / max(1e3, r)
                e_comm = user.p_tx * tau_tx
                f_edge_nom = edge.f_total / max(1, N)
                e_comp = self.net_cfg.kappa_edge * (f_edge_nom**2) * task.total_cycles
                energies[m] = e_comm + e_comp


            best_e = int(np.argmin(distances[i]))
            r_c = nominal_bw * edge_servers[best_e].bandwidth_total * np.log2(1.0 + sinr_matrix[i, best_e])
            tau_tx_c = task.data_size_bits / max(1e3, r_c)
            e_comm_c = user.p_tx * tau_tx_c
            f_cloud_nom = cloud.f_total / max(1, N)
            e_comp_c = self.net_cfg.kappa_cloud * (f_cloud_nom**2) * task.total_cycles
            energies[M + 1] = e_comm_c + e_comp_c

            targets[i] = int(np.argmin(energies))

        fpa_helper = WEKBFPAOptimizer(
            self.opt_cfg, self.cost_cfg, self.net_cfg, use_kb=False, use_channel_adaptive=False
        )
        chrom = np.concatenate([targets.astype(np.float64), np.ones(N), np.ones(N)])
        best_tgt, best_cpu, best_bw = fpa_helper.decode_chromosome(
            chrom, tasks, users, edge_servers, cloud, distances=distances
        )
        res = self.cost_engine.evaluate_allocation(
            tasks, users, edge_servers, cloud, best_tgt, best_cpu, best_bw, sinr_matrix, distances
        )
        duration = time.perf_counter() - start_time

        return OptimizationOutput(
            best_cost=res.total_cost,
            best_targets=best_tgt,
            best_cpu=best_cpu,
            best_bw=best_bw,
            convergence_curve=[res.total_cost],
            wall_clock_time_sec=duration,
            iterations_to_95_percent=1,
            evaluation_result=res,
            algorithm_name="Greedy-Energy",
        )

    def run_pso(
        self,
        tasks: List[Task],
        users: List[UserDevice],
        edge_servers: List[EdgeServer],
        cloud: CloudServer,
        sinr_matrix: np.ndarray,
        distances: np.ndarray,
    ) -> OptimizationOutput:
        start_time = time.perf_counter()
        N = len(tasks)
        M = len(edge_servers)
        num_targets = M + 2
        pop_size = self.opt_cfg.population_size
        max_iters = self.opt_cfg.max_iterations
        chrom_len = 3 * N


        w = self.opt_cfg.pso_w
        c1 = self.opt_cfg.pso_c1
        c2 = self.opt_cfg.pso_c2
        v_max = 0.2 * np.concatenate([
            np.full(N, float(num_targets)),
            np.full(2 * N, 1.0),
        ])

        fpa_helper = WEKBFPAOptimizer(
            self.opt_cfg, self.cost_cfg, self.net_cfg, use_kb=False, use_channel_adaptive=False
        )


        positions = np.zeros((pop_size, chrom_len), dtype=np.float64)
        positions[:, 0:N] = self.rng.uniform(0.0, num_targets - 1, size=(pop_size, N))
        positions[:, N : 3 * N] = self.rng.uniform(0.1, 1.0, size=(pop_size, 2 * N))
        velocities = self.rng.uniform(-0.1, 0.1, size=(pop_size, chrom_len))

        p_best = positions.copy()
        p_best_fitness = np.full(pop_size, float("inf"))
        p_best_eval = [None] * pop_size

        for p in range(pop_size):
            tgt, cpu, bw = fpa_helper.decode_chromosome(
                positions[p], tasks, users, edge_servers, cloud, distances=distances
            )
            eval_res = self.cost_engine.evaluate_allocation(
                tasks, users, edge_servers, cloud, tgt, cpu, bw, sinr_matrix, distances
            )
            p_best_fitness[p] = eval_res.total_cost
            p_best_eval[p] = eval_res

        g_best_idx = int(np.argmin(p_best_fitness))
        g_best_pos = p_best[g_best_idx].copy()
        g_best_fitness = p_best_fitness[g_best_idx]
        g_best_eval = p_best_eval[g_best_idx]

        convergence_curve: List[float] = [g_best_fitness]

        for iteration in range(max_iters):
            r1 = self.rng.uniform(0.0, 1.0, size=(pop_size, chrom_len))
            r2 = self.rng.uniform(0.0, 1.0, size=(pop_size, chrom_len))

            velocities = (
                w * velocities
                + c1 * r1 * (p_best - positions)
                + c2 * r2 * (g_best_pos - positions)
            )
            velocities = np.clip(velocities, -v_max, v_max)
            positions = positions + velocities


            positions[:, 0:N] = np.clip(positions[:, 0:N], 0.0, num_targets - 1)
            positions[:, N : 3 * N] = np.clip(positions[:, N : 3 * N], 0.01, 1.0)

            for p in range(pop_size):
                tgt, cpu, bw = fpa_helper.decode_chromosome(
                    positions[p], tasks, users, edge_servers, cloud, distances=distances
                )
                eval_res = self.cost_engine.evaluate_allocation(
                    tasks, users, edge_servers, cloud, tgt, cpu, bw, sinr_matrix, distances
                )
                cost = eval_res.total_cost

                if cost < p_best_fitness[p]:
                    p_best_fitness[p] = cost
                    p_best[p] = positions[p].copy()
                    p_best_eval[p] = eval_res

                    if cost < g_best_fitness:
                        g_best_fitness = cost
                        g_best_pos = positions[p].copy()
                        g_best_eval = eval_res

            convergence_curve.append(g_best_fitness)

        duration = time.perf_counter() - start_time


        cost_delta = convergence_curve[0] - g_best_fitness
        target_95 = convergence_curve[0] - 0.95 * cost_delta
        iter_95 = max_iters
        for it, val in enumerate(convergence_curve):
            if val <= target_95:
                iter_95 = it
                break

        best_tgt, best_cpu, best_bw = fpa_helper.decode_chromosome(
            g_best_pos, tasks, users, edge_servers, cloud, distances=distances
        )

        return OptimizationOutput(
            best_cost=g_best_fitness,
            best_targets=best_tgt,
            best_cpu=best_cpu,
            best_bw=best_bw,
            convergence_curve=convergence_curve,
            wall_clock_time_sec=duration,
            iterations_to_95_percent=iter_95,
            evaluation_result=g_best_eval,
            algorithm_name="PSO",
        )

    def run_ga(
        self,
        tasks: List[Task],
        users: List[UserDevice],
        edge_servers: List[EdgeServer],
        cloud: CloudServer,
        sinr_matrix: np.ndarray,
        distances: np.ndarray,
    ) -> OptimizationOutput:
        start_time = time.perf_counter()
        N = len(tasks)
        M = len(edge_servers)
        num_targets = M + 2
        pop_size = self.opt_cfg.population_size
        max_iters = self.opt_cfg.max_iterations
        chrom_len = 3 * N

        fpa_helper = WEKBFPAOptimizer(
            self.opt_cfg, self.cost_cfg, self.net_cfg, use_kb=False, use_channel_adaptive=False
        )

        population = np.zeros((pop_size, chrom_len), dtype=np.float64)
        population[:, 0:N] = self.rng.uniform(0.0, num_targets - 1, size=(pop_size, N))
        population[:, N : 3 * N] = self.rng.uniform(0.1, 1.0, size=(pop_size, 2 * N))

        fitness = np.zeros(pop_size, dtype=np.float64)
        evals = [None] * pop_size

        for p in range(pop_size):
            tgt, cpu, bw = fpa_helper.decode_chromosome(
                population[p], tasks, users, edge_servers, cloud, distances=distances
            )
            res = self.cost_engine.evaluate_allocation(
                tasks, users, edge_servers, cloud, tgt, cpu, bw, sinr_matrix, distances
            )
            fitness[p] = res.total_cost
            evals[p] = res

        best_idx = int(np.argmin(fitness))
        g_best_chrom = population[best_idx].copy()
        g_best_fitness = fitness[best_idx]
        g_best_eval = evals[best_idx]
        convergence_curve: List[float] = [g_best_fitness]

        for iteration in range(max_iters):
            new_population = []

            new_population.append(g_best_chrom.copy())

            while len(new_population) < pop_size:

                t1 = self.rng.choice(pop_size, size=self.opt_cfg.ga_tournament_size, replace=False)
                parent1 = population[t1[np.argmin(fitness[t1])]]

                t2 = self.rng.choice(pop_size, size=self.opt_cfg.ga_tournament_size, replace=False)
                parent2 = population[t2[np.argmin(fitness[t2])]]


                child1, child2 = parent1.copy(), parent2.copy()
                if self.rng.uniform(0.0, 1.0) < self.opt_cfg.ga_crossover_prob:
                    pts = np.sort(self.rng.choice(chrom_len, size=2, replace=False))
                    child1[pts[0] : pts[1]] = parent2[pts[0] : pts[1]]
                    child2[pts[0] : pts[1]] = parent1[pts[0] : pts[1]]


                for child in (child1, child2):
                    if self.rng.uniform(0.0, 1.0) < self.opt_cfg.ga_mutation_prob:
                        mut_idx = self.rng.choice(chrom_len)
                        if mut_idx < N:
                            child[mut_idx] = self.rng.uniform(0.0, num_targets - 1)
                        else:
                            child[mut_idx] = self.rng.uniform(0.01, 1.0)


                    child[0:N] = np.clip(child[0:N], 0.0, num_targets - 1)
                    child[N : 3 * N] = np.clip(child[N : 3 * N], 0.01, 1.0)

                    if len(new_population) < pop_size:
                        new_population.append(child)

            population = np.array(new_population)
            for p in range(pop_size):
                tgt, cpu, bw = fpa_helper.decode_chromosome(
                    population[p], tasks, users, edge_servers, cloud, distances=distances
                )
                res = self.cost_engine.evaluate_allocation(
                    tasks, users, edge_servers, cloud, tgt, cpu, bw, sinr_matrix, distances
                )
                fitness[p] = res.total_cost
                evals[p] = res

                if res.total_cost < g_best_fitness:
                    g_best_fitness = res.total_cost
                    g_best_chrom = population[p].copy()
                    g_best_eval = res

            convergence_curve.append(g_best_fitness)

        duration = time.perf_counter() - start_time

        cost_delta = convergence_curve[0] - g_best_fitness
        target_95 = convergence_curve[0] - 0.95 * cost_delta
        iter_95 = max_iters
        for it, val in enumerate(convergence_curve):
            if val <= target_95:
                iter_95 = it
                break

        best_tgt, best_cpu, best_bw = fpa_helper.decode_chromosome(
            g_best_chrom, tasks, users, edge_servers, cloud, distances=distances
        )

        return OptimizationOutput(
            best_cost=g_best_fitness,
            best_targets=best_tgt,
            best_cpu=best_cpu,
            best_bw=best_bw,
            convergence_curve=convergence_curve,
            wall_clock_time_sec=duration,
            iterations_to_95_percent=iter_95,
            evaluation_result=g_best_eval,
            algorithm_name="GA",
        )

    def run_fpa_baseline(
        self,
        tasks: List[Task],
        users: List[UserDevice],
        edge_servers: List[EdgeServer],
        cloud: CloudServer,
        sinr_matrix: np.ndarray,
        distances: np.ndarray,
    ) -> OptimizationOutput:
        fpa = WEKBFPAOptimizer(
            self.opt_cfg,
            self.cost_cfg,
            self.net_cfg,
            use_kb=False,
            use_channel_adaptive=False,
            seed=int(self.rng.integers(1, 100000)),
        )
        return fpa.optimize(tasks, users, edge_servers, cloud, sinr_matrix, distances)
