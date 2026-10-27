"""Latency, energy, resource constraints, and penalized system cost."""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np
from we_kb_fpa.config import CostConfig, NetworkConfig
from we_kb_fpa.environment_model import UserDevice, EdgeServer, CloudServer
from we_kb_fpa.task_model import Task


@dataclass
class TaskMetric:
    task_id: int
    user_id: int
    target_node: int
    latency_total: float
    latency_tx: float
    latency_queue: float
    latency_comp: float
    latency_prop: float
    latency_dl: float
    energy_total: float
    energy_comm: float
    energy_comp: float
    is_deadline_violated: bool
    deadline_sec: float
    cost_j: float


@dataclass
class SystemEvaluationResult:
    total_cost: float
    average_latency: float
    average_energy: float
    deadline_violation_rate: float
    total_penalties: float
    jains_fairness_index: float
    task_metrics: List[TaskMetric]
    edge_cpu_allocations: Dict[int, float]
    edge_bw_allocations: Dict[int, float]


class CostEngine:

    def __init__(
        self,
        cost_config: Optional[CostConfig] = None,
        net_config: Optional[NetworkConfig] = None,
    ):
        self.cost_cfg = cost_config or CostConfig()
        self.net_cfg = net_config or NetworkConfig()

    def compute_local_metrics(
        self, task: Task, user: UserDevice
    ) -> Tuple[float, float]:
        """Calculates latency and energy for local IoT device execution.

        tau_loc = (D_k * C_k) / f_loc
        E_loc = kappa * (f_loc)^2 * (D_k * C_k)
        """
        f_loc = user.f_local
        total_cycles = task.total_cycles

        tau_comp = total_cycles / f_loc
        energy_comp = self.net_cfg.kappa_local * (f_loc**2) * total_cycles

        return tau_comp, energy_comp

    def compute_edge_metrics(
        self,
        task: Task,
        user: UserDevice,
        edge: EdgeServer,
        allocated_f_hz: float,
        achievable_rate_bps: float,
        distance_meters: float,
    ) -> Tuple[float, float, float, float, float, float, float]:
        """Calculates latency & energy breakdown for edge offloading.

        Returns:
            (tau_tx, tau_queue, tau_comp, tau_prop, tau_dl, energy_comm, energy_comp)
        """

        tau_tx = task.data_size_bits / max(1e3, achievable_rate_bps)
        energy_comm = user.p_tx * tau_tx


        tau_queue = edge.queue_delay_current


        f_alloc = max(1e6, allocated_f_hz)
        tau_comp = task.total_cycles / f_alloc
        energy_comp = self.net_cfg.kappa_edge * (f_alloc**2) * task.total_cycles


        tau_prop = distance_meters / 3.0e8


        tau_dl = self.net_cfg.d_dl_constant

        return tau_tx, tau_queue, tau_comp, tau_prop, tau_dl, energy_comm, energy_comp

    def compute_cloud_metrics(
        self,
        task: Task,
        user: UserDevice,
        cloud: CloudServer,
        allocated_f_hz: float,
        uplink_rate_to_bs_bps: float,
    ) -> Tuple[float, float, float, float, float, float, float]:
        """Calculates latency & energy breakdown for cloud offloading.

        Includes cellular uplink + wired/fiber backhaul transfer delay.
        """

        tau_tx_wireless = task.data_size_bits / max(1e3, uplink_rate_to_bs_bps)

        tau_tx_backhaul = task.data_size_bits / cloud.backhaul_bandwidth
        tau_tx = tau_tx_wireless + tau_tx_backhaul
        energy_comm = user.p_tx * tau_tx_wireless

        tau_queue = cloud.queue_delay_current

        f_alloc = max(1e6, allocated_f_hz)
        tau_comp = task.total_cycles / f_alloc
        energy_comp = self.net_cfg.kappa_cloud * (f_alloc**2) * task.total_cycles

        tau_prop = cloud.propagation_delay
        tau_dl = self.net_cfg.d_dl_constant

        return tau_tx, tau_queue, tau_comp, tau_prop, tau_dl, energy_comm, energy_comp

    def evaluate_allocation(
        self,
        tasks: List[Task],
        users: List[UserDevice],
        edge_servers: List[EdgeServer],
        cloud: CloudServer,
        offloading_targets: np.ndarray,
        cpu_allocations: np.ndarray,
        bandwidth_allocations: np.ndarray,
        sinr_matrix: np.ndarray,
        distances: np.ndarray,
    ) -> SystemEvaluationResult:
        """Evaluates complete system objective J and constraint penalties.

        Objective:
        J = sum_k [ alpha * (tau_k / tau_max_k) + (1 - alpha) * (E_k / E_max_k) ] + Penalties
        """
        N = len(tasks)
        M = len(edge_servers)
        cloud_target_id = M + 1

        task_metrics: List[TaskMetric] = []
        total_penalty = 0.0
        alpha = self.cost_cfg.alpha


        edge_cpu_used = {m_id: 0.0 for m_id in range(1, M + 1)}
        edge_bw_used = {m_id: 0.0 for m_id in range(1, M + 1)}
        cloud_cpu_used = 0.0

        for i, (task, user) in enumerate(zip(tasks, users)):
            target = int(offloading_targets[i])


            tau_loc_nominal, e_loc_nominal = self.compute_local_metrics(task, user)
            e_norm_base = max(e_loc_nominal, 1e-6)

            if target == 0:

                tau_comp, e_comp = tau_loc_nominal, e_loc_nominal
                tau_tx = tau_queue = tau_prop = tau_dl = 0.0
                e_comm = 0.0
                tau_k = tau_comp
                e_k = e_comp

            elif 1 <= target <= M:

                edge_idx = target - 1
                edge = edge_servers[edge_idx]
                f_hz = cpu_allocations[i]
                b_frac = bandwidth_allocations[i]

                edge_cpu_used[target] += f_hz
                edge_bw_used[target] += b_frac


                sinr_val = sinr_matrix[i, edge_idx]
                achievable_rate = b_frac * edge.bandwidth_total * np.log2(1.0 + sinr_val)
                achievable_rate = max(1e3, achievable_rate)

                (
                    tau_tx,
                    tau_queue,
                    tau_comp,
                    tau_prop,
                    tau_dl,
                    e_comm,
                    e_comp,
                ) = self.compute_edge_metrics(
                    task, user, edge, f_hz, achievable_rate, distances[i, edge_idx]
                )
                tau_k = tau_tx + tau_queue + tau_comp + tau_prop + tau_dl
                e_k = e_comm + e_comp

            else:

                f_hz = cpu_allocations[i]
                cloud_cpu_used += f_hz

                # Uplink uses best/nearest edge station channel for cellular connection
                best_edge_idx = int(np.argmin(distances[i]))
                b_frac = bandwidth_allocations[i]
                edge_bw_used[best_edge_idx + 1] += b_frac

                sinr_val = sinr_matrix[i, best_edge_idx]
                achievable_rate = (
                    b_frac * edge_servers[best_edge_idx].bandwidth_total * np.log2(1.0 + sinr_val)
                )
                achievable_rate = max(1e3, achievable_rate)

                (
                    tau_tx,
                    tau_queue,
                    tau_comp,
                    tau_prop,
                    tau_dl,
                    e_comm,
                    e_comp,
                ) = self.compute_cloud_metrics(task, user, cloud, f_hz, achievable_rate)
                tau_k = tau_tx + tau_queue + tau_comp + tau_prop + tau_dl
                e_k = e_comm + e_comp


            deadline_violated = tau_k > task.deadline_sec
            if deadline_violated:

                delay_excess_ratio = (tau_k - task.deadline_sec) / task.deadline_sec
                penalty_qos = self.cost_cfg.penalty_deadline_multiplier * (1.0 + delay_excess_ratio) ** 2
                total_penalty += penalty_qos


            norm_tau = tau_k / task.deadline_sec
            norm_e = e_k / e_norm_base
            cost_j_k = alpha * norm_tau + (1.0 - alpha) * norm_e

            task_metrics.append(
                TaskMetric(
                    task_id=task.task_id,
                    user_id=user.device_id,
                    target_node=target,
                    latency_total=tau_k,
                    latency_tx=tau_tx,
                    latency_queue=tau_queue,
                    latency_comp=tau_comp,
                    latency_prop=tau_prop,
                    latency_dl=tau_dl,
                    energy_total=e_k,
                    energy_comm=e_comm,
                    energy_comp=e_comp,
                    is_deadline_violated=deadline_violated,
                    deadline_sec=task.deadline_sec,
                    cost_j=cost_j_k,
                )
            )


        for m_id, edge in enumerate(edge_servers, start=1):
            if edge_cpu_used[m_id] > edge.f_total:
                excess_cpu_ratio = (edge_cpu_used[m_id] - edge.f_total) / edge.f_total
                total_penalty += self.cost_cfg.penalty_cpu_multiplier * (1.0 + excess_cpu_ratio) ** 2

            if edge_bw_used[m_id] > 1.0 + 1e-5:
                excess_bw = edge_bw_used[m_id] - 1.0
                total_penalty += self.cost_cfg.penalty_bandwidth_multiplier * (1.0 + excess_bw) ** 2

        if cloud_cpu_used > cloud.f_total:
            excess_cloud = (cloud_cpu_used - cloud.f_total) / cloud.f_total
            total_penalty += self.cost_cfg.penalty_cpu_multiplier * (1.0 + excess_cloud) ** 2


        avg_latency = float(np.mean([m.latency_total for m in task_metrics]))
        avg_energy = float(np.mean([m.energy_total for m in task_metrics]))
        num_violations = sum(1 for m in task_metrics if m.is_deadline_violated)
        violation_rate = (num_violations / N) * 100.0 if N > 0 else 0.0


        cpu_allocs = [edge_cpu_used[m_id] for m_id in range(1, M + 1)]
        sum_c = np.sum(cpu_allocs)
        sum_c_sq = np.sum(np.array(cpu_allocs) ** 2)
        jains_index = (sum_c**2) / (M * sum_c_sq) if sum_c_sq > 0 else 1.0

        total_system_cost = float(np.sum([m.cost_j for m in task_metrics]) + total_penalty)

        return SystemEvaluationResult(
            total_cost=total_system_cost,
            average_latency=avg_latency,
            average_energy=avg_energy,
            deadline_violation_rate=violation_rate,
            total_penalties=total_penalty,
            jains_fairness_index=float(jains_index),
            task_metrics=task_metrics,
            edge_cpu_allocations=edge_cpu_used,
            edge_bw_allocations=edge_bw_used,
        )
