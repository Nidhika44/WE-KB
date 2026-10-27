"""Historical allocation retrieval and physical feasibility pruning."""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np
from sklearn.neighbors import NearestNeighbors

from we_kb_fpa.config import KnowledgeBaseConfig, NetworkConfig
from we_kb_fpa.environment_model import UserDevice, EdgeServer, CloudServer
from we_kb_fpa.task_model import Task


@dataclass
class KnowledgeRecord:
    state_vector: np.ndarray
    channel_features: np.ndarray
    workload_features: np.ndarray
    solution_targets: np.ndarray
    solution_cpu: np.ndarray
    solution_bw: np.ndarray
    fitness_cost: float


class WirelessEdgeKnowledgeBase:

    def __init__(
        self,
        kb_config: Optional[KnowledgeBaseConfig] = None,
        net_config: Optional[NetworkConfig] = None,
    ):
        self.kb_cfg = kb_config or KnowledgeBaseConfig()
        self.net_cfg = net_config or NetworkConfig()
        self.records: List[KnowledgeRecord] = []
        self._knn_index: Optional[NearestNeighbors] = None
        self._is_index_fitted: bool = False

    def extract_state_feature_vector(
        self,
        sinr_matrix: np.ndarray,
        tasks: List[Task],
        distances: np.ndarray,
    ) -> np.ndarray:
        N = len(tasks)
        M = sinr_matrix.shape[1]


        mean_sinr_db = 10.0 * np.log10(np.maximum(1e-4, np.mean(sinr_matrix, axis=0)))
        std_sinr = np.std(sinr_matrix, axis=0)


        d_mb_array = np.array([t.data_size_mb for t in tasks])
        c_array = np.array([t.cycles_per_bit for t in tasks])
        tau_array = np.array([t.deadline_sec for t in tasks])


        task_summary = np.array([
            np.mean(d_mb_array) / 10.0,
            np.std(d_mb_array) / 10.0,
            np.mean(c_array) / 1500.0,
            np.mean(tau_array) / 0.5,
            np.min(tau_array) / 0.5,
        ])

        channel_summary = np.concatenate([
            mean_sinr_db / 30.0,
            std_sinr / (np.mean(sinr_matrix) + 1e-4),
            np.mean(distances, axis=0) / self.net_cfg.area_width,
        ])

        feature_vector = np.concatenate([channel_summary, task_summary]).astype(np.float64)
        return feature_vector

    def add_experience(
        self,
        sinr_matrix: np.ndarray,
        tasks: List[Task],
        distances: np.ndarray,
        targets: np.ndarray,
        cpu_allocs: np.ndarray,
        bw_allocs: np.ndarray,
        fitness_cost: float,
    ):
        feat_vec = self.extract_state_feature_vector(sinr_matrix, tasks, distances)


        if len(self.records) >= self.kb_cfg.memory_capacity:
            self.records.pop(0)

        record = KnowledgeRecord(
            state_vector=feat_vec,
            channel_features=np.mean(sinr_matrix, axis=0),
            workload_features=np.array([t.data_size_mb for t in tasks]),
            solution_targets=targets.copy(),
            solution_cpu=cpu_allocs.copy(),
            solution_bw=bw_allocs.copy(),
            fitness_cost=fitness_cost,
        )
        self.records.append(record)
        self._is_index_fitted = False

    def _rebuild_index(self):
        if not self.records:
            return
        X = np.array([rec.state_vector for rec in self.records])
        k = min(self.kb_cfg.k_neighbors, len(self.records))
        self._knn_index = NearestNeighbors(
            n_neighbors=k,
            algorithm="kd_tree",
            metric="euclidean",
        )
        self._knn_index.fit(X)
        self._is_index_fitted = True

    def query_nearest_solutions(
        self,
        sinr_matrix: np.ndarray,
        tasks: List[Task],
        distances: np.ndarray,
        k: Optional[int] = None,
    ) -> List[Tuple[KnowledgeRecord, float]]:
        if not self.records:
            return []

        if not self._is_index_fitted or self._knn_index is None:
            self._rebuild_index()

        req_k = k if k is not None else self.kb_cfg.k_neighbors
        k = max(1, min(req_k, len(self.records)))
        query_feat = self.extract_state_feature_vector(sinr_matrix, tasks, distances).reshape(1, -1)

        distances_k, indices_k = self._knn_index.kneighbors(query_feat, n_neighbors=k)
        results = []
        for dist, idx in zip(distances_k[0], indices_k[0]):
            rec = self.records[idx]
            sim_score = 1.0 / (1.0 + dist)
            results.append((rec, sim_score))

        return results

    def generate_pruning_mask(
        self,
        tasks: List[Task],
        users: List[UserDevice],
        edge_servers: List[EdgeServer],
        cloud: CloudServer,
        sinr_matrix: np.ndarray,
        distances: np.ndarray,
    ) -> np.ndarray:
        N = len(tasks)
        M = len(edge_servers)
        num_targets = M + 2
        mask = np.ones((N, num_targets), dtype=np.int32)

        for i in range(N):
            task = tasks[i]
            user = users[i]
            deadline = task.deadline_sec
            min_delays = np.zeros(num_targets, dtype=np.float64)


            min_local_tau = task.total_cycles / user.f_local
            min_delays[0] = min_local_tau
            if min_local_tau > deadline:

                mask[i, 0] = 0


            for m in range(1, M + 1):
                m_idx = m - 1
                edge = edge_servers[m_idx]
                sinr = sinr_matrix[i, m_idx]


                max_rate_bps = edge.bandwidth_total * np.log2(1.0 + sinr)
                min_tx_tau = task.data_size_bits / max(1e3, max_rate_bps)


                min_comp_tau = task.total_cycles / edge.f_total


                tau_prop = distances[i, m_idx] / 3.0e8
                tau_dl = self.net_cfg.d_dl_constant
                tau_q = edge.queue_delay_current

                min_total_edge_tau = min_tx_tau + min_comp_tau + tau_prop + tau_dl + tau_q
                min_delays[m] = min_total_edge_tau


                if min_total_edge_tau > deadline:
                    mask[i, m] = 0


            cloud_s = M + 1
            best_edge_idx = int(np.argmin(distances[i]))
            best_edge = edge_servers[best_edge_idx]
            sinr_best = sinr_matrix[i, best_edge_idx]

            max_uplink_rate = best_edge.bandwidth_total * np.log2(1.0 + sinr_best)
            min_cloud_tx = (task.data_size_bits / max(1e3, max_uplink_rate)) + (
                task.data_size_bits / cloud.backhaul_bandwidth
            )
            min_cloud_comp = task.total_cycles / cloud.f_total
            min_cloud_total = (
                min_cloud_tx
                + min_cloud_comp
                + cloud.propagation_delay
                + self.net_cfg.d_dl_constant
                + cloud.queue_delay_current
            )
            min_delays[cloud_s] = min_cloud_total

            if min_cloud_total > deadline:
                mask[i, cloud_s] = 0


            if np.sum(mask[i]) == 0:

                best_targets = np.argsort(min_delays)[:min(2, num_targets)]
                mask[i, best_targets] = 1

        return mask
