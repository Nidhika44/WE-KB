"""
Tests for Optimization Algorithms (WE-KB-FPA, Baselines, Ablation Variants).
"""

import unittest
import numpy as np
from we_kb_fpa.config import OptimizerConfig, CostConfig, NetworkConfig, KnowledgeBaseConfig
from we_kb_fpa.environment_model import UserDevice, EdgeServer, CloudServer
from we_kb_fpa.task_model import Task
from we_kb_fpa.knowledge_base import WirelessEdgeKnowledgeBase
from we_kb_fpa.we_kb_fpa import WEKBFPAOptimizer
from we_kb_fpa.baselines import BaselineRunner


class TestAlgorithms(unittest.TestCase):

    def setUp(self):
        self.net_cfg = NetworkConfig(num_users=4, num_edge_servers=2)
        self.cost_cfg = CostConfig()
        self.opt_cfg = OptimizerConfig(population_size=10, max_iterations=5)
        self.kb_cfg = KnowledgeBaseConfig()

        self.users = [
            UserDevice(device_id=i, x=100.0, y=100.0, p_tx=0.2, f_local=1.0e9)
            for i in range(4)
        ]
        self.tasks = [
            Task(
                task_id=i + 1,
                user_id=i,
                data_size_bits=2.0e6,
                cycles_per_bit=1000.0,
                deadline_sec=0.500,
            )
            for i in range(4)
        ]
        self.edges = [
            EdgeServer(server_id=1, x=100.0, y=100.0, f_total=40.0e9, bandwidth_total=20.0e6),
            EdgeServer(server_id=2, x=200.0, y=200.0, f_total=40.0e9, bandwidth_total=20.0e6),
        ]
        self.cloud = CloudServer()
        self.sinr = np.full((4, 2), 10.0)
        self.distances = np.full((4, 2), 50.0)
        self.kb = WirelessEdgeKnowledgeBase(self.kb_cfg, self.net_cfg)

    def test_we_kb_fpa_run(self):
        opt = WEKBFPAOptimizer(
            self.opt_cfg, self.cost_cfg, self.net_cfg, use_kb=True, use_channel_adaptive=True
        )
        res = opt.optimize(
            self.tasks, self.users, self.edges, self.cloud, self.sinr, self.distances, knowledge_base=self.kb
        )
        self.assertIsNotNone(res)
        self.assertEqual(len(res.best_targets), 4)
        self.assertGreater(res.best_cost, 0.0)
        self.assertEqual(len(res.convergence_curve), 6)

    def test_baselines(self):
        runner = BaselineRunner(self.opt_cfg, self.cost_cfg, self.net_cfg, seed=42)

        res_rand = runner.run_random(self.tasks, self.users, self.edges, self.cloud, self.sinr, self.distances)
        self.assertIsNotNone(res_rand)

        res_lat = runner.run_greedy_latency(self.tasks, self.users, self.edges, self.cloud, self.sinr, self.distances)
        self.assertIsNotNone(res_lat)

        res_eng = runner.run_greedy_energy(self.tasks, self.users, self.edges, self.cloud, self.sinr, self.distances)
        self.assertIsNotNone(res_eng)

        res_pso = runner.run_pso(self.tasks, self.users, self.edges, self.cloud, self.sinr, self.distances)
        self.assertIsNotNone(res_pso)

        res_ga = runner.run_ga(self.tasks, self.users, self.edges, self.cloud, self.sinr, self.distances)
        self.assertIsNotNone(res_ga)


if __name__ == "__main__":
    unittest.main()
