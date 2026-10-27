"""
Tests for Cost Engine & Evaluator.
"""

import unittest
import numpy as np
from we_kb_fpa.config import CostConfig, NetworkConfig
from we_kb_fpa.environment_model import UserDevice, EdgeServer, CloudServer
from we_kb_fpa.task_model import Task
from we_kb_fpa.cost_engine import CostEngine


class TestCostEngine(unittest.TestCase):

    def setUp(self):
        self.net_cfg = NetworkConfig(d_dl_constant=0.002)
        self.cost_cfg = CostConfig(alpha=0.5)
        self.engine = CostEngine(self.cost_cfg, self.net_cfg)

        self.user = UserDevice(device_id=0, x=100.0, y=100.0, p_tx=0.2, f_local=1.0e9)
        self.task = Task(
            task_id=1,
            user_id=0,
            data_size_bits=2.0e6,
            cycles_per_bit=1000.0,
            deadline_sec=0.500,
        )
        self.edge = EdgeServer(
            server_id=1, x=100.0, y=100.0, f_total=40.0e9, bandwidth_total=20.0e6
        )
        self.cloud = CloudServer()

    def test_local_metrics(self):
        tau_loc, e_loc = self.engine.compute_local_metrics(self.task, self.user)

        self.assertAlmostEqual(tau_loc, 2.0)

        self.assertAlmostEqual(e_loc, 0.2)

    def test_edge_metrics_including_downlink(self):
        f_alloc = 10.0e9
        rate = 50.0e6
        dist = 50.0

        (
            tau_tx,
            tau_q,
            tau_c,
            tau_p,
            tau_dl,
            e_comm,
            e_comp,
        ) = self.engine.compute_edge_metrics(
            self.task, self.user, self.edge, f_alloc, rate, dist
        )


        self.assertAlmostEqual(tau_tx, 0.04)

        self.assertAlmostEqual(tau_c, 0.2)

        self.assertAlmostEqual(tau_dl, 0.002)

    def test_full_allocation_evaluation(self):
        tasks = [self.task]
        users = [self.user]
        edges = [self.edge]
        cloud = self.cloud

        targets = np.array([1])
        cpus = np.array([10.0e9])
        bws = np.array([0.5])
        sinr = np.array([[10.0]])
        distances = np.array([[50.0]])

        res = self.engine.evaluate_allocation(
            tasks, users, edges, cloud, targets, cpus, bws, sinr, distances
        )
        self.assertGreater(res.total_cost, 0.0)
        self.assertEqual(len(res.task_metrics), 1)
        self.assertEqual(res.task_metrics[0].target_node, 1)


if __name__ == "__main__":
    unittest.main()
