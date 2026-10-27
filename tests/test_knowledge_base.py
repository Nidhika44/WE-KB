"""
Tests for Wireless-Edge Knowledge Base (Storage, KNN retrieval, Pruning Mask).
"""

import unittest
import numpy as np
from we_kb_fpa.config import KnowledgeBaseConfig, NetworkConfig
from we_kb_fpa.environment_model import UserDevice, EdgeServer, CloudServer
from we_kb_fpa.task_model import Task
from we_kb_fpa.knowledge_base import WirelessEdgeKnowledgeBase


class TestKnowledgeBase(unittest.TestCase):

    def setUp(self):
        self.net_cfg = NetworkConfig(num_users=3, num_edge_servers=2)
        self.kb_cfg = KnowledgeBaseConfig(k_neighbors=2)
        self.kb = WirelessEdgeKnowledgeBase(self.kb_cfg, self.net_cfg)

        self.users = [
            UserDevice(device_id=i, x=50.0, y=50.0, p_tx=0.2, f_local=1.0e9)
            for i in range(3)
        ]
        self.tasks = [
            Task(
                task_id=i + 1,
                user_id=i,
                data_size_bits=2.0e6,
                cycles_per_bit=1000.0,
                deadline_sec=0.500,
            )
            for i in range(3)
        ]
        self.edges = [
            EdgeServer(server_id=1, x=100.0, y=100.0, f_total=40.0e9, bandwidth_total=20.0e6),
            EdgeServer(server_id=2, x=200.0, y=200.0, f_total=40.0e9, bandwidth_total=20.0e6),
        ]
        self.cloud = CloudServer()
        self.sinr = np.full((3, 2), 12.0)
        self.distances = np.full((3, 2), 60.0)

    def test_experience_addition_and_query(self):
        targets = np.array([1, 2, 0])
        cpus = np.array([10e9, 10e9, 1e9])
        bws = np.array([0.5, 0.5, 0.0])

        self.kb.add_experience(
            self.sinr, self.tasks, self.distances, targets, cpus, bws, fitness_cost=1.23
        )
        self.assertEqual(len(self.kb.records), 1)

        nearest = self.kb.query_nearest_solutions(self.sinr, self.tasks, self.distances, k=1)
        self.assertEqual(len(nearest), 1)
        rec, sim = nearest[0]
        self.assertEqual(rec.fitness_cost, 1.23)
        self.assertGreater(sim, 0.0)

    def test_pruning_mask_shape_and_content(self):
        mask = self.kb.generate_pruning_mask(
            self.tasks, self.users, self.edges, self.cloud, self.sinr, self.distances
        )
        self.assertEqual(mask.shape, (3, 4))

        self.assertTrue(np.all((mask == 0) | (mask == 1)))

        for i in range(3):
            self.assertGreater(np.sum(mask[i]), 0)


if __name__ == "__main__":
    unittest.main()
