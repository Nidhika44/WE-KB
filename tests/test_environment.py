"""
Tests for Environment Model & Mobility Engines.
"""

import unittest
import numpy as np
from we_kb_fpa.config import NetworkConfig
from we_kb_fpa.environment_model import NetworkEnvironment, UserDevice, EdgeServer, CloudServer


class TestEnvironmentModel(unittest.TestCase):

    def setUp(self):
        self.cfg = NetworkConfig(num_users=10, num_edge_servers=3, area_width=500.0, area_height=500.0)
        self.env = NetworkEnvironment(self.cfg, seed=123)

    def test_node_counts(self):
        self.assertEqual(len(self.env.users), 10)
        self.assertEqual(len(self.env.edge_servers), 3)
        self.assertIsNotNone(self.env.cloud)

    def test_user_attributes(self):
        for user in self.env.users:
            self.assertTrue(0.0 <= user.x <= 500.0)
            self.assertTrue(0.0 <= user.y <= 500.0)
            self.assertTrue(user.p_tx > 0)
            self.assertTrue(user.f_local > 0)

    def test_mobility_step(self):
        init_positions = [(u.x, u.y) for u in self.env.users]
        self.env.step_mobility(dt=0.5)
        new_positions = [(u.x, u.y) for u in self.env.users]


        for (x0, y0), (x1, y1) in zip(init_positions, new_positions):
            self.assertTrue(0.0 <= x1 <= 500.0)
            self.assertTrue(0.0 <= y1 <= 500.0)

    def test_distance_matrix(self):
        dist_mat = self.env.get_distance_matrix()
        self.assertEqual(dist_mat.shape, (10, 3))
        self.assertTrue(np.all(dist_mat >= 1.0))

    def test_queue_drain(self):
        edge = self.env.edge_servers[0]
        edge.cpu_queue_load = 1e9
        self.env.update_server_queues(dt=0.1)
        expected_drain = edge.f_total * 0.1
        self.assertAlmostEqual(edge.cpu_queue_load, max(0.0, 1e9 - expected_drain))


if __name__ == "__main__":
    unittest.main()
