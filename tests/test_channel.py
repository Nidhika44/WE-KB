"""
Tests for Wireless Channel Simulator & PHY Layer.
"""

import unittest
import numpy as np
from we_kb_fpa.config import ChannelConfig, NetworkConfig
from we_kb_fpa.channel_model import WirelessChannelSimulator


class TestChannelModel(unittest.TestCase):

    def setUp(self):
        self.net_cfg = NetworkConfig(num_users=8, num_edge_servers=2)
        self.ch_cfg = ChannelConfig()
        self.sim = WirelessChannelSimulator(self.ch_cfg, self.net_cfg, seed=42)

    def test_path_loss(self):
        distances = np.array([[10.0, 100.0], [50.0, 200.0]])
        pl_db = self.sim.compute_path_loss_db(distances)
        self.assertEqual(pl_db.shape, distances.shape)


        self.assertTrue(np.all(pl_db > 0))

    def test_small_scale_fading(self):
        shape = (100, 4)
        fading = self.sim.sample_small_scale_fading(shape)
        self.assertEqual(fading.shape, shape)
        self.assertTrue(np.all(fading >= 0))

        self.assertAlmostEqual(float(np.mean(fading)), 1.0, delta=0.15)

    def test_sinr_and_rates(self):
        distances = np.full((8, 2), 50.0)
        comp_gains, _ = self.sim.update_channel_states(distances)
        p_tx = np.full(8, 0.2)
        sinr = self.sim.compute_sinr_matrix(comp_gains, p_tx)

        self.assertEqual(sinr.shape, (8, 2))
        self.assertTrue(np.all(sinr > 0))


        bw_alloc = np.full((8, 2), 0.125)
        rates = self.sim.compute_data_rates(sinr, bw_alloc)
        self.assertEqual(rates.shape, (8, 2))
        self.assertTrue(np.all(rates > 0))

    def test_channel_variance(self):
        distances = np.full((10, 3), 40.0)
        self.sim.update_channel_states(distances)
        var = self.sim.get_channel_variance()
        self.assertGreater(var, 0.0)


if __name__ == "__main__":
    unittest.main()
