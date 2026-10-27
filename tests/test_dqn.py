"""Learning mechanics, reproducibility, and shared shock observations."""
import unittest
from unittest.mock import patch
import numpy as np
from we_kb_fpa.dqn import DQNBaseline, DQNConfig
from we_kb_fpa.config import NetworkConfig, OptimizerConfig
from we_kb_fpa.simulation_executor import SimulationExecutor


class TestDQN(unittest.TestCase):
    def test_replay_learning_and_target_sync(self):
        agent = DQNBaseline(config=DQNConfig(batch_size=4, target_interval=2))
        agent._initialize(3, 2)
        before = [w.copy() for w in agent.weights]
        for _ in range(4):
            agent.memory.append((np.ones(3), 1, -2., np.ones(3), True))
        agent.learn()
        self.assertTrue(any(not np.array_equal(a, b) for a, b in zip(before, agent.weights)))
        for a, b in zip(before, agent.target):
            np.testing.assert_array_equal(a, b)
        agent.learn()
        for a, b in zip(agent.weights, agent.target):
            np.testing.assert_array_equal(a, b)
        self.assertTrue(np.isfinite(agent.losses).all())

    def test_terminal_does_not_bootstrap(self):
        agent = DQNBaseline(config=DQNConfig(batch_size=1))
        agent._initialize(3, 2)
        agent.target[-1][:] = 1e10
        agent.memory.append((np.zeros(3), 0, -2., np.zeros(3), True))
        agent.learn()
        self.assertAlmostEqual(agent.losses[-1], 1.5)

    def test_shock_and_frozen_inference(self):
        cfg = dict(net_config=NetworkConfig(num_users=4, num_edge_servers=2),
                   opt_config=OptimizerConfig(population_size=6, max_iterations=2), seed=19)
        nominal, shocked = SimulationExecutor(**cfg), SimulationExecutor(**cfg)
        captures = []
        for sim, fade in [(nominal, 0.), (shocked, 35.)]:
            original = sim.dqn.run
            def capture(*args, **kwargs):
                captures.append(args[4].copy())
                return original(*args, **kwargs)
            with patch.object(sim.dqn, "run", side_effect=capture):
                result = sim.run_slot(0, ["DQN"], fading_db=fade)
            self.assertTrue(np.isfinite(result.algorithm_outputs["DQN"].best_cost))
            self.assertEqual(sim.dqn.updates, 0)
            self.assertEqual(len(sim.dqn.memory), 0)
            weights = [w.copy() for w in sim.dqn.weights]
            sim.run_slot(1, ["DQN"])
            for before, after in zip(weights, sim.dqn.weights):
                np.testing.assert_array_equal(before, after)
        np.testing.assert_allclose(captures[1], captures[0] * 10 ** (-3.5))

    def test_seed_reproducibility(self):
        cfg = dict(net_config=NetworkConfig(num_users=4, num_edge_servers=2), seed=6)
        a, b = SimulationExecutor(**cfg), SimulationExecutor(**cfg)
        for slot in range(10):
            x = a.run_slot(slot, ["DQN"], dqn_training=True).algorithm_outputs["DQN"]
            y = b.run_slot(slot, ["DQN"], dqn_training=True).algorithm_outputs["DQN"]
            np.testing.assert_array_equal(x.best_targets, y.best_targets)
            self.assertEqual(x.best_cost, y.best_cost)
        self.assertGreater(a.dqn.updates, 0)

    def test_partial_fade_and_validation(self):
        cfg = dict(net_config=NetworkConfig(num_users=4, num_edge_servers=2), seed=7)
        a, b = SimulationExecutor(**cfg), SimulationExecutor(**cfg)
        seen = []
        for sim, fade in [(a, 0.), (b, 20.)]:
            original = sim.dqn.run
            def capture(*args, **kwargs):
                seen.append(args[4].copy())
                return original(*args, **kwargs)
            with patch.object(sim.dqn, "run", side_effect=capture):
                sim.run_slot(0, ["DQN"], fading_db=fade, faded_edges=[1])
        np.testing.assert_allclose(seen[0][:, 0], seen[1][:, 0])
        np.testing.assert_allclose(seen[0][:, 1] * .01, seen[1][:, 1])
        with self.assertRaises(ValueError):
            a.run_slot(1, ["DQN"], faded_edges=[2])


if __name__ == "__main__":
    unittest.main()
