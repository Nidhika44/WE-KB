import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from experiments.run_topology_shock import run_topology_shock, plot_adaptation, plot_learning
from we_kb_fpa.allocation import uniform_allocation
from we_kb_fpa.config import CostConfig, NetworkConfig, OptimizerConfig
from we_kb_fpa.droo import DROOAdapted, order_preserving_candidates
from we_kb_fpa.dqn import DQNBaseline,DQNConfig
from we_kb_fpa.evaluation import delay_adjusted_metrics, validate, validation_bank
from we_kb_fpa.simulation_executor import SimulationExecutor
from we_kb_fpa.we_kb_fpa import WEKBFPAOptimizer


class TestResearchEvaluation(unittest.TestCase):
    def setUp(self):
        self.sim = SimulationExecutor(net_config=NetworkConfig(num_users=4, num_edge_servers=2),
                                      opt_config=OptimizerConfig(population_size=6, max_iterations=2), seed=18)
        tasks, sinr, distances, variance, _ = self.sim.sample_slot(0)
        self.args = tasks, self.sim.env.users, self.sim.env.edge_servers, self.sim.env.cloud, sinr, distances

    def test_cloud_and_edge_share_same_radio_capacity(self):
        tasks, users, edges, cloud, sinr, distances = self.args
        distances[:] = [10., 100.]
        allocation = uniform_allocation([1, 3, 3, 0], tasks, users, edges, cloud, distances)
        evaluation = self.sim.baseline_runner.cost_engine.evaluate_allocation(
            tasks, users, edges, cloud, *allocation, sinr, distances)
        self.assertAlmostEqual(evaluation.edge_bw_allocations[1], 1.)
        self.assertLessEqual(allocation[1][allocation[0] == 3].sum(), cloud.f_total)
        np.testing.assert_allclose(allocation[2][:3], 1/3)

    def test_optimizer_uses_identical_uniform_allocator(self):
        optimizer = WEKBFPAOptimizer(self.sim.opt_cfg)
        out = optimizer.optimize(*self.args, allocation_mode="uniform")
        expected = uniform_allocation(out.best_targets, *self.args[:4], self.args[5])
        np.testing.assert_array_equal(out.best_targets, expected[0])
        np.testing.assert_allclose(out.best_cpu, expected[1])
        np.testing.assert_allclose(out.best_bw, expected[2])

    def test_joint_allocator_also_respects_shared_uplink(self):
        tasks, users, edges, cloud, sinr, distances = self.args
        distances[:] = [10., 100.]
        optimizer = WEKBFPAOptimizer(self.sim.opt_cfg)
        chromosome = np.concatenate(([1., 3., 3., 0.], np.ones(4), [1., 2., 3., 1.]))
        allocation = optimizer.decode_chromosome(chromosome, tasks, users, edges, cloud, distances=distances)
        np.testing.assert_allclose(allocation[2][:3], [1/6, 2/6, 3/6])
        evaluation = optimizer.cost_engine.evaluate_allocation(tasks, users, edges, cloud, *allocation, sinr, distances)
        self.assertAlmostEqual(evaluation.edge_bw_allocations[1], 1.)

    def test_validation_does_not_update_policy_or_replay(self):
        bank = validation_bank(300042, 2)
        agent = self.sim.dqn
        agent.run(*bank[0][0], training=True)
        weights = copy.deepcopy(agent.weights)
        memory, updates, epsilon = len(agent.memory), agent.updates, agent.epsilon
        validate(agent, bank)
        self.assertEqual((memory, updates, epsilon), (len(agent.memory), agent.updates, agent.epsilon))
        for before, after in zip(weights, agent.weights):
            np.testing.assert_array_equal(before, after)

    def test_dense_rewards_telescope_to_final_objective(self):
        tasks, users, edges, cloud, sinr, distances = self.args
        allocation = uniform_allocation(np.zeros(4, dtype=int), tasks, users, edges, cloud, distances)
        initial = self.sim.dqn.decoder.cost_engine.evaluate_allocation(
            tasks, users, edges, cloud, *allocation, sinr, distances).total_cost
        out = self.sim.dqn.run(*self.args, training=True)
        reward = sum(transition[2] for transition in self.sim.dqn.memory)
        self.assertAlmostEqual(reward, (initial-out.best_cost)/(4*self.sim.dqn.config.reward_scale))

    def test_legacy_log_reward_is_an_explicit_different_objective(self):
        tasks,users,edges,cloud,sinr,distances=self.args
        agent=DQNBaseline(config=DQNConfig(reward_transform='log'))
        allocation=uniform_allocation(np.zeros(len(tasks),dtype=int),tasks,users,edges,cloud,distances)
        initial=agent.decoder.cost_engine.evaluate_allocation(tasks,users,edges,cloud,*allocation,sinr,distances).total_cost
        result=agent.run(*self.args,training=True)
        reward=sum(transition[2] for transition in agent.memory)
        self.assertAlmostEqual(reward,np.log1p(initial/len(tasks))-np.log1p(result.best_cost/len(tasks)))

    def test_droo_paper_quantization_example_and_updates(self):
        candidates = order_preserving_candidates([.2, .4, .7, .9])
        for example in [[0, 0, 1, 1], [0, 1, 1, 1], [0, 0, 0, 1], [1, 1, 1, 1]]:
            self.assertTrue(any(np.array_equal(example, candidate) for candidate in candidates))
        agent = DROOAdapted(seed=4, batch_size=2)
        for _ in range(3):
            out = agent.run(*self.args, training=True)
        self.assertGreater(agent.updates, 0)
        self.assertTrue(np.isfinite(out.best_cost))
        home = np.argmin(self.args[5], axis=1) + 1
        self.assertTrue(np.all((out.best_targets == 0) | (out.best_targets == home)))
        weights = copy.deepcopy(agent.weights)
        agent.run(*self.args, training=False)
        for before, after in zip(weights, agent.weights):
            np.testing.assert_array_equal(before, after)

    def test_delay_sensitivity_zero_and_positive(self):
        out = self.sim.dqn.run(*self.args)
        unchanged = delay_adjusted_metrics(out, CostConfig(), 0.)
        delayed = delay_adjusted_metrics(out, CostConfig(), .1)
        self.assertAlmostEqual(unchanged["delay_adjusted_cost"], out.best_cost)
        self.assertGreater(delayed["delay_adjusted_cost"], out.best_cost)
        self.assertGreaterEqual(delayed["delay_adjusted_violation_pct"], out.evaluation_result.deadline_violation_rate)

    def test_end_to_end_seed_partitions_controls_and_exports(self):
        with tempfile.TemporaryDirectory() as directory:
            df = run_topology_shock(slots=6, training_slots=8, training_budgets=[4, 8], seeds=[42],
                                    tuning_slots=4, validation_slots=2, validation_interval=4,
                                    population=4, iterations=1, fading_values=[25., 35.], output_dir=directory)
            output = Path(directory)
            self.assertEqual(set(df.training_budget), {4, 8})
            self.assertEqual(set(df.fading_db), {0., 25., 35.})
            self.assertTrue((df.groupby(["seed", "training_budget", "fading_db", "slot"]).snapshot_hash.nunique() == 1).all())
            nominal = df[(df.condition == "nominal") & (df.phase == "pre")]
            for fade in [25., 35.]:
                shock = df[(df.fading_db == fade) & (df.phase == "pre")]
                keys = ["training_budget", "slot", "algorithm"]
                pd.testing.assert_frame_equal(nominal.set_index(keys)[["cost", "snapshot_hash"]].sort_index(),
                                              shock.set_index(keys)[["cost", "snapshot_hash"]].sort_index())
            self.assertTrue((df.delay_adjusted_cost >= df.cost - 1e-6).all())
            reports = json.loads((output / "recovery.json").read_text())
            self.assertTrue(all(not report["adaptation_claim_eligible"] for report in reports if not report["nominal_quality_pass"]))
            config = json.loads((output / "config.json").read_text())
            self.assertNotIn(42, config["seed_partitions"]["tuning_validation"])
            self.assertTrue((pd.read_csv(output / "validation.csv").slot <= 8).all())
            for filename in ["paired_summary.csv", "normalized_adaptation.csv", "quality_gate.json", "learning_curves.pdf", "training_budget.pdf"]:
                self.assertTrue((output / filename).exists(), filename)
            plot_adaptation(pd.read_csv(output / "slots.csv"), 6, 35., output)
            plot_learning(pd.read_csv(output / "validation.csv"), df, output)
