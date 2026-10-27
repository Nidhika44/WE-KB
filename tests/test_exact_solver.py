"""
Tests for Exact MINLP Solver.
"""

import unittest
import numpy as np
from we_kb_fpa.config import NetworkConfig, CostConfig
from we_kb_fpa.environment_model import UserDevice, EdgeServer, CloudServer
from we_kb_fpa.task_model import Task
from we_kb_fpa.exact_solver import ExactMINLPSolver, GUROBI_AVAILABLE


class TestExactSolver(unittest.TestCase):

    def setUp(self):
        self.net_cfg = NetworkConfig(num_users=3, num_edge_servers=2)
        self.cost_cfg = CostConfig()
        self.solver = ExactMINLPSolver(self.cost_cfg, self.net_cfg, time_limit_sec=15.0)

        self.users = [
            UserDevice(device_id=i, x=50.0 * (i + 1), y=50.0, p_tx=0.2, f_local=1.0e9)
            for i in range(3)
        ]
        self.tasks = [
            Task(
                task_id=i + 1,
                user_id=i,
                data_size_bits=1.0e6 * (i + 1),
                cycles_per_bit=800.0,
                deadline_sec=0.400,
            )
            for i in range(3)
        ]
        self.edges = [
            EdgeServer(server_id=1, x=100.0, y=100.0, f_total=40.0e9, bandwidth_total=20.0e6),
            EdgeServer(server_id=2, x=200.0, y=200.0, f_total=40.0e9, bandwidth_total=20.0e6),
        ]
        self.cloud = CloudServer()
        self.sinr = np.array([
            [15.0, 10.0],
            [12.0, 18.0],
            [8.0, 14.0],
        ])
        self.distances = np.array([
            [50.0, 150.0],
            [60.0, 100.0],
            [120.0, 80.0],
        ])

    def test_solve_toy_instance(self):
        result = self.solver.solve(
            self.tasks, self.users, self.edges, self.cloud, self.sinr, self.distances
        )
        self.assertIsNotNone(result)
        self.assertTrue(result.is_optimal)
        self.assertEqual(result.optimality_scope,"uniform_allocator_only")
        self.assertEqual(result.assignments_evaluated,4**3)
        self.assertEqual(result.termination_status,"EXHAUSTED")
        self.assertGreater(result.optimal_cost, 0.0)
        self.assertEqual(len(result.offloading_targets), 3)
        self.assertEqual(len(result.cpu_allocations), 3)
        self.assertEqual(len(result.bandwidth_allocations), 3)
        self.assertTrue(np.all(result.offloading_targets >= 0))
        self.assertTrue(np.all(result.offloading_targets <= 3))

    def test_timeout_does_not_claim_optimality(self):
        self.solver.time_limit=1e-12
        result=self.solver.solve(self.tasks,self.users,self.edges,self.cloud,self.sinr,self.distances)
        self.assertFalse(result.is_optimal)
        self.assertEqual(result.termination_status,"TIME_LIMIT")
        self.assertIsNone(result.relative_gap)
        self.assertTrue(np.isfinite(result.optimal_cost))
        evaluated=self.solver.cost_engine.evaluate_allocation(self.tasks,self.users,self.edges,self.cloud,
                    result.offloading_targets,result.cpu_allocations,result.bandwidth_allocations,self.sinr,self.distances)
        self.assertEqual(evaluated.total_cost,result.optimal_cost)


if __name__ == "__main__":
    unittest.main()
