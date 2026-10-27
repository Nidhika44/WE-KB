import copy
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from experiments.run_joint import run_case
from we_kb_fpa.checkpoint import Checkpoint
from we_kb_fpa.config import NetworkConfig
from we_kb_fpa.evaluation import snapshots,fingerprint
from we_kb_fpa.joint_allocation import JointAllocationConfig
from we_kb_fpa.joint_search import JointOffloadingOptimizer,JointSearchConfig,offered_task_metrics


class TestJointSearch(unittest.TestCase):
    def test_same_anchors_and_allocator_give_same_seed_solution(self):
        stream=snapshots(10001,1)
        _,args,_,_=next(stream)
        before=fingerprint(args)
        outputs=[]
        for strategy in ['fpa','ga','anchor','greedy']:
            optimizer=JointOffloadingOptimizer(JointSearchConfig(strategy=strategy,budget_ms=100.,max_evaluations=2),net=stream.sim.net_cfg)
            output=optimizer.optimize(*args)
            self.assertTrue(output.solution.feasible)
            self.assertEqual(fingerprint(args),before)
            outputs.append(output.solution)
        for solution in outputs[1:]:
            self.assertEqual(solution.admitted,outputs[0].admitted)
            self.assertAlmostEqual(solution.service_cost,outputs[0].service_cost,places=7)
            np.testing.assert_allclose(solution.cpu,outputs[0].cpu)
            np.testing.assert_allclose(solution.bandwidth,outputs[0].bandwidth)

    def test_impossible_tasks_are_counted_as_offered_misses(self):
        stream=snapshots(10001,1)
        _,args,_,_=next(stream)
        for task in args[0]:
            task.deadline_sec=.000001
        optimizer=JointOffloadingOptimizer(JointSearchConfig(budget_ms=50.),net=stream.sim.net_cfg)
        output=optimizer.optimize(*args)
        metrics=offered_task_metrics(output,args[0],.05)
        self.assertEqual(metrics['admitted_tasks'],0)
        self.assertEqual(metrics['offered_miss_pct'],100.)
        self.assertEqual(metrics['service_cost'],0.)
        self.assertIsNone(metrics['service_cost_per_admitted'])

    def test_no_kb_has_no_hidden_memory_maintenance(self):
        optimizer=JointOffloadingOptimizer(JointSearchConfig(use_knowledge=False))
        self.assertIsNone(optimizer.knowledge)

    def test_resume_preserves_committed_joint_decisions(self):
        settings=dict(network=asdict(NetworkConfig(num_users=8,num_edge_servers=3)),
                      allocation=asdict(JointAllocationConfig(decision_reserve_sec=.005)),slots=6,checkpoint_every=1,
                      methods={strategy:asdict(JointSearchConfig(strategy=strategy,budget_ms=5.,max_evaluations=2)) for strategy in ['fpa','ga']})
        captured=[]
        save=Checkpoint.save
        def interrupt(checkpoint,state,compact=False):
            save(checkpoint,state,compact)
            if state.get('stream') is not None and state['stream'].next_slot==2:
                captured.extend(copy.deepcopy(state['records']))
                raise KeyboardInterrupt
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(Checkpoint,'save',interrupt):
                with self.assertRaises(KeyboardInterrupt):
                    run_case(settings,Path(folder),10001,35.)
            resumed=run_case(settings,Path(folder),10001,35.)
            self.assertEqual(resumed[:len(captured)],captured)
            self.assertEqual(len(resumed),12)
            self.assertEqual(len({(row['slot'],row['algorithm']) for row in resumed}),12)
            self.assertEqual(run_case(settings,Path(folder),10001,35.),resumed)


if __name__=='__main__':
    unittest.main()
