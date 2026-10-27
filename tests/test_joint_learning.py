import copy
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
import numpy as np
from experiments.train_joint import fit
from we_kb_fpa.config import NetworkConfig
from we_kb_fpa.evaluation import snapshots,fingerprint
from we_kb_fpa.joint_allocation import JointAllocationConfig
from we_kb_fpa.joint_learning import JointLearner,JointLearningConfig,utility
from we_kb_fpa.joint_search import offered_task_metrics


class TestJointLearning(unittest.TestCase):
    def test_reward_preserves_per_snapshot_rank(self):
        from types import SimpleNamespace
        values=[SimpleNamespace(admitted=n,service_cost=c) for n in range(4) for c in [0.,1.,1e8]]
        for left in values:
            for right in values:
                if (-left.admitted,left.service_cost)<(-right.admitted,right.service_cost):
                    self.assertGreater(utility(left),utility(right))

    def test_both_learners_use_shared_physics_and_count_drops(self):
        stream=snapshots(10001,1);_,args,_,_=next(stream)
        for task in args[0]:
            task.deadline_sec=.000001
        original=fingerprint(args)
        for kind in ['dqn','categorical']:
            learner=JointLearner(JointLearningConfig(kind=kind,batch_size=2),net=stream.sim.net_cfg)
            output=learner.run(args,training=True)
            self.assertTrue(output.solution.feasible)
            self.assertEqual(offered_task_metrics(output,args[0],.05)['offered_miss_pct'],100.)
            self.assertEqual(original,fingerprint(args))
            self.assertGreater(output.diagnostics['learning_updates'],0 if kind=='dqn' else -1)

    def test_frozen_has_no_updates_online_state_is_independent(self):
        stream=snapshots(10001,1);_,args,_,_=next(stream)
        learner=JointLearner(JointLearningConfig(batch_size=2),net=stream.sim.net_cfg)
        learner.run(args,training=True)
        frozen=copy.deepcopy(learner);online=copy.deepcopy(learner)
        weights=[w.copy() for w in frozen.agent.weights]
        updates=frozen.agent.updates
        frozen.run(args,training=False);online.run(args,training=True)
        self.assertEqual(frozen.agent.updates,updates)
        self.assertGreater(online.agent.updates,updates)
        for before,after in zip(weights,frozen.agent.weights):
            np.testing.assert_array_equal(before,after)

    def test_training_resume_preserves_model_and_replay(self):
        from unittest.mock import patch
        from we_kb_fpa.checkpoint import Checkpoint
        config=JointLearningConfig(batch_size=2,budget_ms=5.)
        allocation=JointAllocationConfig(decision_reserve_sec=.005)
        net=NetworkConfig(num_users=8,num_edge_servers=3)
        save=Checkpoint.save
        with tempfile.TemporaryDirectory() as folder:
            def interrupt(checkpoint,state,compact=False):
                save(checkpoint,state,compact)
                if state['completed']==2:
                    raise KeyboardInterrupt
            with patch.object(Checkpoint,'save',interrupt):
                with self.assertRaises(KeyboardInterrupt):
                    fit(Path(folder),'resume',config,allocation,net,42,4,2,1)
            result=fit(Path(folder),'resume',config,allocation,net,42,4,2,1)
            again=fit(Path(folder),'resume',config,allocation,net,42,4,2,1)
            self.assertEqual(result['completed'],4)
            self.assertEqual(len(result['agent'].agent.memory),32)
            self.assertEqual(result['agent'].agent.updates,again['agent'].agent.updates)
            for a,b in zip(result['agent'].agent.weights,again['agent'].agent.weights):
                np.testing.assert_array_equal(a,b)


if __name__=='__main__':
    unittest.main()
