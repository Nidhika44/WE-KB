import copy
import pickle
import json
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from experiments.run_revision import initialize, variants, main, summarize, METRICS,run_case
from we_kb_fpa.simulation_executor import SimulationExecutor
from we_kb_fpa.anytime import AnytimeConfig, AnytimeWEKBFPA
from we_kb_fpa.evaluation import snapshots,validate,validation_bank
from we_kb_fpa.learner_state import save_learner_state, load_learner_state
from we_kb_fpa.dqn import DQNBaseline
from we_kb_fpa.droo import DROOAdapted
from we_kb_fpa.guarded_dqn import AnchoredFrozenDQN
from we_kb_fpa.config import NetworkConfig


class TestFairProtocol(unittest.TestCase):
    def test_no_kb_has_no_memory_and_anchors_match_each_budget(self):
        config=dict(seeds=[10001],budgets_ms=[10.,25.],model_seeds=[])
        state=initialize(config,10001,8)
        for name,optimizer in state['optimizers'].items():
            self.assertEqual(name in state['knowledge'],optimizer.config.use_kb)
        for budget in [10.,25.]:
            self.assertEqual(variants([budget])[f'Anchor-Only-{budget:g}ms'].budget_ms,budget)

    def test_binary_scope_matches_droo_for_every_operator(self):
        _,args,variance,_=next(snapshots(10001,1))
        home=np.argmin(args[5],axis=1)+1
        for strategy in ['fpa','ga','pso','anchor']:
            agent=AnytimeWEKBFPA(AnytimeConfig(strategy=strategy,action_scope='binary_nearest',budget_ms=50.,max_evaluations=20))
            result=agent.optimize(*args,channel_variance=variance)
            self.assertTrue(np.all((result.best_targets==0)|(result.best_targets==home)))

    def test_droo_quality_gate_has_matched_action_space(self):
        result=validate(DROOAdapted(),validation_bank(1234,2))
        self.assertEqual(result['reference_action_space'],'binary_nearest_edge')

    def test_channel_guidance_uses_only_shared_current_observation(self):
        sample=SimulationExecutor.sample_slot
        optimize=AnytimeWEKBFPA.optimize
        observed=[]
        def private_variance(sim,*args,**kwargs):
            tasks,sinr,distances,_,mean=sample(sim,*args,**kwargs)
            return tasks,sinr,distances,float('nan'),mean
        def inspect(optimizer,*args,**kwargs):
            expected=float(np.var(np.log1p(args[4])))
            self.assertEqual(kwargs['channel_variance'],expected)
            observed.append(expected)
            return optimize(optimizer,*args,**kwargs)
        config=dict(seeds=[10001],budgets_ms=[.001],model_seeds=[],resume=True,slots=6,
                    scale_slots=1,workers=1,checkpoint_every=1,include_current=False)
        with tempfile.TemporaryDirectory() as folder,patch.object(SimulationExecutor,'sample_slot',private_variance),patch.object(AnytimeWEKBFPA,'optimize',inspect):
            run_case(config,folder,'shock',10001,8,35.)
        self.assertTrue(observed)

    def test_full_state_roundtrip_preserves_next_learning_update(self):
        bank=[args for _,args,_,_ in snapshots(1234,6)]
        for learner in [DQNBaseline(),DROOAdapted(batch_size=2)]:
            for args in bank[:5]:
                learner.run(*args,training=True)
            with tempfile.TemporaryDirectory() as folder:
                path=Path(folder)/'agent.pkl'
                save_learner_state(learner,path,dict(seed=42,training_budget=5))
                restored,metadata=load_learner_state(path)
                self.assertEqual(metadata['seed'],42)
                self.assertEqual(len(restored.memory),len(learner.memory))
                self.assertEqual(restored.adam.steps,learner.adam.steps)
                first=learner.run(*bank[5],training=True)
                second=restored.run(*bank[5],training=True)
                self.assertEqual(first.best_cost,second.best_cost)
                for a,b in zip(learner.weights,restored.weights):
                    np.testing.assert_array_equal(a,b)
                if isinstance(learner,DQNBaseline):
                    for a,b in zip(learner.target,restored.target):
                        np.testing.assert_array_equal(a,b)
                damaged=pickle.loads(path.read_bytes())
                damaged['payload']=damaged['payload']+b'corrupted'
                path.write_bytes(pickle.dumps(damaged))
                with self.assertRaises(ValueError):
                    load_learner_state(path)

    def test_greedy_online_learns_without_exploratory_actions(self):
        _,args,_,_=next(snapshots(1234,1))
        agent=DQNBaseline()
        agent.run(*args)
        frozen=copy.deepcopy(agent)
        agent.epsilon=1.
        a=agent.run(*args,training=True,explore=False)
        b=frozen.run(*args,training=False)
        np.testing.assert_array_equal(a.best_targets,b.best_targets)
        self.assertEqual(len(agent.memory),8)

    def test_frozen_anchor_control_is_explicit_and_does_not_train_the_policy(self):
        _,args,_,_=next(snapshots(1234,1))
        agent=DQNBaseline()
        agent.run(*args,training=False)
        weights=[w.copy() for w in agent.weights]
        guard=AnchoredFrozenDQN(agent,10.,NetworkConfig(num_users=8,num_edge_servers=3),1234)
        result=guard.run(*args,training=False)
        self.assertLessEqual(result.best_cost,guard.last_diagnostics['greedy_cost']+1e-9)
        self.assertEqual(agent.updates,0)
        self.assertEqual(len(agent.memory),0)
        for a,b in zip(weights,agent.weights):
            np.testing.assert_array_equal(a,b)
        with self.assertRaises(ValueError):
            guard.run(*args,training=True)

    def test_pairing_matches_budget_and_droo_action_scope(self):
        with tempfile.TemporaryDirectory() as folder:
            rows=[]
            for algorithm,cost in [('Anytime-10ms',10.),('Anytime-25ms',1.),('Categorical-GA-10ms',8.),
                                   ('Categorical-GA-25ms',0.),('Binary-Anytime-10ms',20.),('DROO-Adapted-Online-FullState',5.)]:
                rows.append(dict(study='shock',users=8,fade=35.,phase='post',seed=10001,algorithm=algorithm,
                                 **{metric:cost for metric in METRICS},budget_overrun=False,search_triggered=True,evaluations=1))
            with patch('experiments.run_revision.plot_results'):
                summarize(pd.DataFrame(rows),Path(folder))
            pairs=pd.read_csv(Path(folder)/'paired_effects.csv')
            ga=pairs[pairs.baseline.str.startswith('Categorical-GA')]
            self.assertTrue(all(a.rsplit('-',1)[1]==b.rsplit('-',1)[1] for a,b in zip(ga.algorithm,ga.baseline)))
            droo=pairs[pairs.baseline.str.startswith('DROO')]
            self.assertTrue(droo.algorithm.str.startswith('Binary-Anytime-').all())

    def test_final_protocol_freeze_and_change_rejection_without_test_access(self):
        with tempfile.TemporaryDirectory() as folder:
            arguments=['run_revision','--quick','--study','scalability','--no-drl','--skip-current',
                       '--evaluation-role','final','--seeds','30001','--budgets-ms','10','--output-dir',folder]
            with patch.object(sys,'argv',arguments+['--freeze-protocol']):
                main()
            protocol=Path(folder)/'frozen_protocol.json'
            self.assertTrue(protocol.exists())
            self.assertFalse((Path(folder)/'slots.csv').exists())
            altered=arguments.copy();altered[altered.index('--budgets-ms')+1]='25'
            with patch.object(sys,'argv',altered+['--protocol',str(protocol)]):
                with self.assertRaises(SystemExit):
                    main()
            with patch.object(sys,'argv',arguments+['--protocol',str(protocol)]),patch('experiments.run_revision.run_case',side_effect=RuntimeError('accepted without reading test data')):
                with self.assertRaisesRegex(RuntimeError,'accepted without reading test data'):
                    main()


if __name__=='__main__':
    unittest.main()
