import copy
from dataclasses import replace
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from experiments.run_revision import initialize, run_case, variants
from we_kb_fpa.anytime import AnytimeConfig, AnytimeWEKBFPA
from we_kb_fpa.checkpoint import Checkpoint
from we_kb_fpa.evaluation import fingerprint


class TestAnytime(unittest.TestCase):
    def setUp(self):
        self.config = dict(seeds=[99001], budgets_ms=[5.], model_seeds=[], models_dir='',
                           model_budget=20000, slots=6, scale_slots=2, workers=1,
                           checkpoint_every=1, resume=True, include_current=False)
        state = initialize(self.config,99001,8)
        sim = state['sim']
        tasks,sinr,distances,variance,_ = sim.sample_slot(0)
        self.args = tasks,sim.env.users,sim.env.edge_servers,sim.env.cloud,sinr,distances
        self.net = sim.net_cfg

    def test_elite_scoring_feasibility_and_input_integrity(self):
        before = fingerprint(self.args)
        for strategy in ['fpa','pso','ga','anchor']:
            optimizer = AnytimeWEKBFPA(AnytimeConfig(strategy=strategy,budget_ms=50,max_evaluations=15),net_config=self.net)
            out = optimizer.optimize(*self.args)
            allocation = out.best_targets
            self.assertTrue(np.issubdtype(allocation.dtype,np.integer))
            self.assertLessEqual(out.best_cost,optimizer.last_diagnostics['greedy_cost']+1e-9)
            self.assertEqual(fingerprint(self.args),before)
            evaluation = optimizer.runner.cost_engine.evaluate_allocation(*self.args[:4],out.best_targets,out.best_cpu,out.best_bw,*self.args[4:])
            self.assertAlmostEqual(out.best_cost,evaluation.total_cost)
            self.assertTrue(all(v<=1+1e-9 for v in evaluation.edge_bw_allocations.values()))
            self.assertTrue(all(a>=b for a,b in zip(out.convergence_curve,out.convergence_curve[1:])))

    def test_event_trigger_detects_fade_and_skips_identical_context(self):
        optimizer = AnytimeWEKBFPA(AnytimeConfig(budget_ms=50,max_evaluations=12),net_config=self.net)
        optimizer.optimize(*self.args)
        self.assertTrue(optimizer.last_diagnostics['search_triggered'])
        optimizer.optimize(*self.args)
        self.assertFalse(optimizer.last_diagnostics['search_triggered'])
        faded = copy.deepcopy(self.args)
        faded[4][:] *= 10**(-35/10)
        optimizer.optimize(*faded)
        self.assertTrue(optimizer.last_diagnostics['search_triggered'])

    def test_tiny_budget_returns_scored_greedy_and_reports_overrun(self):
        optimizer = AnytimeWEKBFPA(AnytimeConfig(budget_ms=.000001),net_config=self.net)
        out = optimizer.optimize(*self.args)
        self.assertTrue(optimizer.last_diagnostics['budget_overrun'])
        self.assertEqual(out.best_cost,optimizer.last_diagnostics['greedy_cost'])

    def test_resume_preserves_committed_slots_and_warm_state(self):
        with tempfile.TemporaryDirectory() as directory:
            original = Checkpoint.save
            captured = []
            def interrupt(checkpoint,state,compact=False):
                original(checkpoint,state,compact)
                if state.get('next_slot')==2:
                    captured.extend(copy.deepcopy(state['records']))
                    self.assertIsNotNone(state['optimizers']['Anytime-5ms'].previous_targets)
                    raise KeyboardInterrupt
            with patch.object(Checkpoint,'save',interrupt):
                with self.assertRaises(KeyboardInterrupt):
                    run_case(self.config,Path(directory),'shock',99001,8,35.)
            resumed = run_case(self.config,Path(directory),'shock',99001,8,35.)
            self.assertEqual(resumed[:len(captured)],captured)
            expected_methods = len(variants([5.]))+2
            self.assertEqual(len(resumed),6*expected_methods)
            self.assertEqual(len({(r['slot'],r['algorithm']) for r in resumed}),len(resumed))
            loaded = run_case(self.config,Path(directory),'shock',99001,8,35.)
            self.assertEqual(loaded,resumed)

    def test_invalid_configuration(self):
        for setting in [dict(budget_ms=float('nan')),dict(population=1),dict(patience=0),dict(strategy='rounding')]:
            with self.assertRaises(ValueError):
                AnytimeConfig(**setting)


if __name__=='__main__':
    unittest.main()
