import unittest
from types import SimpleNamespace
import numpy as np
from we_kb_fpa.anytime import AnytimeConfig,AnytimeWEKBFPA
from we_kb_fpa.computation_control import DelayProfile,ComputationController
from we_kb_fpa.evaluation import snapshots,delay_adjusted_metrics
from we_kb_fpa.config import CostConfig
from experiments.develop_stopping import candidate_grid


class TestComputationControl(unittest.TestCase):
    def setUp(self):
        _,self.args,_,_=next(snapshots(10001,1))
        self.output=AnytimeWEKBFPA(AnytimeConfig(strategy='anchor',budget_ms=10.)).optimize(*self.args)
        self.profile=DelayProfile(self.output,CostConfig())

    def test_profile_matches_existing_objective_at_deadline_boundaries(self):
        thresholds=[m.deadline_sec-m.latency_total for m in self.output.evaluation_result.task_metrics]
        for delay in [0.,.001,.1]+[max(0.,s+1e-8) for s in thresholds]:
            expected=delay_adjusted_metrics(self.output,CostConfig(),delay)['delay_adjusted_cost']
            self.assertAlmostEqual(self.profile.score(delay),expected,places=8)
        self.assertGreaterEqual(self.profile.waiting_cost(0.,.1),0.)

    def test_stop_for_stagnation_and_continue_for_large_recent_gain(self):
        controller=ComputationController(window=8,minimum=4)
        for _ in range(3):
            controller.observe(0.,.001)
        self.assertTrue(controller.continue_search(self.profile,.001))
        controller.observe(0.,.001)
        self.assertFalse(controller.continue_search(self.profile,.001))
        for _ in range(8):
            controller.observe(1e9,.001)
        self.assertTrue(controller.continue_search(self.profile,.001))

    def test_crossing_deadline_changes_the_same_stopping_decision(self):
        def profile(latency):
            return DelayProfile(SimpleNamespace(best_cost=1.,evaluation_result=SimpleNamespace(
                task_metrics=[SimpleNamespace(deadline_sec=1.,latency_total=latency)])),CostConfig())
        controller=ComputationController(window=4,minimum=4,optimism=0.,forecast_factor=1.)
        for _ in range(4):
            controller.observe(2.,.02)
        self.assertTrue(controller.continue_search(profile(.1),.09))
        self.assertFalse(controller.continue_search(profile(.9),.09))

    def test_shared_rule_retains_raw_greedy_bound_and_reports_actual_output(self):
        for strategy in ['fpa','ga']:
            optimizer=AnytimeWEKBFPA(AnytimeConfig(strategy=strategy,deadline_aware=True,event_trigger=False,budget_ms=50.,max_evaluations=30))
            result=optimizer.optimize(*self.args)
            diagnostics=optimizer.last_diagnostics
            self.assertTrue(diagnostics['deadline_aware'])
            self.assertLessEqual(result.best_cost,diagnostics['greedy_cost']+1e-9)
            self.assertEqual(result.best_cost,result.convergence_curve[-1])
            self.assertTrue(np.isfinite(delay_adjusted_metrics(result,CostConfig())['delay_adjusted_cost']))

    def test_parameters_and_observations_are_validated(self):
        for settings in [dict(minimum=0),dict(window=1),dict(optimism=-1.),dict(forecast_factor=.5)]:
            with self.assertRaises(ValueError):
                ComputationController(**settings)
        with self.assertRaises(ValueError):
            ComputationController().observe(-1.,.01)

    def test_development_tuning_effort_is_equal(self):
        grid=candidate_grid(10.)
        counts={}
        for cfg in grid.values():
            arm=(cfg.strategy,cfg.deadline_aware,cfg.stopping_enabled)
            counts[arm]=counts.get(arm,0)+1
            self.assertEqual(cfg.budget_ms,10.)
            self.assertTrue(cfg.use_kb and cfg.warm_start)
            self.assertFalse(cfg.event_trigger)
        self.assertEqual(set(counts.values()),{9})
        self.assertEqual(len(counts),6)


if __name__=='__main__':
    unittest.main()
