import copy
from dataclasses import replace
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from we_kb_fpa.config import NetworkConfig
from we_kb_fpa.evaluation import snapshots,fingerprint
from we_kb_fpa.research_method import ResearchOptimizer,ResearchMethodConfig,ResearchMemory,research_metrics,optimistic_latency
from we_kb_fpa.research_reference import independent_cost_bound,solve_reference
from experiments.run_research import Scenario,method_configs,run_case,set_channel


class TestResearchExtension(unittest.TestCase):
    def setUp(self):
        stream=snapshots(50001,1);_,self.args,_,_=next(stream);self.net=stream.sim.net_cfg

    def test_exact_evaluation_matching_and_retained_greedy(self):
        before=fingerprint(self.args)
        for family in ['fpa','ga','pso']:
            optimizer=ResearchOptimizer(ResearchMethodConfig(family=family,evaluations=17,population=4),self.net,42)
            output=optimizer.optimize(self.args)
            self.assertEqual(output.diagnostics['cost_evaluations'],17)
            self.assertEqual(len(output.convergence_curve),17)
            self.assertLessEqual(output.best_cost,output.diagnostics['greedy_guard_cost']+1e-8)
            self.assertEqual(before,fingerprint(self.args))
            self.assertTrue(np.all(np.diff(output.convergence_curve)<=1e-9))

    def test_factorial_has_all_independent_combinations(self):
        configs=method_configs(dict(evaluations=12,population=4),Scenario('ablation','factorial'))
        factorial=[c for name,c in configs.items() if name.startswith('FPA-P')]
        self.assertEqual(len({(c.pruner,c.phi,c.switch,c.kb) for c in factorial}),16)
        self.assertFalse(configs['FPA+Pruner'].kb)
        self.assertTrue(configs['FPA+Pruner'].pruner)

    def test_shared_kb_initial_population_matches_before_search(self):
        memory=ResearchMemory(2);n=len(self.args[0])
        for target in [0.,1.]:
            memory.add(self.args,np.r_[np.full(n,target),np.ones(2*n)])
        outputs=[]
        for family in ['fpa','ga','pso']:
            output=ResearchOptimizer(ResearchMethodConfig(family=family,evaluations=8,population=6),self.net,7).optimize(self.args,memory)
            outputs.append(output)
        for output in outputs[1:]:
            np.testing.assert_array_equal(output.best_targets,outputs[0].best_targets)
            np.testing.assert_allclose(output.best_cpu,outputs[0].best_cpu)
            self.assertEqual(output.convergence_curve,outputs[0].convergence_curve)

    def test_disabled_pruner_reports_zero_actual_exclusions(self):
        optimizer=ResearchOptimizer(ResearchMethodConfig(pruner=False,evaluations=4,population=2),self.net)
        output=optimizer.optimize(self.args)
        self.assertEqual(output.diagnostics['pruning_rate'],0.)

    def test_reversed_adaptation_and_fixed_controls(self):
        base=ResearchMethodConfig(evaluations=4,population=2)
        optimizer=ResearchOptimizer(base,self.net)
        args=list(copy.deepcopy(self.args));n=len(args[0]);chrom=np.r_[np.ones(n),np.ones(2*n)]
        args[4][:]=.01;low,prob=optimizer.adaptation(args,chrom)
        args[4][:]=1000.;high,_=optimizer.adaptation(args,chrom)
        self.assertTrue(np.all(low>=high))
        reverse=ResearchOptimizer(replace(base,phi_direction=-1),self.net)
        rev,_=reverse.adaptation(args,chrom)
        self.assertTrue(np.all(rev>=high))
        normal=optimizer.adaptation(args,chrom)[1]
        backward=ResearchOptimizer(replace(base,switch_direction=-1),self.net).adaptation(args,chrom)[1]
        self.assertAlmostEqual(normal+backward,1.)

    def test_empty_kb_fifo_bytes_and_shape_rejection(self):
        n=len(self.args[0]);chrom=np.r_[np.zeros(n),np.ones(2*n)]
        empty=ResearchMemory(0);empty.add(self.args,chrom);self.assertEqual(empty.records,[])
        memory=ResearchMemory(2)
        for value in [0.,1.,2.]:
            memory.add(self.args,chrom+value)
        self.assertEqual(len(memory.records),2)
        self.assertGreater(memory.bytes_per_entry(),0)
        self.assertEqual(memory.records[0][2][0],1.)
        changed=copy.deepcopy(self.args);changed[2][0].server_id+=100
        self.assertEqual(memory.query(changed,1,np.random.default_rng(1)),[])

    def test_optimistic_bounds_do_not_exceed_actual_allocated_latency(self):
        output=ResearchOptimizer(ResearchMethodConfig(evaluations=12,population=4),self.net).optimize(self.args)
        bounds=optimistic_latency(self.args,self.net)
        for i,metric in enumerate(output.evaluation_result.task_metrics):
            self.assertLessEqual(bounds[i,metric.target_node],metric.latency_total+1e-9)
        self.assertLessEqual(independent_cost_bound(self.args,self.net),output.best_cost+1e-7)

    def test_no_service_is_not_zero_violations_or_missing_admission(self):
        for task in self.args[0]:
            task.deadline_sec=1e-8
        output=ResearchOptimizer(ResearchMethodConfig(evaluations=4,population=2),self.net).optimize(self.args)
        result=research_metrics(output,self.args,self.net)
        self.assertEqual(result['violation_pct'],100.)
        self.assertEqual(result['physical_violation_lower_bound_pct'],100.)
        self.assertEqual(result['admission_pct'],100.)
        self.assertEqual(result['deadline_feasible_tasks'],0)
        self.assertIsNone(result['feasible_cost_per_task'])

    def test_absolute_channel_axis_is_actual_mean_linear_sinr(self):
        matrix=set_channel(self.args[4],-5.)
        self.assertAlmostEqual(10*np.log10(matrix.mean()),-5.)

    def test_reference_unavailable_is_explicit_and_has_no_fake_optimum(self):
        with patch('we_kb_fpa.research_reference.importlib.util.find_spec',return_value=None):
            reference=solve_reference(self.args,self.net)
        self.assertEqual(reference.status,'UNAVAILABLE')
        self.assertFalse(reference.is_optimal)
        self.assertIsNone(reference.incumbent_cost)
        self.assertGreaterEqual(reference.lower_bound,0)

    def test_partial_shock_and_csi_are_applied_without_oracle_repair(self):
        settings=dict(evaluations=4,population=2,slots=2,shock_slots=4,include_drl=False,
                      checkpoint_every=1,reference_seconds=.1,mip_gap=.01)
        with tempfile.TemporaryDirectory() as folder:
            base=run_case(settings,Path(folder),Scenario('shock','control',users=8,edges=3,q=0),50001,42)
            shock=run_case(settings,Path(folder),Scenario('shock','partial',users=8,edges=3,q=0,fade_db=20.,affected_edges=(0,)),50001,42)
            for left,right in zip(base[:7],shock[:7]):
                self.assertEqual(left['realization_hash'],right['realization_hash'])
            self.assertNotEqual(base[-1]['realization_hash'],shock[-1]['realization_hash'])
            csi=run_case(settings,Path(folder),Scenario('csi','error',users=8,edges=3,q=0,csi_error_db=6.),50001,42)
            self.assertNotEqual(csi[0]['observation_hash'],csi[0]['realization_hash'])
            self.assertEqual(run_case(settings,Path(folder),Scenario('csi','error',users=8,edges=3,q=0,csi_error_db=6.),50001,42),csi)

    def test_interrupted_case_resumes_committed_rows_without_duplicates(self):
        from we_kb_fpa.checkpoint import Checkpoint
        settings=dict(evaluations=4,population=2,slots=4,shock_slots=4,include_drl=False,
                      checkpoint_every=1,reference_seconds=.1,mip_gap=.01)
        save=Checkpoint.save;captured=[]
        def interrupt(checkpoint,state,compact=False):
            save(checkpoint,state,compact)
            if state.get('next_slot')==2:
                captured.extend(copy.deepcopy(state['records']));raise KeyboardInterrupt
        scenario=Scenario('channel','resume',users=8,edges=3,q=0)
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(Checkpoint,'save',interrupt):
                with self.assertRaises(KeyboardInterrupt):
                    run_case(settings,Path(folder),scenario,50001,42)
            rows=run_case(settings,Path(folder),scenario,50001,42)
            self.assertEqual(rows[:len(captured)],captured)
            self.assertEqual(len(rows),len({(row['slot'],row['algorithm']) for row in rows}))


if __name__=='__main__':
    unittest.main()
