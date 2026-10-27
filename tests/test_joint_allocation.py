import time
import unittest
import numpy as np
from we_kb_fpa.config import NetworkConfig,CostConfig
from we_kb_fpa.environment_model import UserDevice,EdgeServer,CloudServer
from we_kb_fpa.task_model import Task
from we_kb_fpa.joint_allocation import FixedPlacementAllocator,JointAllocationConfig,hard_qos_mask


class TestJointAllocation(unittest.TestCase):
    def setUp(self):
        self.net=NetworkConfig(num_users=2,num_edge_servers=1)
        self.users=[UserDevice(i,0.,0.,f_local=1e9,p_tx=.2) for i in range(2)]
        self.tasks=[Task(i,i,1e5,1000.,.5) for i in range(2)]
        self.edges=[EdgeServer(1,0.,0.)]
        self.cloud=CloudServer()
        self.args=(self.tasks,self.users,self.edges,self.cloud,np.full((2,1),10.),np.full((2,1),50.))
        self.allocator=FixedPlacementAllocator(net=self.net)

    def test_joint_objective_and_constraint_gradients(self):
        p,status=self.allocator.problem(self.args,[1,2])
        self.assertEqual(status,'READY')
        x=p['x']*.9
        step=1e-6
        gradient=np.zeros_like(x);jac=np.zeros_like(p['jacobian'](x))
        for j in range(len(x)):
            plus=x.copy();minus=x.copy();plus[j]+=step;minus[j]-=step
            gradient[j]=(p['objective'](plus)-p['objective'](minus))/(2*step)
            jac[:,j]=(p['constraints'](plus)-p['constraints'](minus))/(2*step)
        np.testing.assert_allclose(gradient,p['gradient'](x),rtol=1e-5,atol=1e-6)
        np.testing.assert_allclose(jac,p['jacobian'](x),rtol=1e-5,atol=1e-6)

    def test_single_remote_cpu_matches_analytic_optimum_and_certificate(self):
        out=self.allocator.solve(self.args,[1,-1])
        self.assertTrue(out.feasible)
        self.assertTrue(out.certified,out.diagnostics)
        alpha=self.allocator.cost.alpha
        energy=self.net.kappa_local*self.users[0].f_local**2*self.tasks[0].total_cycles
        expected=(alpha*energy/(2*(1-alpha)*self.net.kappa_edge*self.tasks[0].deadline_sec))**(1/3)
        self.assertAlmostEqual(out.cpu[0]/expected,1.,places=4)
        self.assertLessEqual(out.lower_bound,out.service_cost+1e-7)

    def test_shared_radio_and_cpu_capacities_are_independently_validated(self):
        for targets in [[1,1],[1,2],[2,2],[0,1]]:
            out=self.allocator.solve(self.args,targets)
            self.assertTrue(out.feasible,out.status)
            self.assertLessEqual(out.bandwidth.sum(),1.)
            self.assertLessEqual(out.cpu[out.targets==1].sum(),self.edges[0].f_total)
            self.assertLessEqual(out.cpu[out.targets==2].sum(),self.cloud.f_total)
            self.assertEqual(out.evaluation.total_penalties,0.)
            self.assertTrue(all(m.latency_total+self.allocator.config.decision_reserve_sec<=m.deadline_sec+1e-9 for m in out.evaluation.task_metrics))

    def test_pruning_never_reopens_impossible_targets(self):
        self.tasks[0].deadline_sec=.00001
        mask=hard_qos_mask(self.args,self.allocator.config,self.net)
        self.assertFalse(mask[0].any())
        fixed=self.allocator.solve(self.args,[0,0])
        self.assertFalse(fixed.feasible)
        self.assertTrue(fixed.status.startswith('PROVEN_INFEASIBLE'))
        repaired=self.allocator.repair(self.args,[0,0])
        self.assertEqual(repaired.targets[0],-1)
        self.assertEqual(repaired.admitted,1)

    def test_admission_is_primary_and_all_drop_is_explicit(self):
        dropped=self.allocator.solve(self.args,[-1,-1])
        served=self.allocator.solve(self.args,[0,0])
        self.assertTrue(dropped.feasible)
        self.assertEqual(dropped.admitted,0)
        self.assertEqual(dropped.service_cost,0.)
        self.assertLess(served.rank,dropped.rank)

    def test_expired_solver_has_safe_fallback_without_false_infeasibility(self):
        unresolved=self.allocator.solve(self.args,[1,1],deadline=time.perf_counter()-1)
        self.assertFalse(unresolved.feasible)
        self.assertEqual(unresolved.status,'TIME_LIMIT_UNRESOLVED')
        repaired=self.allocator.repair(self.args,[1,0],deadline=time.perf_counter()-1)
        self.assertTrue(repaired.feasible)
        self.assertEqual(repaired.targets[0],-1)
        self.assertEqual(repaired.targets[1],0)

    def test_phase_one_certifies_coupled_resource_infeasibility(self):
        rate=self.edges[0].bandwidth_total*np.log2(11.)
        for task in self.tasks:
            task.data_size_bits=.1*rate
            task.cycles_per_bit=4e9/task.data_size_bits
            task.deadline_sec=.402
        problem,status=self.allocator.problem(self.args,[1,1])
        self.assertEqual(status,'READY')
        result=self.allocator.solve(self.args,[1,1])
        self.assertFalse(result.feasible)
        self.assertEqual(result.status,'PROVEN_INFEASIBLE_PHASE_I')
        self.assertGreater(result.diagnostics['phase_one_lower_bound'],0.)
        repaired=self.allocator.repair(self.args,[1,1])
        self.assertTrue(repaired.feasible)
        self.assertEqual(repaired.admitted,1)

    def test_coordination_fixture_has_real_assignment_contention(self):
        edges=[EdgeServer(1,0.,0.,f_total=10e9),EdgeServer(2,10.,0.,f_total=10e9,queue_delay_current=.001)]
        cloud=CloudServer(propagation_delay=1.)
        tasks=[Task(i,i,1e5,40000.,.6) for i in range(2)]
        args=(tasks,self.users,edges,cloud,np.full((2,2),10.),np.full((2,2),50.))
        overloaded=self.allocator.solve(args,[1,1])
        coordinated=self.allocator.solve(args,[1,2])
        self.assertFalse(overloaded.feasible)
        self.assertTrue(coordinated.feasible)
        self.assertEqual(coordinated.admitted,2)


if __name__=='__main__':
    unittest.main()
