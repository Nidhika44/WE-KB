import unittest
from unittest.mock import patch
import numpy as np
from experiments.diagnose_dqn import configurations,select_on_calibration
from we_kb_fpa.dqn import DQNBaseline,DQNConfig
from we_kb_fpa.evaluation import snapshots


class TestDQNDiagnosis(unittest.TestCase):
    def test_grid_preserves_objective_and_selection_excludes_check_data(self):
        grid=configurations()
        self.assertEqual(len(grid),6)
        self.assertTrue(all(c.gamma==1. and c.reward_transform=='linear' for c in grid.values()))
        rows=[dict(candidate='a',role='calibration',validation_cost=2.),dict(candidate='b',role='calibration',validation_cost=1.)]
        self.assertEqual(select_on_calibration(rows),'b')
        with self.assertRaises(ValueError):
            select_on_calibration(rows+[dict(candidate='a',role='development_check',validation_cost=0.)])

    def test_constant_scaling_preserves_first_episode_policy_and_scales_return(self):
        _,args,_,_=next(snapshots(10001,1))
        agents=[DQNBaseline(config=DQNConfig(reward_scale=s),seed=17) for s in [100.,1000.]]
        outputs=[agent.run(*args,training=True,explore=False) for agent in agents]
        np.testing.assert_array_equal(outputs[0].best_targets,outputs[1].best_targets)
        np.testing.assert_allclose([r[2]/10 for r in agents[0].memory],[r[2] for r in agents[1].memory])

    def test_huber_backprop_matches_finite_differences(self):
        agent=DQNBaseline(config=DQNConfig(hidden=2,batch_size=2))
        agent._initialize(2,2)
        agent.weights=[np.full((2,2),.2),np.full(2,.1),np.full((2,2),.3),np.full(2,.1),np.full((2,2),.01),np.zeros(2)]
        states=np.array([[.3,1.2],[1.,.4]])
        actions=np.array([0,1]);rewards=np.array([.2,-.3])
        for state,action,reward in zip(states,actions,rewards):
            agent.memory.append((state,int(action),float(reward),np.zeros(2),True))
        captured=[]
        with patch.object(agent.adam,'update',side_effect=lambda weights,grads,rate:captured.extend(g.copy() for g in grads)):
            agent.learn()
        def loss():
            error=agent._forward(states,agent.weights)[0][np.arange(2),actions]-rewards
            return np.mean(np.where(abs(error)<=1,.5*error**2,abs(error)-.5))
        epsilon=1e-6
        for weight,gradient in zip(agent.weights,captured):
            numerical=np.zeros_like(weight)
            for index in np.ndindex(weight.shape):
                original=weight[index]
                weight[index]=original+epsilon;plus=loss()
                weight[index]=original-epsilon;minus=loss()
                weight[index]=original
                numerical[index]=(plus-minus)/(2*epsilon)
            np.testing.assert_allclose(gradient,numerical,atol=1e-8,rtol=1e-5)


if __name__=='__main__':
    unittest.main()
