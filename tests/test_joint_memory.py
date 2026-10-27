import copy
from types import SimpleNamespace
import unittest
import numpy as np
from we_kb_fpa.evaluation import snapshots
from we_kb_fpa.joint_memory import JointStateMemory


class TestJointMemory(unittest.TestCase):
    def setUp(self):
        _,self.args,_,_=next(snapshots(10001,1))

    def test_user_permutation_retrieves_the_correct_assignment(self):
        memory=JointStateMemory()
        targets=np.array([0,1,2,3,4,-1,1,0],dtype=np.int32)
        memory.add(self.args,SimpleNamespace(targets=targets))
        order=np.array([7,1,5,2,0,6,3,4])
        tasks,users,edges,cloud,sinr,distances=self.args
        permuted=([tasks[i] for i in order],[users[i] for i in order],edges,cloud,sinr[order],distances[order])
        np.testing.assert_array_equal(memory.query(permuted,k=1)[0],targets[order])

    def test_task_identity_and_server_context_are_retained(self):
        memory=JointStateMemory()
        first=memory.observation(self.args)[-1]
        changed=copy.deepcopy(self.args)
        changed[0][0].deadline_sec,changed[0][1].deadline_sec=changed[0][1].deadline_sec,changed[0][0].deadline_sec
        second=memory.observation(changed)[-1]
        self.assertFalse(np.array_equal(first,second))
        changed=copy.deepcopy(self.args)
        changed[3].backhaul_bandwidth*=.5
        self.assertFalse(np.array_equal(first,memory.observation(changed)[-1]))

    def test_different_edge_identity_does_not_reuse_indexed_targets(self):
        memory=JointStateMemory()
        memory.add(self.args,SimpleNamespace(targets=np.zeros(8,dtype=np.int32)))
        changed=copy.deepcopy(self.args)
        changed[2][0].server_id=123456
        self.assertEqual(memory.query(changed),[])


if __name__=='__main__':
    unittest.main()
