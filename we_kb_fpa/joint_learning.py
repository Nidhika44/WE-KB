"""Matched sequential DQN and categorical DROO-inspired candidate learning.

The categorical candidate learner is an adaptation, not a reproduction of binary DROO.
Both policies use only current observations and the shared constrained allocator.
"""
from collections import deque
from dataclasses import dataclass
import time
import numpy as np
from we_kb_fpa.allocation import slot_features
from we_kb_fpa.dqn import DQNBaseline,DQNConfig
from we_kb_fpa.neural import Adam
from we_kb_fpa.joint_allocation import FixedPlacementAllocator,JointAllocationConfig,hard_qos_mask,latency_lower_bounds
from we_kb_fpa.joint_search import JointSearchOutput


def utility(solution):
    cost=max(0.,solution.service_cost)
    return solution.admitted-cost/(1.+cost)


@dataclass(frozen=True)
class JointLearningConfig:
    kind: str='dqn'
    budget_ms: float=50.
    hidden: int=64
    batch_size: int=32
    learning_rate: float=.0003
    candidates: int=8
    epsilon_decay: float=.9995

    def __post_init__(self):
        if self.kind not in ('dqn','categorical') or self.budget_ms<=0 or self.candidates<1 or self.hidden<1 or self.batch_size<1 or self.learning_rate<=0:
            raise ValueError('Invalid joint learner settings')


class JointLearner:
    def __init__(self,config=None,allocation_config=None,net=None,seed=42,online=False):
        self.config=config or JointLearningConfig()
        self.allocator=FixedPlacementAllocator(allocation_config or JointAllocationConfig(decision_reserve_sec=self.config.budget_ms/1000),net=net)
        if self.allocator.config.decision_reserve_sec<self.config.budget_ms/1000:
            raise ValueError('Learners must reserve the same full decision budget')
        self.agent=DQNBaseline(seed=seed,config=DQNConfig(hidden=self.config.hidden,batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,epsilon_decay=self.config.epsilon_decay))
        self.online=online
        self.replay=deque(maxlen=10000)
        self.adam=Adam()
        self.updates=0

    def propose(self,args):
        n,m=len(args[0]),len(args[2]);a=m+3
        context=slot_features(*args)
        if self.agent.weights is None:
            raise ValueError('Hybrid controls require an already trained policy')
        if self.config.kind=='categorical':
            return self.agent._forward(context,self.agent.weights)[0].reshape(n,a).argmax(axis=1)-1
        assigned=np.zeros((n,a))
        for i in range(n):
            x=np.concatenate((context,[i/n],assigned.ravel()))
            action=int(self.agent._forward(x,self.agent.weights)[0].argmax())
            assigned[i,action]=1
        return assigned.argmax(axis=1)-1

    def _categorical_learn(self):
        c=self.config
        if len(self.replay)<c.batch_size:
            return
        batch=[self.replay[i] for i in self.agent.rng.choice(len(self.replay),c.batch_size,replace=False)]
        xs,labels=map(np.asarray,zip(*batch))
        w=self.agent.weights
        logits,h1,h2=self.agent._forward(xs,w)
        logits=logits.reshape(c.batch_size,*labels.shape[1:])
        p=np.exp(logits-logits.max(axis=-1,keepdims=True));p/=p.sum(axis=-1,keepdims=True)
        dz=((p-labels)/(c.batch_size*labels.shape[1])).reshape(c.batch_size,-1)
        dh2=(dz@w[4].T)*(h2>0);dh1=(dh2@w[2].T)*(h1>0)
        self.adam.update(w,[xs.T@dh1,dh1.sum(0),h1.T@dh2,dh2.sum(0),h2.T@dz,dz.sum(0)],c.learning_rate)
        self.updates+=1

    def optimize(self,*args):
        return self.run(args,training=self.online)

    def run(self,args,training=False):
        start=time.perf_counter();deadline=start+self.config.budget_ms/1000
        n,m=len(args[0]),len(args[2]);a=m+3
        context=slot_features(*args)
        assigned=np.zeros((n,a))
        def state(i):
            return np.concatenate((context,[i/n],assigned.ravel()))
        x=state(0) if self.config.kind=='dqn' else context
        outputs=a if self.config.kind=='dqn' else n*a
        if self.agent.weights is None:
            self.agent._initialize(len(x),outputs)
        if self.agent.weights[0].shape[0]!=len(x) or self.agent.weights[-1].size!=outputs:
            raise ValueError('Learner observation/action topology changed')
        transitions=[];candidates=[]
        if self.config.kind=='dqn':
            for i in range(n):
                s=state(i)
                action=int(self.agent.rng.integers(a)) if training and self.agent.rng.random()<self.agent.epsilon else int(self.agent._forward(s,self.agent.weights)[0].argmax())
                assigned[i,action]=1
                transitions.append((s,action,0.,state(i+1) if i+1<n else np.zeros_like(s),i+1==n))
            candidates=[assigned.argmax(axis=1)-1]
        else:
            logits=self.agent._forward(context,self.agent.weights)[0].reshape(n,a)
            p=np.exp(logits-logits.max(axis=1,keepdims=True));p/=p.sum(axis=1,keepdims=True)
            candidates=[np.zeros(n,dtype=int),np.argmin(latency_lower_bounds(args,self.allocator.net),axis=1),p.argmax(axis=1)-1]
            candidates.extend(np.array([self.agent.rng.choice(a,p=row) for row in p])-1 for _ in range(max(0,self.config.candidates-3)))
        best=None;calls=0;unresolved=0;evaluations=0
        for targets in candidates:
            if best is not None and time.perf_counter()>=deadline:
                break
            solution=self.allocator.repair(args,targets,deadline)
            evaluations+=1;calls+=solution.diagnostics.get('allocator_calls',1)
            unresolved+=solution.diagnostics.get('unresolved_calls',0)
            if best is None or solution.rank<best.rank:
                best=solution
        if not best.feasible:
            raise RuntimeError('Learner allocation failed independent validation')
        inference=time.perf_counter()-start
        if training:
            if self.config.kind=='dqn':
                s,action,_,next_s,done=transitions[-1]
                transitions[-1]=(s,action,utility(best),next_s,done)
                self.agent.memory.extend(transitions)
                self.agent.learn()
                self.agent.epsilon=max(self.agent.config.epsilon_min,self.agent.epsilon*self.config.epsilon_decay)
            else:
                label=np.eye(a)[best.targets+1]
                self.replay.append((context.copy(),label))
                self._categorical_learn()
        elapsed=time.perf_counter()-start
        mask=hard_qos_mask(args,self.allocator.config,self.allocator.net)
        elapsed=time.perf_counter()-start
        return JointSearchOutput(best,elapsed,[best.rank],dict(assignments_evaluated=evaluations,
            allocator_calls=calls,unresolved_allocator_calls=unresolved,certified_fixed_assignments=int(best.certified),
            selected_fixed_assignment_certified=best.certified,selected_certificate_gap=best.certificate_gap,
            selected_solver_status=best.status,pruned_target_fraction=float(1-mask.mean()),
            individual_admission_upper_bound=int(mask.any(axis=1).sum()),
            reserved_admission_upper_bound_reached=bool(best.admitted==mask.any(axis=1).sum()),
            budget_overrun=elapsed*1000>self.config.budget_ms,local_only_domain=bool(not mask[:,1:].any()),
            search_executed=self.config.kind=='categorical',inference_allocation_ms=1000*inference,
            learning_ms=1000*(elapsed-inference),learning_updates=self.agent.updates if self.config.kind=='dqn' else self.updates))
