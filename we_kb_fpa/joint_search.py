"""Matched categorical FPA/GA search with shared hard-QoS admission and allocation."""
from dataclasses import dataclass
import time
import numpy as np
from we_kb_fpa.config import NetworkConfig
from we_kb_fpa.joint_allocation import FixedPlacementAllocator,JointAllocationConfig,hard_qos_mask,latency_lower_bounds
from we_kb_fpa.joint_memory import JointStateMemory
from we_kb_fpa.we_kb_fpa import WEKBFPAOptimizer


@dataclass(frozen=True)
class JointSearchConfig:
    strategy: str = 'fpa'
    budget_ms: float = 50.
    population: int = 8
    max_evaluations: int = 24
    use_pruning: bool = True
    use_knowledge: bool = True

    def __post_init__(self):
        if (self.strategy not in ('fpa','ga','anchor','greedy') or not np.isfinite(self.budget_ms) or self.budget_ms<=0 or
                self.population<2 or self.max_evaluations<2):
            raise ValueError('Invalid joint search configuration')


@dataclass
class JointSearchOutput:
    solution: object
    wall_clock_time_sec: float
    convergence: list
    diagnostics: dict


class JointOffloadingOptimizer:
    def __init__(self,config=None,allocation_config=None,net=None,cost=None,seed=42):
        self.config=config or JointSearchConfig()
        self.net=net or NetworkConfig(num_users=8,num_edge_servers=3)
        allocation_config=allocation_config or JointAllocationConfig(decision_reserve_sec=self.config.budget_ms/1000)
        if allocation_config.decision_reserve_sec<self.config.budget_ms/1000:
            raise ValueError('Every method must reserve the full common decision budget')
        self.allocator=FixedPlacementAllocator(allocation_config,cost,self.net)
        self.rng=np.random.default_rng(seed)
        self.guide=WEKBFPAOptimizer(net_config=self.net,seed=seed)
        self.previous=None
        self.knowledge=JointStateMemory() if self.config.use_knowledge else None
        self.policy=None

    def optimize(self,*args):
        started=time.perf_counter()
        cfg=self.config
        deadline=started+cfg.budget_ms/1000
        tasks,users,edges,cloud,sinr,distances=args
        n,m=len(tasks),len(edges)
        mask=hard_qos_mask(args,self.allocator.config,self.net)
        lower=latency_lower_bounds(args,self.net)
        options=[np.r_[-1,np.flatnonzero(mask[i])].astype(np.int32) if cfg.use_pruning else
                 np.arange(-1,m+2,dtype=np.int32) for i in range(n)]
        cache={}
        calls=0;unresolved=0;certified=0;evaluations=0;longest=.0005
        best=None
        curve=[]

        def evaluate(targets):
            nonlocal best,calls,unresolved,certified,evaluations,longest
            targets=np.asarray(targets,dtype=np.int32)
            key=targets.tobytes()
            if key in cache:
                return cache[key]
            tick=time.perf_counter()
            solution=self.allocator.repair(args,targets,deadline)
            if not solution.feasible:
                raise RuntimeError('A joint search returned a physically infeasible allocation')
            evaluations+=1
            calls+=solution.diagnostics.get('allocator_calls',1)
            unresolved+=solution.diagnostics.get('unresolved_calls',0)
            certified+=int(solution.certified)
            if best is None or solution.rank<best.rank:
                best=solution
            cache[key]=solution
            curve.append(best.rank)
            longest=max(longest,time.perf_counter()-tick)
            return solution

        def remaining():
            return evaluations<cfg.max_evaluations and time.perf_counter()+1.2*longest<deadline

        local=np.zeros(n,dtype=np.int32)
        evaluate(local)
        greedy=np.argmin(lower,axis=1).astype(np.int32)
        greedy[~mask.any(axis=1)]=-1
        anchors=[greedy]
        if self.previous is not None and cfg.strategy!='greedy':
            anchors.append(self.previous.copy())
        if self.knowledge is not None and self.knowledge.records and cfg.strategy!='greedy' and remaining():
            anchors.extend(self.knowledge.query(args,k=3))
        if self.policy is not None and remaining():
            anchors.append(self.policy.propose(args))
        population=[best.targets.copy()]
        fitness=[best.rank]
        for anchor in anchors:
            if remaining():
                solution=evaluate(anchor)
                population.append(solution.targets.copy());fitness.append(solution.rank)
        local_only=all(not mask[i,1:].any() for i in range(n))
        search=cfg.strategy in ('fpa','ga') and not (cfg.use_pruning and local_only)
        while search and len(population)<cfg.population and remaining():
            candidate=best.targets.copy()
            for i in range(n):
                if self.rng.random()<.3:
                    candidate[i]=self.rng.choice(options[i])
            solution=evaluate(candidate)
            population.append(solution.targets.copy());fitness.append(solution.rank)
        variance=float(np.var(np.log1p(sinr)))
        switch=self.guide._compute_dynamic_switch_prob(variance)
        attempts=0
        while search and remaining() and attempts<10*cfg.max_evaluations:
            attempts+=1
            index=int(self.rng.integers(len(population)))
            candidate=population[index].copy()
            if cfg.strategy=='ga':
                parents=[]
                for _ in range(2):
                    tournament=self.rng.integers(len(population),size=3)
                    parents.append(population[min(tournament,key=lambda j:fitness[j])])
                candidate=np.where(self.rng.random(n)<.5,parents[0],parents[1]).astype(np.int32)
            elif self.rng.random()<switch:
                radio_targets=candidate.copy()
                cloud_users=radio_targets==m+1
                radio_targets[cloud_users]=np.argmin(distances[cloud_users],axis=1)+1
                scales=self.guide._compute_channel_guided_step_sizes(sinr,radio_targets)
                copy_mask=self.rng.random(n)<np.clip(scales*np.abs(self.guide._sample_mantegna_levy(n)),.05,.95)
                candidate[copy_mask]=best.targets[copy_mask]
            else:
                donor=population[int(self.rng.integers(len(population)))]
                copy_mask=self.rng.random(n)<.5
                candidate[copy_mask]=donor[copy_mask]
            mutable=[i for i in range(n) if np.any(options[i]!=candidate[i])]
            if not mutable:
                break
            changed=int(self.rng.choice(mutable))
            candidate[changed]=self.rng.choice(options[changed][options[changed]!=candidate[changed]])
            solution=evaluate(candidate)
            if solution.rank<fitness[index]:
                population[index]=solution.targets.copy();fitness[index]=solution.rank
        self.previous=best.targets.copy()
        if self.knowledge is not None:
            self.knowledge.add(args,best)
        elapsed=time.perf_counter()-started
        return JointSearchOutput(best,elapsed,curve,dict(assignments_evaluated=evaluations,
            individual_admission_upper_bound=int(mask.any(axis=1).sum()),
            reserved_admission_upper_bound_reached=bool(best.admitted==mask.any(axis=1).sum()),
            allocator_calls=calls,unresolved_allocator_calls=unresolved,
            certified_fixed_assignments=certified,selected_fixed_assignment_certified=best.certified,
            selected_certificate_gap=best.certificate_gap,selected_solver_status=best.status,
            pruned_target_fraction=float(1-mask.mean()),budget_overrun=elapsed*1000>cfg.budget_ms,
            local_only_domain=local_only,search_executed=bool(search)))


def offered_task_metrics(output,tasks,reserve_sec):
    solution=output.solution
    n=len(tasks)
    dropped=n-solution.admitted
    actual_late=0
    metrics=[]
    if solution.evaluation is not None:
        metrics=solution.evaluation.task_metrics
        actual_late=sum(metric.latency_total+output.wall_clock_time_sec>metric.deadline_sec for metric in metrics)
    energy=sum(metric.energy_total for metric in metrics)
    return dict(offered_tasks=n,admitted_tasks=solution.admitted,dropped_tasks=dropped,
                admission_pct=100*solution.admitted/n,service_cost=solution.service_cost,
                service_cost_per_admitted=solution.service_cost/solution.admitted if solution.admitted else None,
                offered_miss_pct=100*(dropped+actual_late)/n,
                admitted_late_after_decision=actual_late,
                decision_reservation_overrun=output.wall_clock_time_sec>reserve_sec,
                runtime_ms=1000*output.wall_clock_time_sec,
                mean_cpu_ghz=float(np.mean(solution.cpu[solution.targets>0])/1e9) if np.any(solution.targets>0) else 0.,
                mean_radio_fraction=float(np.mean(solution.bandwidth[solution.targets>0])) if np.any(solution.targets>0) else 0.,
                total_admitted_energy_j=energy,
                mean_admitted_energy_j=energy/solution.admitted if solution.admitted else None,
                mean_admitted_service_latency_ms=float(np.mean([metric.latency_total for metric in metrics])*1000) if metrics else None,
                mean_admitted_completion_latency_ms=float(np.mean([metric.latency_total+output.wall_clock_time_sec for metric in metrics])*1000) if metrics else None,
                minimum_reserved_deadline_margin_ms=min((metric.deadline_sec-metric.latency_total-reserve_sec)*1000 for metric in metrics) if metrics else None)
