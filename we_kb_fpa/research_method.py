"""Evaluation-matched extensions of the original continuous 3N search family."""
from dataclasses import dataclass
import pickle
import time
import numpy as np
from we_kb_fpa.baselines import BaselineRunner
from we_kb_fpa.config import OptimizerConfig,NetworkConfig
from we_kb_fpa.knowledge_base import WirelessEdgeKnowledgeBase
from we_kb_fpa.we_kb_fpa import WEKBFPAOptimizer,OptimizationOutput


@dataclass(frozen=True)
class ResearchMethodConfig:
    family: str='fpa'
    evaluations: int=620
    population: int=20
    pruner: bool=True
    phi: bool=True
    switch: bool=True
    kb: bool=True
    phi_direction: int=1
    switch_direction: int=1
    random_anchor: bool=False
    gamma_min: float=-20.
    gamma_max: float=30.
    variance_min: float=0.
    variance_max: float=4.

    def __post_init__(self):
        if (self.family not in ('fpa','pso','ga') or self.evaluations<2 or self.population<2 or
            self.phi_direction not in (-1,1) or self.switch_direction not in (-1,1) or
            self.gamma_min>=self.gamma_max or self.variance_min>=self.variance_max):
            raise ValueError('Invalid research search settings')


def optimistic_latency(args,net=None):
    """Necessary latency bounds for the legacy simulator, including its rate/CPU floors."""
    net=net or NetworkConfig()
    tasks,users,edges,cloud,sinr,distances=args
    result=np.zeros((len(tasks),len(edges)+2))
    for i,(task,user) in enumerate(zip(tasks,users)):
        result[i,0]=task.total_cycles/user.f_local
        for j,edge in enumerate(edges):
            result[i,j+1]=(task.data_size_bits/max(1e3,edge.bandwidth_total*np.log2(1+sinr[i,j]))+
                task.total_cycles/max(1e6,edge.f_total)+edge.queue_delay_current+distances[i,j]/3e8+net.d_dl_constant)
        j=int(np.argmin(distances[i]))
        result[i,-1]=(task.data_size_bits/max(1e3,edges[j].bandwidth_total*np.log2(1+sinr[i,j]))+
            task.data_size_bits/cloud.backhaul_bandwidth+task.total_cycles/max(1e6,cloud.f_total)+
            cloud.queue_delay_current+cloud.propagation_delay+net.d_dl_constant)
    return result


class ResearchMemory:
    def __init__(self,capacity=100):
        if capacity<0:
            raise ValueError('KB capacity cannot be negative')
        self.capacity=capacity;self.records=[]
        self.extractor=WirelessEdgeKnowledgeBase()

    @staticmethod
    def topology(args):
        return (tuple(user.device_id for user in args[1]),tuple(edge.server_id for edge in args[2]))

    def feature(self,args):
        return self.extractor.extract_state_feature_vector(args[4],args[0],args[5])

    def add(self,args,chromosome):
        if not self.capacity:
            return
        self.records.append((self.topology(args),self.feature(args),np.asarray(chromosome,dtype=float).copy()))
        if len(self.records)>self.capacity:
            self.records.pop(0)

    def query(self,args,k,rng,random_anchor=False):
        eligible=[record for record in self.records if record[0]==self.topology(args)]
        if not eligible:
            return []
        if random_anchor:
            indices=rng.choice(len(eligible),min(k,len(eligible)),replace=False)
        else:
            feature=self.feature(args)
            indices=np.argsort([np.linalg.norm(record[1]-feature) for record in eligible],kind='stable')[:k]
        return [eligible[i][2].copy() for i in indices]

    def bytes_per_entry(self):
        return float(np.mean([len(pickle.dumps(record,protocol=5)) for record in self.records])) if self.records else 0.


class ResearchOptimizer:
    def __init__(self,config=None,net=None,seed=42):
        self.config=config or ResearchMethodConfig();self.net=net or NetworkConfig()
        self.rng=np.random.default_rng(seed)
        self.base=WEKBFPAOptimizer(OptimizerConfig(population_size=self.config.population),net_config=self.net,use_kb=False,seed=seed)
        self.runner=BaselineRunner(net_config=self.net)

    def adaptation(self,args,chrom):
        c=self.config;n=len(args[0]);m=len(args[2])
        target=np.clip(np.rint(chrom[:n]).astype(int),0,m+1)
        home=np.argmin(args[5],axis=1)
        selected=np.where(target==m+1,home,np.clip(target-1,0,m-1))
        db=10*np.log10(np.maximum(args[4][np.arange(n),selected],1e-12))
        db[target==0]=10.
        quality=np.clip((db-c.gamma_min)/(c.gamma_max-c.gamma_min),0,1)
        if c.phi_direction<0:
            quality=1-quality
        phi=.05*(1+1.5*(1-quality)) if c.phi else np.full(n,.05)
        variance=float(np.var(np.log1p(args[4])))
        v=np.clip((variance-c.variance_min)/(c.variance_max-c.variance_min),0,1)
        if c.switch_direction<0:
            v=1-v
        switch=float(.2+.6*v) if c.switch else .8
        return phi,switch

    def optimize(self,args,memory=None):
        start=time.perf_counter();c=self.config;n=len(args[0]);m=len(args[2])
        lower=optimistic_latency(args,self.net)
        mask=lower<=np.array([task.deadline_sec for task in args[0]])[:,None]
        self.base.current_distances=args[5];self.base.allocation_mode='joint'
        greedy=self.runner.run_greedy_latency(*args)
        greedy_chrom=np.r_[greedy.best_targets,np.ones(2*n)]
        def repair(chrom,prune=True):
            x=np.asarray(chrom,dtype=float).copy();x[:n]=np.clip(x[:n],0,m+1);x[n:]=np.clip(x[n:],.01,1.)
            if c.pruner and prune:
                for i,target in enumerate(np.rint(x[:n]).astype(int)):
                    if not mask[i,target] and mask[i].any():
                        choices=np.flatnonzero(mask[i]);x[i]=choices[np.argmin(np.abs(choices-target))]
            return x
        evaluated=0;curve=[];best=None;best_chrom=None;best_eval=None
        def evaluate(chrom,prune=True,known=None):
            nonlocal evaluated,best,best_chrom,best_eval
            chrom=repair(chrom,prune)
            allocation=self.base.decode_chromosome(chrom,*args[:4],distances=args[5])
            result=known or self.base.cost_engine.evaluate_allocation(*args[:4],*allocation,*args[4:])
            evaluated+=1
            if best is None or result.total_cost<best:
                best=result.total_cost;best_chrom=chrom.copy();best_eval=result
            curve.append(best)
            return chrom,result.total_cost
        # The unpruned, resource-repaired greedy guard also consumes one evaluation.
        evaluate(greedy_chrom,False,greedy.evaluation_result)
        population=[evaluate(greedy_chrom)[0]]
        retrieval=time.perf_counter()
        anchors=memory.query(args,max(1,int(c.population*.3)),self.rng,c.random_anchor) if c.kb and memory else []
        retrieval=time.perf_counter()-retrieval
        if anchors:
            population.extend(anchors)
            population.extend(anchor+self.rng.normal(0,.03,3*n) for anchor in anchors[:1])
        while len(population)<c.population:
            population.append(np.r_[self.rng.uniform(0,m+1,n),self.rng.uniform(.1,1.,2*n)])
        pop=[];fitness=[]
        for chrom in population[:c.population]:
            if evaluated>=c.evaluations:
                break
            x,cost=evaluate(chrom);pop.append(x);fitness.append(cost)
        if not pop:
            pop=[best_chrom.copy()];fitness=[best]
        pop=np.asarray(pop);fitness=np.asarray(fitness);pbest=pop.copy();pfit=fitness.copy();velocity=np.zeros_like(pop)
        index=0
        while evaluated<c.evaluations:
            j=index%len(pop);index+=1
            if c.family=='pso':
                velocity[j]=.7*velocity[j]+1.5*self.rng.random(3*n)*(pbest[j]-pop[j])+1.5*self.rng.random(3*n)*(best_chrom-pop[j])
                limits=np.r_[np.full(n,.2*(m+1)),np.full(2*n,.2)]
                velocity[j]=np.clip(velocity[j],-limits,limits);candidate=pop[j]+velocity[j]
            elif c.family=='ga':
                parents=[]
                for _ in range(2):
                    choices=self.rng.integers(len(pop),size=3);parents.append(pop[choices[np.argmin(fitness[choices])]])
                candidate=np.where(self.rng.random(3*n)<.5,*parents).copy()
                mutation=self.rng.random(3*n)<.1
                candidate[mutation]=np.r_[self.rng.uniform(0,m+1,n),self.rng.uniform(.01,1.,2*n)][mutation]
            else:
                phi,switch=self.adaptation(args,pop[j])
                if self.rng.random()<switch:
                    candidate=pop[j]+np.r_[phi,np.full(2*n,.05)]*self.base._sample_mantegna_levy(3*n)*(best_chrom-pop[j])
                else:
                    a,b=self.rng.integers(len(pop),size=2);candidate=pop[j]+self.rng.random()*(pop[a]-pop[b])
            x,cost=evaluate(candidate)
            if c.family=='pso' or cost<fitness[j]:
                pop[j]=x;fitness[j]=cost
            if cost<pfit[j]:
                pbest[j]=x;pfit[j]=cost
        allocation=self.base.decode_chromosome(best_chrom,*args[:4],distances=args[5])
        elapsed=time.perf_counter()-start
        output=OptimizationOutput(best,*allocation,curve,elapsed,len(curve)-1,best_eval,'Greedy-seeded WE-KB-FPA')
        output.diagnostics=dict(cost_evaluations=evaluated,retrieval_ms=1000*retrieval,
            kb_entries=len(memory.records) if c.kb and memory else 0,
            kb_bytes_per_entry=memory.bytes_per_entry() if c.kb and memory else 0.,
            compatible_anchors=len(anchors),pruning_rate=float(np.mean((~mask)&mask.any(axis=1)[:,None])) if c.pruner else 0.,
            optimistically_impossible_target_fraction=float(1-mask.mean()),
            unavoidable_rows=int((~mask.any(axis=1)).sum()),greedy_guard_cost=greedy.best_cost,
            greedy_dominance_verified=best<=greedy.best_cost+1e-8)
        output.best_chromosome=best_chrom
        return output


def research_metrics(output,args,net=None):
    metrics=output.evaluation_result.task_metrics;n=len(metrics);delay=output.wall_clock_time_sec
    feasible=[metric for metric in metrics if metric.latency_total<=metric.deadline_sec]
    delayed=[metric for metric in metrics if metric.latency_total+delay<=metric.deadline_sec]
    bounds=optimistic_latency(args,net);deadlines=np.array([task.deadline_sec for task in args[0]])
    return dict(search_objective=output.best_cost,feasible_cost=sum(metric.cost_j for metric in feasible),
        feasible_cost_per_task=float(np.mean([metric.cost_j for metric in feasible])) if feasible else None,
        delay_feasible_cost=sum(metric.cost_j for metric in delayed),
        offered_tasks=n,attempted_tasks=n,admitted_tasks=n,admission_pct=100.,
        deadline_feasible_tasks=len(feasible),deadline_feasible_pct=100*len(feasible)/n,
        violation_pct=100*(n-len(feasible))/n,delay_adjusted_violation_pct=100*(n-len(delayed))/n,
        physical_violation_lower_bound_pct=100*np.mean(bounds.min(axis=1)>deadlines),
        delay_physical_violation_lower_bound_pct=100*np.mean(bounds.min(axis=1)+delay>deadlines),
        runtime_ms=1000*delay,feasible_task_ids=';'.join(str(metric.task_id) for metric in feasible))
