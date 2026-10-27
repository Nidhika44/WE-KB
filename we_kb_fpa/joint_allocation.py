"""Hard-QoS admission and convex CPU/radio allocation for fixed placements.

For remote task i, latency is A_i/b_i+C_i/u_i+D_i and energy is
p_i*A_i/b_i+K_i*u_i**2. CPU shares u and radio shares b have separate
per-host/access-station capacities. The objective and deadline sublevels
are convex on the positive resource box. A tangent Lagrangian bound
certifies only this fixed-placement problem, never the discrete search.
"""
from dataclasses import dataclass,field
import time
import numpy as np
from scipy.optimize import minimize,nnls
from we_kb_fpa.config import CostConfig,NetworkConfig
from we_kb_fpa.cost_engine import CostEngine


@dataclass(frozen=True)
class JointAllocationConfig:
    decision_reserve_sec: float = .05
    max_iterations: int = 60
    feasibility_tolerance: float = 1e-7
    certificate_relative_tolerance: float = 1e-4
    capacity_margin: float = 1e-6

    def __post_init__(self):
        if (not np.isfinite(self.decision_reserve_sec) or self.decision_reserve_sec<0 or
                self.max_iterations<1 or self.feasibility_tolerance<=0 or self.certificate_relative_tolerance<=0 or
                not 0<self.capacity_margin<1e-3):
            raise ValueError('Invalid joint allocator configuration')


@dataclass
class JointSolution:
    targets: np.ndarray
    cpu: np.ndarray
    bandwidth: np.ndarray
    admitted: int
    service_cost: float
    feasible: bool
    status: str
    certified: bool = False
    lower_bound: float | None = None
    certificate_gap: float | None = None
    evaluation: object = None
    diagnostics: dict = field(default_factory=dict)

    @property
    def rank(self):
        return (-self.admitted,self.service_cost)


def latency_lower_bounds(args,net=None):
    net=net or NetworkConfig()
    tasks,users,edges,cloud,sinr,distances=args
    n,m=len(tasks),len(edges)
    bounds=np.full((n,m+2),np.inf)
    for i,(task,user) in enumerate(zip(tasks,users)):
        bounds[i,0]=task.total_cycles/user.f_local
        for j,edge in enumerate(edges):
            rate=edge.bandwidth_total*np.log1p(sinr[i,j])/np.log(2.)
            if rate>=1e3 and edge.f_total>=1e6:
                bounds[i,j+1]=(task.data_size_bits/rate+task.total_cycles/edge.f_total+
                               edge.queue_delay_current+distances[i,j]/3e8+net.d_dl_constant)
        access=int(np.argmin(distances[i]))
        rate=edges[access].bandwidth_total*np.log1p(sinr[i,access])/np.log(2.)
        if rate>=1e3 and cloud.f_total>=1e6:
            bounds[i,m+1]=(task.data_size_bits/rate+task.data_size_bits/cloud.backhaul_bandwidth+
                           task.total_cycles/cloud.f_total+cloud.queue_delay_current+cloud.propagation_delay+net.d_dl_constant)
    return bounds


def hard_qos_mask(args,config=None,net=None):
    config=config or JointAllocationConfig()
    limits=np.array([task.deadline_sec for task in args[0]])-config.decision_reserve_sec
    return (latency_lower_bounds(args,net)<=limits[:,None])&(limits[:,None]>0)


def tangent_dual_bound(x,objective,gradient,constraints,jacobian,lower,upper):
    """Any nonnegative multiplier gives a valid convex tangent lower bound."""
    values=np.asarray(constraints(x))
    derivative=np.asarray(gradient(x))
    jac=np.asarray(jacobian(x))
    active=np.flatnonzero(values>=-1e-5)
    columns=[jac[index] for index in active]
    for j in range(len(x)):
        if x[j]-lower[j]<=1e-6:
            column=np.zeros(len(x));column[j]=-1.;columns.append(column)
        if upper[j]-x[j]<=1e-6:
            column=np.zeros(len(x));column[j]=1.;columns.append(column)
    multipliers=np.zeros(len(values))
    if columns:
        try:
            fitted,_=nnls(np.column_stack(columns),-derivative,maxiter=1000)
            multipliers[active]=fitted[:len(active)]
        except (RuntimeError,ValueError):
            pass
    lagrangian=objective(x)+multipliers@values
    slope=derivative+jac.T@multipliers
    extremum=np.where(slope>=0,lower,upper)
    return float(lagrangian+slope@(extremum-x))


class _Expired(Exception):
    pass


class FixedPlacementAllocator:
    def __init__(self,config=None,cost=None,net=None):
        self.config=config or JointAllocationConfig()
        self.cost=cost or CostConfig()
        self.net=net or NetworkConfig()
        if not 0<=self.cost.alpha<=1:
            raise ValueError('Convex allocation requires alpha in [0,1]')
        if min(self.net.kappa_local,self.net.kappa_edge,self.net.kappa_cloud)<0:
            raise ValueError('Energy coefficients must be nonnegative')
        self.engine=CostEngine(self.cost,self.net)

    def problem(self,args,targets):
        tasks,users,edges,cloud,sinr,distances=args
        targets=np.asarray(targets,dtype=np.int32)
        n,m=len(tasks),len(edges)
        if (len(users)!=n or sinr.shape!=(n,m) or distances.shape!=(n,m) or
                not np.isfinite(sinr).all() or np.any(sinr<0) or not np.isfinite(distances).all() or np.any(distances<0) or
                any(task.deadline_sec<=0 or task.total_cycles<=0 or task.data_size_bits<=0 for task in tasks) or
                any(user.f_local<=0 or user.p_tx<0 for user in users) or
                any(edge.f_total<1e6 or edge.bandwidth_total<=0 or edge.queue_delay_current<0 for edge in edges) or
                cloud.f_total<1e6 or cloud.backhaul_bandwidth<=0 or cloud.queue_delay_current<0):
            raise ValueError('Invalid physical inputs for joint allocation')
        if targets.shape!=(n,) or np.any((targets < -1)|(targets>m+1)):
            raise ValueError('Targets must be drop=-1, local=0, edge=1..M, cloud=M+1')
        mask=hard_qos_mask(args,self.config,self.net)
        admitted=np.flatnonzero(targets>=0)
        if any(not mask[i,targets[i]] for i in admitted):
            return None,'PROVEN_INFEASIBLE_INDIVIDUAL'
        remote=np.flatnonzero(targets>0)
        local=np.flatnonzero(targets==0)
        constant=0.
        for i in local:
            latency,energy=self.engine.compute_local_metrics(tasks[i],users[i])
            constant+=self.cost.alpha*latency/tasks[i].deadline_sec+(1-self.cost.alpha)*energy/max(energy,1e-6)
        if not len(remote):
            return dict(targets=targets,remote=remote,constant=constant),'LOCAL_OR_EMPTY'
        a=[];c=[];d=[];k=[];power=[];norm=[];deadlines=[];limits=[];capacities=[];accesses=[]
        for i in remote:
            target=targets[i]
            host=edges[target-1] if target<=m else cloud
            access=target-1 if target<=m else int(np.argmin(distances[i]))
            rate=edges[access].bandwidth_total*np.log1p(sinr[i,access])/np.log(2.)
            _,energy=self.engine.compute_local_metrics(tasks[i],users[i])
            fixed=host.queue_delay_current+self.net.d_dl_constant
            if target<=m:
                fixed+=distances[i,access]/3e8
                kappa=self.net.kappa_edge
            else:
                fixed+=cloud.propagation_delay+tasks[i].data_size_bits/cloud.backhaul_bandwidth
                kappa=self.net.kappa_cloud
            a.append(tasks[i].data_size_bits/rate);c.append(tasks[i].total_cycles/host.f_total)
            d.append(fixed);k.append(kappa*tasks[i].total_cycles*host.f_total**2)
            power.append(users[i].p_tx);norm.append(max(energy,1e-6));deadlines.append(tasks[i].deadline_sec)
            limits.append((tasks[i].deadline_sec-self.config.decision_reserve_sec)*(1-1e-7))
            capacities.append(host.f_total);accesses.append(access)
        a,c,d,k,power,norm,deadlines,limits,capacities=map(np.asarray,(a,c,d,k,power,norm,deadlines,limits,capacities))
        count=len(remote);cap=1-self.config.capacity_margin
        lower_cpu=np.maximum(1e6/capacities,c/np.maximum(limits-d-a,1e-30))
        rates=np.array([edges[j].bandwidth_total*np.log1p(sinr[i,j])/np.log(2.) for i,j in zip(remote,accesses)])
        lower_radio=np.maximum(1e3/rates,a/np.maximum(limits-d-c,1e-30))
        lower=np.r_[lower_cpu,lower_radio];upper=np.full(2*count,cap)
        if np.any(lower>upper):
            return None,'PROVEN_INFEASIBLE_RESOURCE_BOUNDS'
        groups=[]
        for target in np.unique(targets[remote]):
            groups.append(np.flatnonzero(targets[remote]==target))
        for access in np.unique(accesses):
            groups.append(count+np.flatnonzero(np.asarray(accesses)==access))
        if any(lower[group].sum()>cap for group in groups):
            return None,'PROVEN_INFEASIBLE_CAPACITY_BOUND'
        x=lower.copy()
        for group in groups:
            x[group]+=(cap-x[group].sum())/len(group)
        weights_cpu=self.cost.alpha*c/deadlines
        weights_radio=self.cost.alpha*a/deadlines+(1-self.cost.alpha)*power*a/norm
        weights_energy=(1-self.cost.alpha)*k/norm
        def latency(z):
            return a/z[count:]+c/z[:count]+d
        def objective(z):
            return float(constant+np.sum(weights_cpu/z[:count]+weights_radio/z[count:]+weights_energy*z[:count]**2+self.cost.alpha*d/deadlines))
        def gradient(z):
            return np.r_[-weights_cpu/z[:count]**2+2*weights_energy*z[:count],-weights_radio/z[count:]**2]
        def constraints(z):
            return np.r_[latency(z)/limits-1.,[z[group].sum()-cap for group in groups]]
        def jacobian(z):
            result=np.zeros((count+len(groups),2*count))
            indices=np.arange(count)
            result[indices,indices]=-c/z[:count]**2/limits
            result[indices,count+indices]=-a/z[count:]**2/limits
            for j,group in enumerate(groups):
                result[count+j,group]=1.
            return result
        return dict(targets=targets,remote=remote,constant=constant,x=x,lower=lower,upper=upper,
                    objective=objective,gradient=gradient,constraints=constraints,jacobian=jacobian,
                    capacities=capacities,latency=latency,count=count),'READY'

    def solve(self,args,targets,deadline=None):
        problem,status=self.problem(args,targets)
        targets=np.asarray(targets,dtype=np.int32).copy()
        empty=JointSolution(targets,np.zeros(len(targets)),np.zeros(len(targets)),int((targets>=0).sum()),float('inf'),False,status)
        if problem is None:
            return empty
        if status=='LOCAL_OR_EMPTY':
            return self._score(args,targets,np.zeros(len(targets)),np.zeros(len(targets)),True,problem['constant'],0.,'FIXED_PLACEMENT_CERTIFIED')
        p=problem;lower=p['lower'];upper=p['upper'];x=p['x'];tol=self.config.feasibility_tolerance
        def expired(_=None):
            if deadline is not None and time.perf_counter()>=deadline:
                raise _Expired
        def feasible(z):
            return np.all(z>=lower-tol) and np.all(z<=upper+tol) and np.max(p['constraints'](z))<=tol
        try:
            expired()
            if not feasible(x):
                slack=max(0.,float(np.max(p['constraints'](x)[:p['count']])))+.01
                v0=np.r_[x,slack]
                def phase_constraints(v):
                    values=p['constraints'](v[:-1]).copy()
                    values[:p['count']]-=v[-1]
                    return values
                def phase_jac(v):
                    result=np.c_[p['jacobian'](v[:-1]),np.zeros(len(p['constraints'](v[:-1])))]
                    result[:p['count'],-1]=-1.
                    return result
                phase_gradient=np.r_[np.zeros(len(x)),1.]
                phase=minimize(lambda v:v[-1],v0,jac=lambda v:phase_gradient,method='SLSQP',
                               bounds=list(zip(np.r_[lower,0.],np.r_[upper,slack])),
                               constraints=[dict(type='ineq',fun=lambda v:-phase_constraints(v),jac=lambda v:-phase_jac(v))],
                               callback=expired,options=dict(maxiter=self.config.max_iterations,ftol=1e-10))
                x=phase.x[:-1]
                if not feasible(x):
                    bound=tangent_dual_bound(phase.x,lambda v:v[-1],lambda v:phase_gradient,
                                              phase_constraints,phase_jac,np.r_[lower,0.],np.r_[upper,slack])
                    empty.status='PROVEN_INFEASIBLE_PHASE_I' if bound>1e-6 else 'UNRESOLVED_FEASIBILITY'
                    empty.diagnostics=dict(phase_one_lower_bound=bound)
                    return empty
            seed=x.copy()
            feasible_points=[seed]
            def callback(z):
                if feasible(z):
                    feasible_points.append(z.copy())
                expired()
            try:
                optimized=minimize(p['objective'],x,jac=p['gradient'],method='SLSQP',bounds=list(zip(lower,upper)),
                                   constraints=[dict(type='ineq',fun=lambda z:-p['constraints'](z),jac=lambda z:-p['jacobian'](z))],
                                   callback=callback,options=dict(maxiter=self.config.max_iterations,ftol=1e-10))
                if feasible(optimized.x):
                    feasible_points.append(optimized.x)
            except _Expired:
                pass
            x=min(feasible_points,key=p['objective'])
            x=np.clip(x,lower,upper)
            value=p['objective'](x)
            bound=tangent_dual_bound(x,p['objective'],p['gradient'],p['constraints'],p['jacobian'],lower,upper)
            gap=value-bound
            certified=bool(-tol<=gap<=self.config.certificate_relative_tolerance*max(1.,abs(value)))
            cpu=np.zeros(len(targets));bw=np.zeros(len(targets))
            cpu[p['remote']]=x[:p['count']]*p['capacities'];bw[p['remote']]=x[p['count']:]
            return self._score(args,targets,cpu,bw,certified,bound,gap,
                               'FIXED_PLACEMENT_CERTIFIED' if certified else 'FEASIBLE_UNCERTIFIED')
        except _Expired:
            empty.status='TIME_LIMIT_UNRESOLVED'
            return empty

    def _score(self,args,targets,cpu,bw,certified,bound,gap,status):
        tasks,users,edges,cloud,sinr,distances=args
        cpu=cpu.copy();bw=bw.copy()
        for i in np.flatnonzero(targets==0):
            cpu[i]=users[i].f_local
        admitted=np.flatnonzero(targets>=0)
        if not len(admitted):
            return JointSolution(targets,cpu,bw,0,0.,True,'EMPTY_ADMISSION',True,0.,0.)
        evaluation=self.engine.evaluate_allocation([tasks[i] for i in admitted],[users[i] for i in admitted],edges,cloud,
            targets[admitted],cpu[admitted],bw[admitted],sinr[admitted],distances[admitted])
        limits=np.array([tasks[i].deadline_sec-self.config.decision_reserve_sec for i in admitted])
        actual=np.array([metric.latency_total for metric in evaluation.task_metrics])
        valid=bool(np.all(actual<=limits+1e-9) and evaluation.total_penalties<1e-7)
        model_difference=evaluation.total_cost-(bound+gap)
        consistent=abs(model_difference)<=1e-6*max(1.,abs(evaluation.total_cost))
        checked_status=status if consistent else 'FEASIBLE_OBJECTIVE_MISMATCH'
        return JointSolution(targets,cpu,bw,len(admitted),evaluation.total_cost,valid,checked_status if valid else 'FAILED_INDEPENDENT_VALIDATION',
                             certified and valid and consistent,bound,gap,evaluation,dict(model_objective_difference=model_difference))

    def repair(self,args,requested,deadline=None):
        targets=np.asarray(requested,dtype=np.int32).copy()
        mask=hard_qos_mask(args,self.config,self.net)
        individual_drops=0
        for i in range(len(targets)):
            if targets[i]>=0 and not mask[i,targets[i]]:
                targets[i]=-1;individual_drops+=1
        calls=0;unknown=0
        while True:
            solved=self.solve(args,targets,deadline)
            calls+=1
            if solved.feasible:
                solved.diagnostics.update(allocator_calls=calls,unresolved_calls=unknown,
                                           individually_infeasible_drops=individual_drops,
                                           repair_drops=int((np.asarray(requested)>=0).sum()-solved.admitted))
                return solved
            unknown+=int(not solved.status.startswith('PROVEN_INFEASIBLE'))
            remote=np.flatnonzero(targets>0)
            if not len(remote):
                raise RuntimeError('Local/drop fallback failed independent validation')
            if deadline is not None and time.perf_counter()>=deadline:
                targets[remote]=-1
            else:
                lower=latency_lower_bounds(args,self.net)
                limits=np.array([task.deadline_sec-self.config.decision_reserve_sec for task in args[0]])
                victim=max(remote,key=lambda i:lower[i,targets[i]]/max(1e-12,limits[i]))
                targets[victim]=-1
