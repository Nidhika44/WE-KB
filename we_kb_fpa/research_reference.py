"""Time-limited joint MIQCP reference with independent simulator rescoring."""
from dataclasses import dataclass,asdict
import importlib.util
import time
import numpy as np
from we_kb_fpa.config import CostConfig,NetworkConfig
from we_kb_fpa.cost_engine import CostEngine


def independent_cost_bound(args,net=None,cost=None):
    net=net or NetworkConfig();cost=cost or CostConfig();alpha=cost.alpha
    tasks,users,edges,cloud,sinr,distances=args;total=0.
    for i,(task,user) in enumerate(zip(tasks,users)):
        cycles=task.total_cycles;enorm=max(net.kappa_local*user.f_local**2*cycles,1e-6)
        values=[alpha*cycles/user.f_local/task.deadline_sec+(1-alpha)*net.kappa_local*user.f_local**2*cycles/enorm]
        for target,server in enumerate([*edges,cloud],1):
            j=target-1 if target<=len(edges) else int(np.argmin(distances[i]))
            tx=task.data_size_bits/max(1e3,edges[j].bandwidth_total*np.log2(1+sinr[i,j]))
            fixed=server.queue_delay_current+net.d_dl_constant
            kappa=net.kappa_edge if target<=len(edges) else net.kappa_cloud
            fixed+=distances[i,j]/3e8 if target<=len(edges) else cloud.propagation_delay+task.data_size_bits/cloud.backhaul_bandwidth
            if (1-alpha)*kappa>0:
                f=(alpha*enorm/(2*(1-alpha)*kappa*task.deadline_sec))**(1/3) if alpha else 1e6
            else:
                f=server.f_total
            f=np.clip(f,1e6,max(1e6,server.f_total))
            values.append(alpha*(tx+cycles/f+fixed)/task.deadline_sec+(1-alpha)*(user.p_tx*tx+kappa*f*f*cycles)/enorm)
        total+=min(values)
    return float(total)


@dataclass
class ReferenceReport:
    status: str
    runtime_sec: float
    lower_bound: float
    incumbent_cost: float|None=None
    reported_mip_gap: float|None=None
    is_optimal: bool=False
    simulator_match: bool=False
    solver_objective: float|None=None
    scope: str='Legacy joint penalized objective; continuous resource relaxation of bounded chromosome ratios'
    message: str=''


def solve_reference(args,net=None,time_limit=60.,mip_gap=.01,seed=42):
    if time_limit<=0 or not 0<mip_gap<1:
        raise ValueError('Invalid reference limits')
    net=net or NetworkConfig();cost=CostConfig();lower=independent_cost_bound(args,net,cost)
    if importlib.util.find_spec('gurobipy') is None:
        return ReferenceReport('UNAVAILABLE',0.,lower,message='gurobipy is not installed; analytical lower bound only')
    import gurobipy as gp
    start=time.perf_counter();model=None
    try:
        model=gp.Model('legacy_joint_reference')
        model.Params.OutputFlag=0;model.Params.NonConvex=2;model.Params.Threads=1
        model.Params.TimeLimit=time_limit;model.Params.MIPGap=mip_gap;model.Params.Seed=seed%2000000000
        tasks,users,edges,cloud,sinr,distances=args;n=len(tasks);m=len(edges)
        x=model.addVars(n,m+2,vtype=gp.GRB.BINARY,name='target')
        f=model.addVars(n,m+1,lb=0.,name='cpu_GHz')
        b=model.addVars(n,m+1,lb=0.,ub=1.,name='radio_fraction')
        host=[*edges,cloud];tx={};comp={};energy={};latency={};norm_energy={};penalty={}
        for i,(task,user) in enumerate(zip(tasks,users)):
            model.addConstr(gp.quicksum(x[i,j] for j in range(m+2))==1)
            parts=[];energies=[]
            for j,server in enumerate(host):
                access=j if j<m else int(np.argmin(distances[i]))
                rate=edges[access].bandwidth_total*np.log2(1+sinr[i,access])
                capacity=max(1e6,server.f_total)/1e9
                model.addConstr(f[i,j]<=capacity*x[i,j+1]);model.addConstr(b[i,j]<=x[i,j+1])
                # Preserve the original simulator's numerical floors exactly.
                fe=model.addVar(lb=.001,ub=capacity,name=f'effective_cpu_{i}_{j}')
                raw_rate=model.addVar(lb=0.,ub=max(0.,rate/1e6))
                re=model.addVar(lb=.001,ub=max(.001,rate/1e6))
                model.addGenConstrMax(fe,[f[i,j]],constant=.001)
                model.addConstr(raw_rate==b[i,j]*rate/1e6)
                model.addGenConstrMax(re,[raw_rate],constant=.001)
                max_tx=task.data_size_bits/1e3;max_comp=task.total_cycles/1e6
                tx[i,j]=model.addVar(lb=0.,ub=max_tx)
                comp[i,j]=model.addVar(lb=0.,ub=max_comp)
                model.addConstr(tx[i,j]<=max_tx*x[i,j+1]);model.addConstr(comp[i,j]<=max_comp*x[i,j+1])
                model.addQConstr(tx[i,j]*re>=task.data_size_bits/1e6*x[i,j+1])
                model.addQConstr(comp[i,j]*fe>=task.total_cycles/1e9*x[i,j+1])
                kappa=net.kappa_edge if j<m else net.kappa_cloud
                max_energy=kappa*(capacity*1e9)**2*task.total_cycles
                energy[i,j]=model.addVar(lb=0.,ub=max_energy)
                model.addQConstr(energy[i,j]>=kappa*1e18*task.total_cycles*fe*fe-max_energy*(1-x[i,j+1]))
                fixed=server.queue_delay_current+net.d_dl_constant
                fixed+=distances[i,access]/3e8 if j<m else cloud.propagation_delay+task.data_size_bits/cloud.backhaul_bandwidth
                parts.append(tx[i,j]+comp[i,j]+fixed*x[i,j+1])
                energies.append(user.p_tx*tx[i,j]+energy[i,j])
            local_latency=task.total_cycles/user.f_local
            local_energy=net.kappa_local*user.f_local**2*task.total_cycles
            upper=max(local_latency,max(task.data_size_bits/1e3+task.total_cycles/1e6+server.queue_delay_current+
                net.d_dl_constant+cloud.propagation_delay+task.data_size_bits/cloud.backhaul_bandwidth+np.max(distances[i])/3e8 for server in host))
            latency[i]=model.addVar(lb=0.,ub=upper)
            norm_energy[i]=model.addVar(lb=0.)
            model.addConstr(latency[i]==local_latency*x[i,0]+gp.quicksum(parts))
            model.addConstr(norm_energy[i]==(local_energy*x[i,0]+gp.quicksum(energies))/max(local_energy,1e-6))
            violated=model.addVar(vtype=gp.GRB.BINARY)
            model.addConstr(latency[i]<=task.deadline_sec+upper*violated)
            max_penalty=cost.penalty_deadline_multiplier*(upper/task.deadline_sec)**2
            penalty[i]=model.addVar(lb=0.,ub=max_penalty)
            model.addQConstr(penalty[i]>=cost.penalty_deadline_multiplier/task.deadline_sec**2*latency[i]*latency[i]-max_penalty*(1-violated))
        for j,server in enumerate(host):
            model.addConstr(gp.quicksum(f[i,j] for i in range(n))<=server.f_total/1e9)
        for access in range(m):
            terms=[b[i,j] for i in range(n) for j in range(m+1) if (j if j<m else int(np.argmin(distances[i])))==access]
            model.addConstr(gp.quicksum(terms)<=1)
        model.setObjective(gp.quicksum(cost.alpha*latency[i]/tasks[i].deadline_sec+(1-cost.alpha)*norm_energy[i]+penalty[i] for i in range(n)),gp.GRB.MINIMIZE)
        model.optimize()
        bound=max(lower,float(model.ObjBound)) if np.isfinite(model.ObjBound) else lower
        statuses={2:'OPTIMAL_WITH_REQUESTED_TOLERANCE',3:'INFEASIBLE',9:'TIME_LIMIT',11:'INTERRUPTED',12:'NUMERIC'}
        report=ReferenceReport(statuses.get(model.Status,str(model.Status)),time.perf_counter()-start,bound)
        if model.SolCount:
            targets=np.array([max(range(m+2),key=lambda j:x[i,j].X) for i in range(n)])
            cpu=np.array([users[i].f_local if target==0 else f[i,int(target)-1].X*1e9 for i,target in enumerate(targets)])
            bandwidth=np.array([0. if target==0 else b[i,int(target)-1].X for i,target in enumerate(targets)])
            for target,server in enumerate(host,1):
                selected=targets==target
                if cpu[selected].sum()>server.f_total:
                    cpu[selected]*=server.f_total/cpu[selected].sum()
            home=np.argmin(distances,axis=1)
            access=np.where(targets==m+1,home,targets-1)
            for edge in range(m):
                selected=(targets>0)&(access==edge)
                if bandwidth[selected].sum()>1:
                    bandwidth[selected]/=bandwidth[selected].sum()
            result=CostEngine(cost,net).evaluate_allocation(*args[:4],targets,cpu,bandwidth,*args[4:])
            report.incumbent_cost=result.total_cost;report.solver_objective=float(model.ObjVal)
            report.simulator_match=bool(np.isclose(result.total_cost,model.ObjVal,rtol=1e-4,atol=1e-5))
            report.reported_mip_gap=float(model.MIPGap)
            report.is_optimal=bool(model.Status==gp.GRB.OPTIMAL and report.simulator_match and model.MIPGap<=1e-8)
            if not report.simulator_match:
                report.lower_bound=lower;report.status='OBJECTIVE_MISMATCH';report.is_optimal=False
        return report
    except gp.GurobiError as error:
        return ReferenceReport('SOLVER_ERROR',time.perf_counter()-start,lower,message=str(error))
    finally:
        if model is not None:
            model.dispose()
