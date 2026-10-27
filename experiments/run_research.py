"""Research extensions of the original workbook's penalized-cost implementation."""
import argparse
import copy
from dataclasses import dataclass,asdict,replace
import hashlib
import itertools
from importlib.metadata import version,PackageNotFoundError
import json
import os
from pathlib import Path
import platform
import time
import numpy as np
import pandas as pd
from we_kb_fpa.checkpoint import Checkpoint,RunLock,namespace,implementation_signature
from we_kb_fpa.config import NetworkConfig,TaskConfig,ChannelConfig,CostConfig
from we_kb_fpa.simulation_executor import SimulationExecutor
from we_kb_fpa.baselines import BaselineRunner
from we_kb_fpa.cost_engine import CostEngine
from we_kb_fpa.dqn import DQNBaseline
from we_kb_fpa.droo import DROOAdapted
from we_kb_fpa.evaluation import fingerprint
from we_kb_fpa.research_method import ResearchMethodConfig,ResearchOptimizer,ResearchMemory,research_metrics,optimistic_latency
from we_kb_fpa.research_reference import solve_reference
from experiments.research_report import report


@dataclass(frozen=True)
class Scenario:
    study: str
    name: str
    users: int=20
    edges: int=4
    sinr_db: float|None=None
    task_mbit: float|None=None
    tau_scale: float=1.
    fade_db: float=0.
    affected_edges: tuple=()
    q: int=100
    kb_sinr_db: float=10.
    random_anchor: bool=False
    csi_error_db: float=0.
    csi_age: int=0
    historical_users: int|None=None
    historical_edges: int|None=None


def scenarios():
    result=[]
    result += [Scenario('scalability',f'N{n}',users=n) for n in [10,20,50,100,200,500]]
    result += [Scenario('channel',f'SINR{v:g}',sinr_db=v) for v in [-5.,0.,5.,10.,15.,20.]]
    result += [Scenario('workload',f'D{v:g}',task_mbit=v) for v in [.5,1.,1.5,2.,2.5,3.5,5.]]
    result += [Scenario('ablation','factorial',sinr_db=-5.)]
    result += [Scenario('kb',f'Q{q}',q=q,sinr_db=10.) for q in [0,10,100,1000]]
    result += [Scenario('kb','matched',sinr_db=-5.,kb_sinr_db=-5.),Scenario('kb','mismatch',sinr_db=-5.,kb_sinr_db=10.),
               Scenario('kb','random-anchor',sinr_db=-5.,kb_sinr_db=-5.,random_anchor=True),
               Scenario('kb','changed-N',users=30,historical_users=20),Scenario('kb','changed-M',edges=3,historical_edges=4)]
    result += [Scenario('shock',f'partial-{fade}',users=8,edges=3,fade_db=fade,affected_edges=(0,)) for fade in [10.,20.,35.]]
    result += [Scenario('shock','all-edge-stress-35',users=8,edges=3,fade_db=35.,affected_edges=(0,1,2))]
    result += [Scenario('deadline',f'SINR{sinr:g}-tau{tau:g}',users=8,edges=3,sinr_db=sinr,tau_scale=tau)
               for sinr,tau in itertools.product([-5.,10.],[.25,.5,.75,1.,1.5,2.,3.])]
    result += [Scenario('exact',f'N{n}',users=n,edges=3,q=10) for n in [3,5,8,10,15,20,25,30]]
    result += [Scenario('csi',f'error-{error:g}',csi_error_db=error) for error in [0.,1.,3.,6.]]
    result += [Scenario('csi',f'age-{age}',csi_age=age) for age in [1,3,5]]
    result += [Scenario('drl','partial-20',users=8,edges=3,fade_db=20.,affected_edges=(0,)),
               Scenario('drl','all-edge-stress-35',users=8,edges=3,fade_db=35.,affected_edges=(0,1,2))]
    return result


def method_configs(settings,scenario):
    base=ResearchMethodConfig(evaluations=settings['evaluations'],population=settings['population'],random_anchor=scenario.random_anchor)
    methods={'WE-KB-FPA-Greedy':base,'PSO+WE-KB':replace(base,family='pso'),
             'GA+WE-KB':replace(base,family='ga'),'FPA+Pruner':replace(base,kb=False,phi=False,switch=False),
             'FPA-Greedy':replace(base,kb=False,phi=False,switch=False,pruner=False)}
    if scenario.study=='ablation':
        for pruning,phi,switch,kb in itertools.product([False,True],repeat=4):
            name=f'FPA-P{int(pruning)}-Phi{int(phi)}-Pt{int(switch)}-KB{int(kb)}'
            methods[name]=replace(base,pruner=pruning,phi=phi,switch=switch,kb=kb)
        methods['FPA-ReversePhi']=replace(base,phi_direction=-1)
        methods['FPA-ReversePt']=replace(base,switch_direction=-1)
        methods['FPA-ReverseBoth']=replace(base,phi_direction=-1,switch_direction=-1)
    return methods


def set_channel(sinr,db):
    result=sinr.copy()
    if db is not None:
        result*=10**(db/10)/max(float(result.mean()),1e-30)
    return result


def historical_bank(scenario,seed):
    memory=ResearchMemory(scenario.q)
    if not scenario.q:
        return memory,0.
    tick=time.perf_counter()
    net=NetworkConfig(num_users=scenario.historical_users or scenario.users,num_edge_servers=scenario.historical_edges or scenario.edges)
    sim=SimulationExecutor(net_config=net,seed=seed+10000000)
    runner=BaselineRunner(net_config=net)
    for slot in range(scenario.q):
        tasks,sinr,distances,_,_=sim.sample_slot(slot)
        sinr=set_channel(sinr,scenario.kb_sinr_db)
        args=(tasks,sim.env.users,sim.env.edge_servers,sim.env.cloud,sinr,distances)
        output=runner.run_greedy_latency(*args)
        memory.add(args,np.r_[output.best_targets,np.ones(2*len(tasks))])
    return memory,time.perf_counter()-tick


def load_legacy_models(directory,seed,budget):
    directory=Path(directory)
    quality=json.loads((directory/'quality_gate.json').read_text())
    quality=next(row for row in quality if row['seed']==seed and row['training_budget']==budget)
    result={}
    for kind,cls in [('dqn',DQNBaseline),('droo',DROOAdapted)]:
        path=directory/f'{kind}_seed_{seed}_budget_{budget}.npz'
        with np.load(path) as data:
            agent=cls(net_config=NetworkConfig(num_users=8,num_edge_servers=3),seed=seed)
            count=sum(key.startswith('weight_') for key in data.files)
            agent.weights=[data[f'weight_{i}'].copy() for i in range(count)]
            if kind=='dqn':
                agent.target=[weight.copy() for weight in agent.weights]
        result['DQN-Frozen-Legacy' if kind=='dqn' else 'DROO-Adapted-Frozen-Legacy']=(agent,quality[kind]['nominal_quality_pass'])
    return result


def run_case(settings,root,scenario,seed,model_seed):
    metadata=dict(scenario=asdict(scenario),seed=seed,settings=settings)
    checkpoint=Checkpoint(root,f'{scenario.study}-{scenario.name}-{seed}',metadata)
    state=checkpoint.load()
    if state and state.get('complete'):
        return state['records']
    if state is None:
        net=NetworkConfig(num_users=scenario.users,num_edge_servers=scenario.edges)
        memory,warmup=historical_bank(scenario,seed)
        state=dict(sim=SimulationExecutor(net_config=net,seed=seed),memory=memory,kb_build_seconds=warmup,
                   methods={name:ResearchOptimizer(config,net,seed) for name,config in method_configs(settings,scenario).items()},
                   rng=np.random.default_rng(seed+50000000),previous=[],records=[],next_slot=0,models={})
        if scenario.study in ('deadline','drl') and settings['include_drl']:
            state['models']=load_legacy_models(settings['models_dir'],model_seed,settings['model_budget'])
    sim=state['sim'];net=sim.net_cfg;engine=CostEngine(net_config=net);runner=BaselineRunner(net_config=net)
    slots=settings['shock_slots'] if scenario.study in ('shock','drl') else settings['slots']
    if scenario.study=='exact':
        slots=1
    for slot in range(state['next_slot'],slots):
        if slot:
            sim.env.update_server_queues(sim.sim_cfg.time_step)
        tasks,sinr,distances,_,_=sim.sample_slot(slot)
        for task in tasks:
            task.deadline_sec*=scenario.tau_scale
        if scenario.task_mbit is not None:
            for task in tasks:
                task.data_size_bits=state['rng'].uniform(.8,1.2)*scenario.task_mbit*1e6
        realized=set_channel(sinr,scenario.sinr_db)
        if slot>=slots//2 and scenario.fade_db:
            realized[:,scenario.affected_edges]*=10**(-scenario.fade_db/10)
        state['previous'].append(realized.copy())
        observed=state['previous'][max(0,len(state['previous'])-1-scenario.csi_age)].copy()
        if scenario.csi_error_db:
            observed*=10**(state['rng'].normal(0,scenario.csi_error_db,observed.shape)/10)
        state['previous']=state['previous'][-(scenario.csi_age+1):]
        args=(tasks,sim.env.users,sim.env.edge_servers,sim.env.cloud,observed,distances)
        actual=(*args[:4],realized,distances)
        expected=fingerprint(args);names=list(state['methods'])+['Greedy-Latency','Greedy-Uniform']+list(state['models'])
        deadlines=np.array([task.deadline_sec for task in tasks])[:,None]
        excluded=optimistic_latency(args,net)>deadlines
        possible=optimistic_latency(actual,net)<=deadlines
        incorrect_exclusions=float(np.mean(excluded&possible))
        offset=slot%len(names);outputs={}
        for name in names[offset:]+names[:offset]:
            if name in state['methods']:
                output=state['methods'][name].optimize(args,state['memory'])
            elif name in state['models']:
                output=state['models'][name][0].run(*args,training=False)
            else:
                output=runner.run_greedy_latency(*args)
            if fingerprint(args)!=expected:
                raise RuntimeError('Algorithm mutated shared observations')
            predicted=output.best_cost
            result=engine.evaluate_allocation(*actual[:4],output.best_targets,output.best_cpu,output.best_bw,*actual[4:])
            output.best_cost=result.total_cost;output.evaluation_result=result
            outputs[name]=output
            row=dict(study=scenario.study,scenario=scenario.name,seed=seed,model_seed=model_seed if scenario.study in ('deadline','drl') else None,
                algorithm=name,slot=slot,phase='post' if slot>=slots//2 else 'pre',N=scenario.users,M=scenario.edges,
                tau_scale=scenario.tau_scale,target_sinr_db=scenario.sinr_db,mean_sinr_db=10*np.log10(realized.mean()),
                observed_mean_sinr_db=10*np.log10(observed.mean()),task_mbit=scenario.task_mbit,fade_db=scenario.fade_db,
                affected_edges=';'.join(map(str,scenario.affected_edges)),q=scenario.q,kb_sinr_db=scenario.kb_sinr_db,
                random_anchor=scenario.random_anchor,csi_error_db=scenario.csi_error_db,csi_age=scenario.csi_age,
                csi_aging_available_slots=min(slot,scenario.csi_age),observation_hash=expected,realization_hash=fingerprint(actual),
                observed_pruned_but_realized_possible_pct=100*incorrect_exclusions,
                predicted_search_objective=predicted,nominal_quality_pass=state['models'][name][1] if name in state['models'] else None,
                kb_build_seconds=state['kb_build_seconds'],**research_metrics(output,actual,net),**getattr(output,'diagnostics',{}))
            state['records'].append(row)
        if scenario.study=='exact':
            reference=asdict(solve_reference(actual,net,settings['reference_seconds'],settings['mip_gap'],seed))
            best_known=min(output.best_cost for output in outputs.values())
            if reference['incumbent_cost'] is not None:
                best_known=min(best_known,reference['incumbent_cost'])
            for row in state['records']:
                row.update({f'reference_{key}':value for key,value in reference.items()})
                row['best_known_cost']=best_known
                row['best_known_ratio']=row['search_objective']/best_known if best_known>0 else None
                row['certified_gap_upper_pct']=100*(row['search_objective']/reference['lower_bound']-1) if reference['lower_bound']>0 else None
                row['gap_to_proven_optimum_pct']=100*(row['search_objective']/reference['incumbent_cost']-1) if reference['is_optimal'] and reference['incumbent_cost'] else None
        actual_greedy=outputs['Greedy-Latency'].best_cost
        for row in state['records']:
            if row['slot']==slot:
                row['realized_greedy_dominance_verified']=row['search_objective']<=actual_greedy+1e-8
        state['next_slot']=slot+1
        if (slot+1)%settings['checkpoint_every']==0:
            checkpoint.save(state)
    checkpoint.save(dict(complete=True,records=state['records']),compact=True)
    return state['records']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick',action='store_true')
    parser.add_argument('--rigorous',action='store_true')
    parser.add_argument('--study',nargs='+',choices=['all','scalability','channel','workload','ablation','kb','shock','deadline','exact','csi','drl'],default=['all'])
    parser.add_argument('--seeds',nargs='+',type=int)
    parser.add_argument('--slots',type=int)
    parser.add_argument('--shock-slots',type=int)
    parser.add_argument('--evaluations',type=int)
    parser.add_argument('--population',type=int)
    parser.add_argument('--checkpoint-every',type=int,default=1)
    parser.add_argument('--output-dir',default='output/research_extension')
    parser.add_argument('--models-dir',default='output/topology_shock')
    parser.add_argument('--model-budget',type=int,default=20000)
    parser.add_argument('--skip-drl',action='store_true')
    parser.add_argument('--reference-seconds',type=float,default=60.)
    parser.add_argument('--mip-gap',type=float,default=.01)
    parser.add_argument('--require-gurobi',action='store_true')
    parser.add_argument('--role',choices=['development','final'],default='development')
    parser.add_argument('--freeze-protocol',action='store_true')
    parser.add_argument('--protocol')
    parser.add_argument('--plan-only',action='store_true')
    parser.add_argument('--plots-only',action='store_true')
    args=parser.parse_args()
    if args.plots_only:
        output=Path(args.output_dir)
        saved=json.loads((output/'config.json').read_text())['settings']
        with RunLock(output/'run.lock'):
            report(pd.read_csv(output/'per_seed_slots.csv',dtype={'feasible_task_ids':'string'}),output,saved)
        return
    if args.quick and args.rigorous:
        parser.error('Choose quick or rigorous')
    first=60001 if args.role=='final' else 50001
    seeds=args.seeds or list(range(first,first+(30 if args.rigorous else 2)))
    settings=dict(protocol='legacy-greedy-seeded-research-v1',role=args.role,seeds=seeds,
        slots=args.slots or (10 if args.rigorous else 4),shock_slots=args.shock_slots or (100 if args.rigorous else 6),
        evaluations=args.evaluations or (620 if args.rigorous else 24),population=args.population or (20 if args.rigorous else 6),
        checkpoint_every=args.checkpoint_every,include_drl=not args.skip_drl,models_dir=str(Path(args.models_dir).resolve()),
        model_budget=args.model_budget,reference_seconds=args.reference_seconds,mip_gap=args.mip_gap,
        network=asdict(NetworkConfig()),tasks=asdict(TaskConfig()),channel=asdict(ChannelConfig()),cost=asdict(CostConfig()),
        adaptation=asdict(ResearchMethodConfig()),hardware=dict(processor=platform.processor(),python=platform.python_version(),
            threads={name:os.environ.get(name) for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']}),
        objective='Original penalized all-task normalized latency-energy objective. Feasible-only cost is an additional conditional reporting metric.',
        admission='No rejection action in original implementation: admission=100%; deadline-feasible percentage is separately reported.',
        pruning='Necessary per-task hard-deadline bounds, not proof of global penalized-objective dominance. All-false rows stay marked infeasible; soft search retains options.',
        greedy='Every metaheuristic retains an unpruned scored greedy incumbent, counted in its common evaluation budget. Greedy-Latency and Greedy-Uniform are allocation aliases.',
        knowledge='Frozen, independent offline greedy history shared by all methods; FIFO capacity, original summary Euclidean retrieval; reject mismatched ordered user/edge identities.',
        csi='Optimize on estimated/aged SINR, rescore unchanged allocations on realized SINR. No realized-channel repair or oracle assistance.',
        drl='Secondary frozen inference using unchanged original DQN/DROO architectures and saved weights; no replay reset presented as online adaptation.',
        recovery='Old recovery-rate definition is omitted; no recovery claims are generated.',
        provenance='Original workbook sourced output/topology_shock and output/data; preserved source archive in output/research_extension_archive.')
    try:
        settings['gurobi_version']=version('gurobipy')
    except PackageNotFoundError:
        settings['gurobi_version']=None
    settings['exact_slots']=1
    settings['operators']=dict(pso_inertia=.7,pso_cognitive=1.5,pso_social=1.5,pso_velocity_fraction=.2,
        ga_crossover='uniform',ga_coordinate_mutation=.1,ga_tournament=3,kb_population_fraction=.3,
        kb_perturbation_std=.03,levy_exponent=1.5)
    settings['history_seed_rule']='evaluation seed + 10000000; historical simulation is separate from evaluation'
    if (len(set(seeds))!=len(seeds) or any(seed<0 for seed in seeds) or settings['slots']<2 or settings['shock_slots']<4 or
        settings['checkpoint_every']<1 or settings['reference_seconds']<=0 or not 0<settings['mip_gap']<1):
        parser.error('Invalid seed, slot, checkpoint or reference configuration')
    ResearchMethodConfig(evaluations=settings['evaluations'],population=settings['population'])
    if args.role=='final' and (not args.rigorous or len(seeds)<30 or min(seeds)<60000):
        parser.error('Final protocol requires rigorous settings and at least 30 reserved seeds >=60000')
    selected=[scenario for scenario in scenarios() if 'all' in args.study or scenario.study in args.study]
    if args.require_gurobi and settings['gurobi_version'] is None:
        parser.error('gurobipy is unavailable. Install the optional gurobi extra and configure a license before requiring that benchmark.')
    if settings['gurobi_version'] is None and any(s.study=='exact' for s in selected):
        print('Gurobi unavailable: reference study will report analytical lower bounds and best-known costs, not proven optimality gaps.',flush=True)
    settings['scenarios']=[asdict(scenario) for scenario in selected]
    if settings['include_drl'] and any(s.study in ('deadline','drl') for s in selected):
        files=[Path(args.models_dir)/'quality_gate.json']
        for seed in range(42,42+len(seeds)):
            files += [Path(args.models_dir)/f'{kind}_seed_{seed}_budget_{args.model_budget}.npz' for kind in ['dqn','droo']]
        if any(not path.exists() for path in files):
            parser.error('Legacy learner bank missing; provide --models-dir or explicitly use --skip-drl')
        settings['learner_checksums']={str(path.resolve()):hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=True)
    frozen=dict(settings=settings,implementation=implementation_signature())
    if args.plan_only:
        print(json.dumps(dict(cases=len(seeds)*len(selected),scenarios=len(selected),settings=settings),indent=2));return
    path=Path(args.protocol) if args.protocol else output/'frozen_protocol.json'
    if args.freeze_protocol:
        if path.exists():
            parser.error('Preserve the existing frozen protocol')
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(frozen,indent=2));print(f'Frozen: {path}. No simulation run.');return
    if args.role=='final' and (not path.exists() or json.loads(path.read_text())!=json.loads(json.dumps(frozen))):
        parser.error('Freeze this exact final protocol before running reserved seeds')
    tick=time.perf_counter()
    with RunLock(output/'run.lock'):
        root=namespace(output/'checkpoints',settings);(output/'config.json').write_text(json.dumps(frozen,indent=2))
        records=[];cases=[(seed,scenario,index) for index,seed in enumerate(seeds) for scenario in selected]
        for case,(seed,scenario,index) in enumerate(cases,1):
            rows=run_case(settings,root,scenario,seed,42+index);records.extend(rows)
            directory=output/'per_seed';directory.mkdir(exist_ok=True)
            pd.DataFrame(rows).to_csv(directory/f'{scenario.study}-{scenario.name}-{seed}.csv',index=False)
            pending=output/'per_seed_slots.pending.csv';pd.DataFrame(records).to_csv(pending,index=False);pending.replace(output/'per_seed_slots.csv')
            print(f'Completed {case}/{len(cases)}: {scenario.study}/{scenario.name}, seed {seed}; elapsed {(time.perf_counter()-tick)/60:.1f} min',flush=True)
        report(pd.DataFrame(records),output,settings)
        (output/'timing.json').write_text(json.dumps(dict(seconds=time.perf_counter()-tick,cases=len(cases)),indent=2))
    print(f'Research outputs: {output.resolve()}; repeat the same command to resume.')


if __name__=='__main__':
    main()
