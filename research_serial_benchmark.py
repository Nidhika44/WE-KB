"""Isolated snapshot timing study; never supplies proxy times to parallel trajectories."""
from dataclasses import asdict
import copy
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from resume_research_fast import cases_for
from we_kb_fpa.checkpoint import Checkpoint,identity_hash
from we_kb_fpa.config import NetworkConfig
from we_kb_fpa.simulation_executor import SimulationExecutor
from we_kb_fpa.baselines import BaselineRunner
from we_kb_fpa.cost_engine import CostEngine
from we_kb_fpa.evaluation import fingerprint
from we_kb_fpa.research_method import ResearchOptimizer,research_metrics
from experiments.run_research import historical_bank,method_configs,load_legacy_models,set_channel
from experiments.research_report import report


def samples(settings,scenario,seed):
    slots=settings['shock_slots'] if scenario.study in ('shock','drl') else settings['slots']
    positions={0,slots//2} if scenario.study!='exact' else {0}
    sim=SimulationExecutor(net_config=NetworkConfig(num_users=scenario.users,num_edge_servers=scenario.edges),seed=seed)
    rng=np.random.default_rng(seed+50000000);history=[]
    for slot in range(max(positions)+1):
        if slot:
            sim.env.update_server_queues(sim.sim_cfg.time_step)
        tasks,sinr,distances,_,_=sim.sample_slot(slot)
        for task in tasks:
            task.deadline_sec*=scenario.tau_scale
        if scenario.task_mbit is not None:
            for task in tasks:
                task.data_size_bits=rng.uniform(.8,1.2)*scenario.task_mbit*1e6
        realized=set_channel(sinr,scenario.sinr_db)
        if slot>=slots//2 and scenario.fade_db:
            realized[:,scenario.affected_edges]*=10**(-scenario.fade_db/10)
        history.append(realized.copy())
        observed=history[max(0,len(history)-1-scenario.csi_age)].copy()
        if scenario.csi_error_db:
            observed*=10**(rng.normal(0,scenario.csi_error_db,observed.shape)/10)
        history=history[-(scenario.csi_age+1):]
        if slot in positions:
            args=(tasks,sim.env.users,sim.env.edge_servers,sim.env.cloud,observed,distances)
            yield slot,'pre' if slot<slots//2 else 'post',args,realized,sim.net_cfg


def timing_case(settings,root,scenario,seed,model_seed):
    metadata=dict(protocol='fresh-controller-serial-snapshots-v1',settings=settings,scenario=asdict(scenario),seed=seed,
                  harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  sampling='slot 0 and midpoint; exact study slot 0 only; independent fresh controller seed = seed + 100003*slot')
    checkpoint=Checkpoint(root,f'{scenario.study}-{scenario.name}-{seed}',metadata)
    state=checkpoint.load() or dict(records=[],done_slots=[])
    if state.get('complete'):
        return state['records']
    memory,warmup=historical_bank(scenario,seed)
    for slot,phase,args,realized,net in samples(settings,scenario,seed):
        if slot in state['done_slots']:
            continue
        controller_seed=seed+100003*slot
        methods={name:ResearchOptimizer(config,net,controller_seed) for name,config in method_configs(settings,scenario).items()}
        models=load_legacy_models(settings['models_dir'],model_seed,settings['model_budget']) if settings['include_drl'] and scenario.study in ('deadline','drl') else {}
        runner=BaselineRunner(net_config=net);engine=CostEngine(net_config=net)
        names=list(methods)+['Greedy-Latency','Greedy-Uniform']+list(models)
        # Rotation is independent of quality outcomes and balances method order across seeds.
        offset=(seed+slot)%len(names);expected=fingerprint(args)
        for name in names[offset:]+names[:offset]:
            if name in methods:
                output=methods[name].optimize(args,memory)
            elif name in models:
                output=models[name][0].run(*args,training=False)
            else:
                output=runner.run_greedy_latency(*args)
            if fingerprint(args)!=expected:
                raise RuntimeError('Timing controller changed the common observation')
            actual=(*args[:4],realized,args[5]);predicted=output.best_cost
            output.evaluation_result=engine.evaluate_allocation(*actual[:4],output.best_targets,output.best_cpu,output.best_bw,*actual[4:])
            output.best_cost=output.evaluation_result.total_cost
            state['records'].append(dict(study=scenario.study,scenario=scenario.name,seed=seed,slot=slot,phase=phase,
                algorithm=name,model_seed=model_seed if scenario.study in ('deadline','drl') else None,
                N=scenario.users,M=scenario.edges,tau_scale=scenario.tau_scale,target_sinr_db=scenario.sinr_db,
                mean_sinr_db=10*np.log10(realized.mean()),task_mbit=scenario.task_mbit,q=scenario.q,
                csi_error_db=scenario.csi_error_db,csi_age=scenario.csi_age,fade_db=scenario.fade_db,
                affected_edges=';'.join(map(str,scenario.affected_edges)),kb_sinr_db=scenario.kb_sinr_db,
                benchmark_mode='isolated_serial_fresh_controller_snapshots',controller_seed=controller_seed,
                observation_hash=expected,realization_hash=fingerprint(actual),predicted_search_objective=predicted,
                nominal_quality_pass=models[name][1] if name in models else None,
                **research_metrics(output,actual,net),**getattr(output,'diagnostics',{})))
        state['done_slots'].append(slot);checkpoint.save(state)
    state['complete']=True;checkpoint.save(state,compact=True)
    return state['records']


def run_serial_benchmark(settings,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    protocol=dict(protocol='fresh-controller-serial-snapshots-v1',settings=settings,
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Prespecified serial snapshots on original hardware. Not a replay of the parallel decision trajectories. '
              'No proxy runtime substituted into parallel rows. Two snapshots per seed/scenario; exact: one. '
              'Fresh independent controller RNG at each snapshot; all operators/budgets/history and realized CSI scoring unchanged.')
    path=output/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=json.loads(json.dumps(protocol)):
        raise ValueError('Serial benchmark protocol changed')
    path.write_text(json.dumps(protocol,indent=2))
    root=output/'checkpoints'/identity_hash(protocol)[:24];root.mkdir(parents=True,exist_ok=True)
    records=[];tick=time.perf_counter();cases=cases_for(settings)
    for index,(seed,scenario,model_seed) in enumerate(cases,1):
        records.extend(timing_case(settings,root,scenario,seed,model_seed))
        if index%25==0:
            pending=output/'slots.pending.csv';pd.DataFrame(records).to_csv(pending,index=False);pending.replace(output/'slots.csv')
        print(f'Serial snapshot benchmark {index}/{len(cases)}; elapsed {(time.perf_counter()-tick)/60:.1f} min',flush=True)
    frame=pd.DataFrame(records);frame.to_csv(output/'slots.csv',index=False)
    # Exact optimization references belong to the quality run, not this timing-only study.
    settings_report=copy.deepcopy(settings);settings_report['benchmark_scope']=protocol['scope']
    report(frame[frame.study!='exact'],output,settings_report)
    (output/'complete.json').write_text(json.dumps(dict(cases=len(cases),seconds=time.perf_counter()-tick),indent=2))
    return frame
