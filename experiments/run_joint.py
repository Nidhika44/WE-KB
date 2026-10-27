"""Separate hard-QoS joint-allocation protocol; legacy soft-cost results are preserved."""
import argparse
from dataclasses import asdict,replace
import json
import os
import platform
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from we_kb_fpa.checkpoint import Checkpoint,RunLock,namespace,implementation_signature
from we_kb_fpa.config import NetworkConfig,CostConfig
from we_kb_fpa.evaluation import snapshots,fingerprint
from we_kb_fpa.joint_allocation import JointAllocationConfig
from we_kb_fpa.joint_search import JointSearchConfig,JointOffloadingOptimizer,offered_task_metrics
from experiments.train_joint import load_models,load_model


def methods(budget,population,evaluations):
    base=JointSearchConfig(budget_ms=budget,population=population,max_evaluations=evaluations)
    return {'FPA-Joint':base,'GA-Joint':replace(base,strategy='ga'),
            'Anchor-Joint':replace(base,strategy='anchor'),'Greedy-Joint':replace(base,strategy='greedy',use_knowledge=False),
            'FPA-Joint-NoKB':replace(base,use_knowledge=False),'GA-Joint-NoKB':replace(base,strategy='ga',use_knowledge=False),
            'FPA-Joint-NoPrune':replace(base,use_pruning=False),'GA-Joint-NoPrune':replace(base,strategy='ga',use_pruning=False)}


def run_case(settings,root,seed,fade,resume=True):
    checkpoint=Checkpoint(root,f'joint-{seed}-{fade:g}',dict(seed=seed,fade=fade),resume)
    state=checkpoint.load()
    if state and state.get('complete'):
        return state['records']
    if state is None:
        allocation=JointAllocationConfig(**settings['allocation'])
        net=NetworkConfig(**settings['network'])
        state=dict(stream=snapshots(seed,settings['slots'],fading_db=fade,midpoint=settings['slots']//2),
                   optimizers={name:JointOffloadingOptimizer(JointSearchConfig(**cfg),allocation,net,seed=seed)
                               for name,cfg in settings['methods'].items()},records=[])
        if settings.get('learner_bank'):
            bank=settings['learner_bank']
            model_seed=settings['model_assignment'][str(seed)]
            for kind,label in [('dqn','DQN-Joint'),('categorical','DROO-Inspired-Categorical')]:
                for online in [False,True]:
                    learner=load_model(bank,kind,model_seed)
                    learner.online=online
                    state['optimizers'][label+('-Online' if online else '-Frozen')]=learner
                hybrid=JointOffloadingOptimizer(JointSearchConfig(strategy='anchor',budget_ms=settings['allocation']['decision_reserve_sec']*1000),allocation,net,seed=seed)
                hybrid.policy=load_model(bank,kind,model_seed)
                state['optimizers'][label+'-SharedAnchors']=hybrid
    last_saved=time.perf_counter()
    for slot,args,_,_ in state['stream']:
        expected=fingerprint(args)
        names=list(state['optimizers'])
        offset=slot%len(names)
        for name in names[offset:]+names[:offset]:
            output=state['optimizers'][name].optimize(*args)
            if fingerprint(args)!=expected:
                raise RuntimeError('Joint method mutated the shared snapshot')
            row=dict(seed=seed,fade=fade,slot=slot,phase='pre' if slot<settings['slots']//2 else 'post',
                     algorithm=name,snapshot_hash=expected,
                     model_seed=settings.get('model_assignment',{}).get(str(seed)),
                     admitted_task_ids=';'.join(str(task.task_id) for task,target in zip(args[0],output.solution.targets) if target>=0),
                     **offered_task_metrics(output,args[0],settings['allocation']['decision_reserve_sec']),**output.diagnostics)
            state['records'].append(row)
        if (slot+1)%settings['checkpoint_every']==0 or time.perf_counter()-last_saved>=60:
            checkpoint.save(state)
            last_saved=time.perf_counter()
    checkpoint.save(dict(complete=True,records=state['records']),compact=True)
    return state['records']


def summarize(frame,output):
    metrics=['admission_pct','offered_miss_pct','service_cost','service_cost_per_admitted','runtime_ms',
             'budget_overrun','decision_reservation_overrun','assignments_evaluated','unresolved_allocator_calls',
             'selected_fixed_assignment_certified']
    metrics += [name for name in ['individual_admission_upper_bound','reserved_admission_upper_bound_reached'] if name in frame]
    metrics += [name for name in ['total_admitted_energy_j','mean_admitted_energy_j','mean_admitted_service_latency_ms',
                                  'mean_admitted_completion_latency_ms','minimum_reserved_deadline_margin_ms'] if name in frame]
    seeds=frame.groupby(['seed','fade','phase','algorithm'])[metrics].mean().reset_index()
    seeds.to_csv(output/'seed_summary.csv',index=False)
    seeds.groupby(['fade','phase','algorithm'])[metrics].mean().to_csv(output/'summary.csv')
    intervals=[]
    for (fade,phase,algorithm),group in seeds.groupby(['fade','phase','algorithm']):
        for metric in metrics:
            values=group[metric].dropna();n=len(values)
            intervals.append(dict(fade=fade,phase=phase,algorithm=algorithm,metric=metric,independent_seeds=n,
                mean=float(values.mean()) if n else None,
                ci95_margin=float(t.ppf(.975,n-1)*values.std()/np.sqrt(n)) if n>1 else None))
    pd.DataFrame(intervals).to_csv(output/'summary_ci95.csv',index=False)
    reports=[]
    for baseline in sorted(set(frame.algorithm)-{'FPA-Joint'}):
        left=seeds[seeds.algorithm=='FPA-Joint']
        right=seeds[seeds.algorithm==baseline]
        paired=left.merge(right,on=['seed','fade','phase'],suffixes=('_fpa','_baseline'))
        for (fade,phase),group in paired.groupby(['fade','phase']):
            for metric in ['admission_pct','offered_miss_pct','runtime_ms']:
                difference=group[metric+'_fpa']-group[metric+'_baseline']
                n=len(difference)
                reports.append(dict(baseline=baseline,fade=fade,phase=phase,metric=metric,
                                    mean_difference=float(difference.mean()),independent_seeds=n,
                                    ci95_margin=float(t.ppf(.975,n-1)*difference.std()/np.sqrt(n)) if n>1 else None))
    pd.DataFrame(reports).to_csv(output/'paired_effects.csv',index=False)
    comparisons=[]
    for baseline in sorted(set(frame.algorithm)-{'FPA-Joint'}):
        selected=frame[frame.algorithm=='FPA-Joint'].merge(frame[frame.algorithm==baseline],on=['seed','fade','slot','phase'],suffixes=('_fpa','_baseline'))
        selected['baseline']=baseline
        selected['same_admission_count']=selected.admitted_tasks_fpa==selected.admitted_tasks_baseline
        selected['same_admitted_task_set']=selected.admitted_task_ids_fpa.fillna('')==selected.admitted_task_ids_baseline.fillna('')
        selected['conditional_cost_difference']=np.where(selected.same_admitted_task_set,
            selected.service_cost_fpa-selected.service_cost_baseline,np.nan)
        comparisons.append(selected[['seed','fade','slot','phase','baseline','admitted_tasks_fpa','admitted_tasks_baseline',
            'same_admission_count','same_admitted_task_set','service_cost_fpa','service_cost_baseline','conditional_cost_difference']])
    if comparisons:
        pd.concat(comparisons).to_csv(output/'cost_comparability.csv',index=False)
    plot(frame,output)


def plot(frame,output):
    directory=output/'figures'
    directory.mkdir(exist_ok=True)
    primary=['FPA-Joint','GA-Joint','Greedy-Joint','DQN-Joint-Frozen','DQN-Joint-Online',
             'DROO-Inspired-Categorical-Frozen','DROO-Inspired-Categorical-Online']
    for fade,scenario in frame.groupby('fade'):
        fig,axes=plt.subplots(1,3,figsize=(13,4))
        for axis,metric,label in zip(axes,['admission_pct','offered_miss_pct','runtime_ms'],
                ['Admitted tasks (%)','Misses among all offered tasks (%)','Complete decision time (ms)']):
            for name,group in scenario[scenario.algorithm.isin(primary)].groupby('algorithm'):
                values=group.groupby('slot')[metric].agg(['mean','std','count'])
                margin=t.ppf(.975,np.maximum(1,values['count']-1))*values['std'].fillna(0)/np.sqrt(values['count'])
                axis.plot(values.index,values['mean'],label=name)
                lower=np.maximum(0.,values['mean']-margin)
                upper=np.minimum(100.,values['mean']+margin) if metric!='runtime_ms' else values['mean']+margin
                axis.fill_between(values.index,lower,upper,alpha=.12)
            axis.axvline(scenario.slot.max()/2,color='black',linestyle='--')
            axis.set(xlabel='Slot',ylabel=label)
            if metric!='runtime_ms':
                axis.set_ylim(0,100)
            axis.grid(alpha=.2)
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        for suffix in ['pdf','png']:
            fig.savefig(directory/f'joint_admission_fade_{fade:g}.{suffix}',dpi=250)
        plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--quick',action='store_true')
    mode.add_argument('--rigorous',action='store_true')
    parser.add_argument('--seeds',type=int,nargs='+')
    parser.add_argument('--slots',type=int)
    parser.add_argument('--fading-values',type=float,nargs='+',default=[15.,35.])
    parser.add_argument('--budget-ms',type=float,default=50.)
    parser.add_argument('--population',type=int,default=8)
    parser.add_argument('--max-evaluations',type=int,default=24)
    parser.add_argument('--allocator-iterations',type=int,default=60)
    parser.add_argument('--checkpoint-every',type=int,default=5)
    parser.add_argument('--output-dir',default='output/joint_taskaware')
    parser.add_argument('--fresh',action='store_true')
    parser.add_argument('--plots-only',action='store_true')
    parser.add_argument('--role',choices=['development','final'],default='development')
    parser.add_argument('--freeze-protocol',action='store_true')
    parser.add_argument('--protocol')
    parser.add_argument('--models-dir')
    args=parser.parse_args()
    output=Path(args.output_dir)
    output.mkdir(parents=True,exist_ok=True)
    if args.plots_only:
        with RunLock(output/'run.lock'):
            summarize(pd.read_csv(output/'slots.csv',dtype={'admitted_task_ids':'string'}),output)
        return
    first=30001 if args.role=='final' else 10001
    seeds=args.seeds or list(range(first,first+(30 if args.rigorous else 2)))
    slots=args.slots or (100 if args.rigorous else 12)
    if (slots<6 or slots%2 or args.checkpoint_every<1 or len(set(seeds))!=len(seeds) or
            any(seed<0 or seed>=100000 for seed in seeds) or
            any(not np.isfinite(fade) or fade<=0 for fade in args.fading_values)):
        parser.error('Use even slots >=6, unique seeds, positive fades and checkpoint interval')
    if args.role=='final' and any(seed<30000 for seed in seeds):
        parser.error('Final runs require reserved seeds >=30000 and a frozen protocol')
    try:
        allocation=JointAllocationConfig(decision_reserve_sec=args.budget_ms/1000,max_iterations=args.allocator_iterations)
        configurations=methods(args.budget_ms,args.population,args.max_evaluations)
    except ValueError as error:
        parser.error(str(error))
    settings=dict(protocol='joint-hard-admission-v2',role=args.role,seeds=seeds,slots=slots,
                  fades=sorted(set([0.]+args.fading_values)),checkpoint_every=args.checkpoint_every,
                  network=asdict(NetworkConfig(num_users=8,num_edge_servers=3)),cost=asdict(CostConfig()),
                  allocation=asdict(allocation),methods={name:asdict(cfg) for name,cfg in configurations.items()},
                  objective='Lexicographic: maximize admitted task count, then minimize normalized latency-energy cost of admitted tasks.',
                  qos='Service latency plus the same reserved decision budget must meet each admitted deadline. Drops count as misses among all offered tasks.',
                  domain='CPU >=1 MHz for admitted remote tasks; allocated radio rate >=1 kbps; shared uplink capacity includes cloud traffic.',
                  certificate='Convex fixed-placement bounds only. Neither heuristic admission optimality nor global joint MINLP optimality is claimed.',
                  repair='Same deterministic drop repair for every method; unresolved numerical solves are recorded, not labelled infeasible.',
                  comparison='Different formulation from legacy penalized-deadline runs; do not mix their costs or reuse their DRL policies as trained baselines.',
                  drl='Not included in this bounded joint check; admission actions and the new objective require matched retraining.',
                  runtime='All controller work and memory insertion counted. Soft decision budgets; actual offered misses and overruns reported.')
    settings['decision_window']='A common conservative controller reservation is imposed on every algorithm. Service cost excludes controller execution; actual misses include it.'
    settings['knowledge']='Full task/user/channel/server context, canonical user ordering, targets mapped back by user ID. Same memory rule for FPA, GA and anchor control.'
    settings['units']='Seconds, CPU Hz, radio fractions; service_cost is dimensionless, energy is joules. Conditional service costs/energy must be read together with admission.'
    settings['timing_environment']=dict(processor=platform.processor(),threads={name:os.environ.get(name) for name in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']})
    if args.models_dir:
        bank=load_models(args.models_dir,allocation,NetworkConfig(**settings['network']))
        if args.role=='final' and bank['status']=='smoke_only':
            parser.error('Smoke-trained models cannot be used as final trained baselines')
        model_seeds=bank['training']['seeds']
        if args.role=='final' and len(model_seeds)!=len(seeds):
            parser.error('Final evaluation requires one independent trained replicate per evaluation seed')
        settings['learner_bank']=bank
        settings['model_assignment']={str(seed):model_seeds[index%len(model_seeds)] for index,seed in enumerate(seeds)}
        settings['drl']='Matched sequential DQN without greedy fallback; explicitly adapted categorical DROO-inspired learner with local/greedy candidate augmentation. Frozen and online variants.'
        settings['inference']='Same fixed-placement allocator, capacities, repair, reserve and observable context; neural candidate count differs and complete wall time is measured.'
        settings['anchor_controls']='DQN pure policies have no greedy fallback; categorical candidate replay includes local/greedy seeds. SharedAnchors frozen-policy controls use the same local, greedy, previous and knowledge seeds as Anchor-Joint, plus the learned argmax proposal. Inference and all evaluations are timed.'
    elif args.role=='final':
        parser.error('The final comparison requires --models-dir with matched trained learners')
    frozen=dict(settings=settings,implementation=implementation_signature())
    if args.freeze_protocol:
        if args.role!='final':
            parser.error('Freeze a declared final protocol, not a development run')
        path=Path(args.protocol) if args.protocol else output/'frozen_protocol.json'
        if path.exists():
            parser.error('Preserve the existing frozen protocol and select a new path')
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(frozen,indent=2))
        print(f'Protocol frozen at {path}; no experiment run.')
        return
    if args.role=='final' and (not args.protocol or json.loads(Path(args.protocol).read_text())!=frozen):
        parser.error('Freeze the exact configuration/code before accessing final evaluation seeds')
    started=time.perf_counter()
    with RunLock(output/'run.lock'):
        root=namespace(output/'checkpoints',settings)
        (output/'config.json').write_text(json.dumps(settings,indent=2))
        records=[]
        cases=[(seed,fade) for seed in seeds for fade in settings['fades']]
        for index,(seed,fade) in enumerate(cases,1):
            records.extend(run_case(settings,root,seed,fade,not args.fresh))
            pending=output/'slots.pending.csv'
            pd.DataFrame(records).to_csv(pending,index=False)
            pending.replace(output/'slots.csv')
            print(f'Completed {index}/{len(cases)} joint cases; elapsed {(time.perf_counter()-started):.1f} s',flush=True)
        summarize(pd.DataFrame(records),output)
        (output/'timing.json').write_text(json.dumps(dict(seconds=time.perf_counter()-started,cases=len(cases)),indent=2))
    print(f'Joint-protocol results: {output.resolve()}. Repeat the same command to resume.')


if __name__=='__main__':
    main()
