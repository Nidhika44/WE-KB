"""Checkpointed evaluation of the budgeted revision; preserves legacy outputs."""
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy.stats import t
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from we_kb_fpa.anytime import AnytimeConfig, AnytimeWEKBFPA
from we_kb_fpa.allocation import binary_greedy_allocation
from we_kb_fpa.checkpoint import Checkpoint, RunLock, namespace, implementation_signature
from we_kb_fpa.config import NetworkConfig, OptimizerConfig, CostConfig
from we_kb_fpa.dqn import DQNBaseline, DQNConfig
from we_kb_fpa.guarded_dqn import AnchoredFrozenDQN
from we_kb_fpa.droo import DROOAdapted
from we_kb_fpa.evaluation import delay_adjusted_metrics, fingerprint
from we_kb_fpa.knowledge_base import WirelessEdgeKnowledgeBase
from we_kb_fpa.learner_state import load_learner_state
from we_kb_fpa.simulation_executor import SimulationExecutor
from we_kb_fpa.we_kb_fpa import WEKBFPAOptimizer, OptimizationOutput


def variants(budgets):
    settings = {}
    for budget in budgets:
        base = AnytimeConfig(budget_ms=budget)
        settings.update({f"Anytime-{budget:g}ms": base,
                         f"No-KB-{budget:g}ms": replace(base, use_kb=False),
                         f"No-Warm-{budget:g}ms": replace(base, warm_start=False),
                         f"No-Event-{budget:g}ms": replace(base, event_trigger=False),
                         f"KB-Categorical-PSO-{budget:g}ms": replace(base, strategy="pso", event_trigger=False),
                         f"KB-Categorical-GA-{budget:g}ms": replace(base, strategy="ga", event_trigger=False),
                         f"Categorical-PSO-{budget:g}ms": replace(base, strategy="pso", use_kb=False, event_trigger=False),
                         f"Categorical-GA-{budget:g}ms": replace(base, strategy="ga", use_kb=False, event_trigger=False),
                         f"Anchor-Only-{budget:g}ms": replace(base,strategy="anchor"),
                         f"Binary-Anytime-{budget:g}ms": replace(base,action_scope="binary_nearest"),
                         f"Binary-GA-{budget:g}ms": replace(base,strategy="ga",event_trigger=False,action_scope="binary_nearest")})
        for strategy in ['fpa','ga']:
            settings[f'Deadline-{strategy.upper()}-{budget:g}ms'] = replace(base,strategy=strategy,event_trigger=False,deadline_aware=True)
            settings[f'DelayRank-{strategy.upper()}-{budget:g}ms'] = replace(base,strategy=strategy,event_trigger=False,deadline_aware=True,stopping_enabled=False)
            settings[f'Binary-Deadline-{strategy.upper()}-{budget:g}ms'] = replace(base,strategy=strategy,event_trigger=False,deadline_aware=True,action_scope='binary_nearest')
    return settings


def initialize(config, seed, users):
    net = NetworkConfig(num_users=users, num_edge_servers=3)
    opt = OptimizerConfig(population_size=20, max_iterations=30)
    sim = SimulationExecutor(net_config=net, opt_config=opt, seed=seed)
    configured = ({name:AnytimeConfig(**value) for name,value in config['optimizer_variants'].items()}
                  if config.get('optimizer_variants') else variants(config['budgets_ms']))
    optimizers = {name: AnytimeWEKBFPA(cfg, net_config=net, seed=seed)
                  for name, cfg in configured.items() if users==8 or cfg.action_scope=='all'}
    knowledge = {name: WirelessEdgeKnowledgeBase(net_config=net) for name,optimizer in optimizers.items() if optimizer.config.use_kb}
    knowledge['Current-WE-Joint'] = WirelessEdgeKnowledgeBase(net_config=net)
    agents = {}
    qualities = {}
    model_seed = config['model_seeds'][config['seeds'].index(seed) % len(config['model_seeds'])] if config['model_seeds'] else None
    if users == 8 and model_seed is not None:
        base = Path(config['models_dir'])
        selected = json.loads((base/'config.json').read_text())['selected_dqn']
        if config.get('learner_state','full')=='weights' and 'reward_transform' not in selected:
            selected['reward_transform']='log'
        dqn = DQNBaseline(net_config=net, seed=seed, config=DQNConfig(**selected))
        droo = DROOAdapted(net_config=net, seed=seed)
        for agent, prefix in [(dqn, 'dqn'), (droo, 'droo')]:
            if config.get('learner_state','full')=='full':
                loaded,metadata = load_learner_state(base/f'{prefix}_seed_{model_seed}_budget_{config["model_budget"]}.agent.pkl')
                if metadata['seed']!=model_seed or metadata['training_budget']!=config['model_budget'] or metadata['algorithm']!=prefix:
                    raise ValueError('Full-state model provenance does not match requested seed/budget')
                if prefix=='dqn':
                    dqn = loaded
                else:
                    droo = loaded
                qualities[prefix] = metadata['quality']
            else:
                with np.load(base/f'{prefix}_seed_{model_seed}_budget_{config["model_budget"]}.npz') as data:
                    agent.weights = [data[f'weight_{i}'].copy() for i in range(6)]
                    if prefix == 'dqn':
                        agent.target = [data[f'target_{i}'].copy() for i in range(6)]
                        agent.epsilon = float(data['epsilon'])
        suffix = 'FullState' if config.get('learner_state','full')=='full' else 'Reloaded'
        agents = {'DQN-Frozen':copy.deepcopy(dqn),f'DQN-Online-{suffix}':dqn,
                  f'DQN-Online-Greedy-{suffix}':copy.deepcopy(dqn),f'DROO-Adapted-Online-{suffix}':droo}
        for budget in config['budgets_ms']:
            agents[f'DQN-Frozen-AnchorSafe-{budget:g}ms']=AnchoredFrozenDQN(agents['DQN-Frozen'],budget,net,seed)
    return dict(next_slot=0, sim=sim, optimizers=optimizers, knowledge=knowledge,
                agents=agents, qualities=qualities,records=[], model_seed=model_seed, complete=False)


def run_case(config, checkpoint_root, study, seed, users, fade):
    identity = dict(study=study, seed=seed, users=users, fade=fade)
    checkpoint = Checkpoint(checkpoint_root, f'{study}-{seed}-{users}-{fade:g}', identity, config['resume'])
    state = checkpoint.load()
    if state and state['complete']:
        return state['records']
    if state is None:
        state = initialize(config, seed, users)
    horizon = config['slots'] if study == 'shock' else config['scale_slots']
    last_saved = time.perf_counter()
    for slot in range(state['next_slot'], horizon):
        sim = state['sim']
        if slot:
            sim.env.update_server_queues(sim.sim_cfg.time_step)
        tasks, sinr, distances, variance, _ = sim.sample_slot(slot, fading_db=fade if slot >= horizon//2 else 0.)
        variance = float(np.var(np.log1p(sinr)))
        args = tasks, sim.env.users, sim.env.edge_servers, sim.env.cloud, sinr, distances
        before = fingerprint(args)
        names = ['Greedy-Uniform'] + list(state['agents']) + list(state['optimizers'])
        if users==8:
            names.append('Binary-Greedy-Uniform')
        if config['include_current']:
            names.append('Current-WE-Joint')
        offset = slot % len(names)
        for name in names[offset:] + names[:offset]:
            diagnostic = {}
            decision_seconds = None
            learning_seconds = 0.
            if name == 'Greedy-Uniform':
                out = sim.baseline_runner.run_greedy_latency(*args)
            elif name=='Binary-Greedy-Uniform':
                tick=time.perf_counter()
                allocation=binary_greedy_allocation(*args,sim.net_cfg)
                result=sim.baseline_runner.cost_engine.evaluate_allocation(*args[:4],*allocation,*args[4:])
                out=OptimizationOutput(result.total_cost,*allocation,[result.total_cost],time.perf_counter()-tick,1,result,name)
            elif name in state['agents']:
                kwargs = dict(training=not name.startswith('DQN-Frozen'))
                if name.startswith('DQN-Online-Greedy'):
                    kwargs['explore'] = False
                out = state['agents'][name].run(*args, **kwargs)
                diagnostic=getattr(state['agents'][name],'last_diagnostics',{}).copy()
                decision_seconds = state['agents'][name].last_decision_seconds
                learning_seconds = state['agents'][name].last_learning_seconds
            else:
                tick = time.perf_counter()
                if name == 'Current-WE-Joint':
                    out = WEKBFPAOptimizer(sim.opt_cfg, net_config=sim.net_cfg, seed=seed+slot).optimize(
                        *args, channel_variance=variance, knowledge_base=state['knowledge'][name])
                else:
                    optimizer = state['optimizers'][name]
                    out = optimizer.optimize(*args, channel_variance=variance, knowledge_base=state['knowledge'].get(name))
                    diagnostic = optimizer.last_diagnostics.copy()
                decision_seconds = time.perf_counter()-tick
                kb = state['knowledge'].get(name)
                if kb is not None:
                    kb.add_experience(sinr, tasks, distances, out.best_targets, out.best_cpu, out.best_bw, out.best_cost)
                # Charge retrieval, optimizer construction, and memory insertion to the decision.
                out.wall_clock_time_sec = time.perf_counter()-tick
                learning_seconds = out.wall_clock_time_sec-decision_seconds
                if 'budget_ms' in diagnostic:
                    diagnostic['budget_overrun'] = out.wall_clock_time_sec*1000 > diagnostic['budget_ms']
            if fingerprint(args) != before:
                raise RuntimeError(f'{name} mutated the shared snapshot')
            result = out.evaluation_result
            if decision_seconds is None:
                decision_seconds = out.wall_clock_time_sec
            decision_only = {'decision_only_'+key:value for key,value in
                             delay_adjusted_metrics(out,CostConfig(),decision_seconds).items()}
            source_quality = state['qualities'] or config.get('source_quality', {}).get(str(state['model_seed']), {})
            learner_quality = source_quality.get('dqn' if name.startswith('DQN') else 'droo', {}) if name in state['agents'] else {}
            state['records'].append(dict(**identity, slot=slot, phase='pre' if slot<horizon//2 else 'post',
                algorithm=name, model_seed=state['model_seed'], cost=out.best_cost,
                source_nominal_quality_pass=learner_quality.get('nominal_quality_pass'),
                source_nominal_cost_ratio=learner_quality.get('nominal_cost_ratio'),
                runtime_ms=out.wall_clock_time_sec*1000, violation_pct=result.deadline_violation_rate,
                decision_ms=decision_seconds*1000,learning_or_memory_ms=learning_seconds*1000,
                energy_j=result.average_energy, allocation='joint' if name=='Current-WE-Joint' else 'uniform',
                action_space='binary_nearest_edge' if name.startswith(('Binary-','DROO-')) else 'local_all_edges_cloud',
                timing_contended=config['workers']>1, snapshot_hash=before,
                **delay_adjusted_metrics(out, CostConfig()), **decision_only, **diagnostic))
        state['next_slot'] = slot+1
        if (slot+1)%config['checkpoint_every'] == 0 or time.perf_counter()-last_saved >= 60:
            checkpoint.save(state)
            last_saved = time.perf_counter()
    checkpoint.save(dict(complete=True, records=state['records']), compact=True)
    return state['records']


METRICS = ['cost', 'delay_adjusted_cost', 'runtime_ms', 'violation_pct', 'delay_adjusted_violation_pct', 'energy_j',
           'decision_ms','learning_or_memory_ms','decision_only_delay_adjusted_cost','decision_only_delay_adjusted_violation_pct']
KEYS = ['study', 'users', 'fade', 'phase', 'algorithm']


def summarize(frame, output):
    means = frame.groupby(KEYS+['seed'])[METRICS].mean().reset_index()
    rows = []
    for keys, group in means.groupby(KEYS):
        for metric in METRICS:
            values = group[metric]
            n = len(values)
            rows.append(dict(zip(KEYS, keys), metric=metric, mean=float(values.mean()),
                             std=float(values.std()) if n>1 else None, seeds=n,
                             ci95_margin=float(t.ppf(.975,n-1)*values.std()/np.sqrt(n)) if n>1 else None))
    summary = pd.DataFrame(rows)
    summary.to_csv(output/'summary.csv', index=False)
    paired = []
    join = ['study', 'users', 'fade', 'phase', 'seed']
    baselines = [name for name in frame.algorithm.unique() if name.startswith(('Greedy-', 'Anchor-', 'No-Event-','DelayRank-','Binary-DelayRank-','Binary-Greedy-','Binary-GA-','Categorical-GA-',
                 'KB-Categorical-GA-','Categorical-PSO-','KB-Categorical-PSO-','DQN-','DROO-'))]
    for baseline in baselines:
        reference = means[means.algorithm==baseline][join+METRICS]
        proposed = (means.algorithm.str.startswith(('Binary-Anytime-','Binary-Deadline-FPA-','Binary-Deadline-GA-'))
                    if baseline.startswith(('DROO-','Binary-')) else means.algorithm.str.startswith(('Anytime-','Deadline-FPA-','Deadline-GA-')))
        if baseline.endswith('ms'):
            proposed &= means.algorithm.str.endswith('-'+baseline.rsplit('-',1)[1])
        joined = means[proposed].merge(reference, on=join, suffixes=('', '_baseline'))
        for keys, group in joined.groupby(KEYS):
            for metric in METRICS:
                difference = group[metric]-group[f'{metric}_baseline']
                n = len(difference)
                paired.append(dict(zip(KEYS,keys), baseline=baseline, metric=metric,
                                   mean_difference=float(difference.mean()), seeds=n,
                                   ci95_margin=float(t.ppf(.975,n-1)*difference.std()/np.sqrt(n)) if n>1 else None))
    pd.DataFrame(paired).to_csv(output/'paired_effects.csv', index=False)
    diagnostics = frame.groupby(KEYS).agg(runtime_mean_ms=('runtime_ms','mean'), runtime_max_ms=('runtime_ms','max'),
                                        budget_overrun_rate=('budget_overrun','mean'), search_trigger_rate=('search_triggered','mean'),
                                        mean_evaluations=('evaluations','mean')).reset_index()
    diagnostics.to_csv(output/'diagnostics.csv', index=False)
    plot_results(frame, summary, output)
    return summary


def plot_results(frame, summary, output):
    figure_dir = output/'figures'
    figure_dir.mkdir(exist_ok=True)
    primary = summary.algorithm.str.startswith(('Anytime-','Deadline-','DelayRank-', 'Current-', 'Greedy-', 'DQN-', 'DROO-'))
    for study in frame.study.unique():
        selected = summary[(summary.study==study)&(summary.phase=='post')&primary]
        for fade in selected.fade.unique():
            fig, axes = plt.subplots(1, 3, figsize=(15,4))
            for ax, metric, label in zip(axes, ['cost','delay_adjusted_violation_pct','runtime_ms'],
                                        ['System cost','Violations including computation (%)','Computation (ms)']):
                data = selected[(selected.fade==fade)&(selected.metric==metric)]
                for algorithm, group in data.groupby('algorithm'):
                    x = group.users if study=='scalability' else [algorithm]
                    y = group['mean'].to_numpy()
                    error = group.ci95_margin.fillna(0).to_numpy()
                    lower = np.minimum(error, np.maximum(0,y-1e-12))
                    ax.errorbar(x, y, yerr=np.vstack([lower,error]), marker='o', label=algorithm, capsize=2)
                ax.set_ylabel(label)
                if metric=='cost':
                    ax.set_yscale('symlog',linthresh=1.)
                elif metric=='runtime_ms':
                    ax.set_yscale('log')
                else:
                    ax.set_ylim(0,100)
                if study=='shock':
                    ax.tick_params(axis='x',labelrotation=90,labelsize=5)
                else:
                    ax.set_xlabel('Users')
                ax.grid(alpha=.2)
            if study=='scalability':
                axes[-1].legend(fontsize=5)
            fig.tight_layout()
            for suffix in ['pdf','png']:
                fig.savefig(figure_dir/f'{study}_fade_{fade:g}.{suffix}',dpi=250)
            plt.close(fig)
    for budget in sorted(frame.budget_ms.dropna().unique()):
        selected = summary[(summary.study=='shock')&(summary.phase=='post')&
                           (summary.algorithm.str.endswith(f'-{budget:g}ms')|summary.algorithm.isin(['Greedy-Uniform']))]
        for fade in selected.fade.unique():
            fig, axes = plt.subplots(1,2,figsize=(12,5))
            for ax,metric in zip(axes,['cost','delay_adjusted_violation_pct']):
                data = selected[(selected.fade==fade)&(selected.metric==metric)].sort_values('algorithm')
                y = data['mean'].to_numpy()
                error = data.ci95_margin.fillna(0).to_numpy()
                ax.errorbar(data.algorithm,y,yerr=np.vstack([np.minimum(error,np.maximum(0,y-1e-12)),error]),fmt='o',capsize=2)
                ax.tick_params(axis='x',labelrotation=90,labelsize=7)
                ax.set_ylabel('System cost' if metric=='cost' else 'Violations including computation (%)')
                if metric!='cost':
                    ax.set_ylim(0,100)
            fig.tight_layout()
            for suffix in ['pdf','png']:
                fig.savefig(figure_dir/f'ablation_{budget:g}ms_fade_{fade:g}.{suffix}',dpi=250)
            plt.close(fig)
    shock = frame[frame.study=='shock']
    for fade in shock.fade.unique():
        fig, axes = plt.subplots(1,3,figsize=(15,4))
        for ax, metric in zip(axes,['cost','delay_adjusted_violation_pct','runtime_ms']):
            for name, group in shock[shock.fade==fade].groupby('algorithm'):
                if name.startswith(('Anytime-','Deadline-','DelayRank-', 'Current-', 'Greedy-', 'DQN-', 'DROO-')):
                    stats = group.groupby('slot')[metric].agg(['mean','std','count'])
                    error = t.ppf(.975,np.maximum(1,stats['count']-1))*stats['std'].fillna(0)/np.sqrt(stats['count'])
                    ax.plot(stats.index,stats['mean'],label=name)
                    ax.fill_between(stats.index,np.maximum(1e-12,stats['mean']-error),stats['mean']+error,alpha=.1)
            ax.axvline(shock.slot.max()/2,color='black',linestyle='--')
            ax.set(xlabel='Slot',ylabel=metric)
            if metric=='cost':
                ax.set_yscale('symlog',linthresh=1.)
            elif metric=='runtime_ms':
                ax.set_yscale('log')
            else:
                ax.set_ylim(0,100)
        axes[0].legend(fontsize=5)
        fig.tight_layout()
        for suffix in ['pdf','png']:
            fig.savefig(figure_dir/f'adaptation_fade_{fade:g}.{suffix}',dpi=250)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rigorous', action='store_true')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--study', choices=['all','shock','scalability'], default='all')
    parser.add_argument('--seeds', type=int, nargs='+')
    parser.add_argument('--slots', type=int)
    parser.add_argument('--scale-slots', type=int)
    parser.add_argument('--users', type=int, nargs='+')
    parser.add_argument('--budgets-ms', type=float, nargs='+', default=[10.,25.,50.])
    parser.add_argument('--optimizer-configs',help='Explicit development-selected optimizer settings JSON; recorded in the frozen protocol')
    parser.add_argument('--fading-values', type=float, nargs='+')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--checkpoint-every', type=int, default=10)
    parser.add_argument('--skip-current', action='store_true', help='Omit expensive legacy optimizer; changes experiment scope')
    parser.add_argument('--models-dir', default='output/learners_corrected')
    parser.add_argument('--model-seeds', type=int, nargs='+')
    parser.add_argument('--model-budget', type=int, default=80000)
    parser.add_argument('--learner-state',choices=['full','weights'],default='full',help='Weights-only is a separately labelled reset experiment')
    parser.add_argument('--require-nominal-quality',action='store_true',help='Reject the entire comparison if a source learner fails its held-out gate; never drop individual seeds')
    parser.add_argument('--evaluation-role',choices=['development','final'],default='development')
    parser.add_argument('--freeze-protocol',action='store_true',help='Write the final protocol without running any experiment')
    parser.add_argument('--protocol',help='Frozen protocol JSON required for final evaluation')
    parser.add_argument('--no-drl', action='store_true', help='Explicitly omit learning baselines')
    parser.add_argument('--output-dir', default='output/revision_corrected')
    parser.add_argument('--fresh', action='store_true')
    parser.add_argument('--plots-only', action='store_true')
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True,exist_ok=True)
    if (output/'config.json').exists() and json.loads((output/'config.json').read_text()).get('protocol_schema')!=2:
        parser.error('Preserve legacy/development results: choose a new --output-dir for the corrected protocol')
    if args.plots_only:
        with RunLock(output/'run.lock'):
            summarize(pd.read_csv(output/'slots.csv'), output)
        return
    if args.quick and args.rigorous:
        parser.error('Choose either --quick or --rigorous')
    first_seed = 30001 if args.evaluation_role=='final' else 10001
    seeds = args.seeds or list(range(first_seed,first_seed+(2 if args.quick else 30 if args.rigorous else 5)))
    if args.evaluation_role=='final' and (args.workers!=1 or args.learner_state!='full' or any(s<30000 for s in seeds)):
        parser.error('Final evaluation requires serial timing, full learner state, and reserved seeds >=30000')
    require_quality = args.require_nominal_quality or args.evaluation_role=='final'
    slots = args.slots or (6 if args.quick else 100 if args.rigorous else 20)
    users = args.users or ([10,20] if args.quick else [10,20,50,100,200,500] if args.rigorous else [10,50,100])
    fades = args.fading_values or ([35.] if args.quick else [15.,25.,35.])
    if (slots<6 or slots%2 or args.workers<1 or args.checkpoint_every<1 or len(set(seeds))!=len(seeds)
            or len(set(users))!=len(users)
            or any(s<0 or s>=100000 for s in seeds) or any(n<1 for n in users)
            or any(not np.isfinite(f) or f<=0 for f in fades)):
        parser.error('Use even slots >=6, unique seeds in [0,100000), positive users, fades, workers and checkpoint interval')
    try:
        for b in args.budgets_ms:
            AnytimeConfig(budget_ms=b)
        optimizer_settings=variants(sorted(set(args.budgets_ms)))
        if args.optimizer_configs:
            optimizer_settings={name:AnytimeConfig(**value) for name,value in json.loads(Path(args.optimizer_configs).read_text()).items()}
            if not optimizer_settings or any(cfg.budget_ms not in args.budgets_ms for cfg in optimizer_settings.values()):
                parser.error('Optimizer settings must be nonempty and match --budgets-ms')
    except ValueError as error:
        parser.error(str(error))
    models = [] if args.no_drl else (args.model_seeds or list(range(42,42+len(seeds))))
    if args.evaluation_role=='final' and models and (len(models)!=len(seeds) or len(set(models))!=len(models)):
        parser.error('Final evaluation requires one distinct trained model seed per evaluation seed')
    hashes = {}
    source_quality = {}
    if models and set(models)&set(seeds):
        parser.error('Evaluation seeds must differ from the saved model seed identifiers')
    if models and args.study!='scalability':
        source_config=json.loads((Path(args.models_dir)/'config.json').read_text()) if (Path(args.models_dir)/'config.json').exists() else {}
        if args.evaluation_role=='final' and (source_config.get('selected_dqn',{}).get('reward_transform')!='linear' or
                                              source_config.get('selected_dqn',{}).get('gamma')!=1.):
            parser.error('Final raw-cost comparison requires a linear-cost, gamma=1 trained DQN; legacy log-reward models are development evidence only')
        extension = 'agent.pkl' if args.learner_state=='full' else 'npz'
        paths = [Path(args.models_dir)/'config.json'] + [Path(args.models_dir)/f'{prefix}_seed_{seed}_budget_{args.model_budget}.{extension}'
                 for seed in models for prefix in ['dqn','droo']]
        for path in paths:
            if not path.is_file():
                parser.error(f'Missing trained model: {path}. Train full-state models with python -m experiments.prepare_learners first; old weights cannot reconstruct replay/Adam.')
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        quality_path = Path(args.models_dir)/'quality_gate.json'
        if quality_path.exists():
            hashes[str(quality_path)] = hashlib.sha256(quality_path.read_bytes()).hexdigest()
            source_quality = {str(row['seed']): row for row in json.loads(quality_path.read_text())
                              if row['training_budget']==args.model_budget and row['seed'] in models}
        if require_quality:
            failures = [seed for seed in models if str(seed) not in source_quality or
                        not all(source_quality[str(seed)][kind]['nominal_quality_pass'] for kind in ['dqn','droo'])]
            if failures:
                parser.error(f'Nominal quality gate failed or missing for model seeds {failures}. Improve training on development data; no seeds were dropped.')
    else:
        models = []
    config = dict(protocol_schema=2,evaluation_role=args.evaluation_role,seeds=seeds,slots=slots,scale_slots=args.scale_slots or (2 if args.quick else 10),
                  users=users,fades=sorted(set([0.]+fades)),budgets_ms=sorted(set(args.budgets_ms)),
                  workers=args.workers,checkpoint_every=args.checkpoint_every,include_current=not args.skip_current,
                  models_dir=str(Path(args.models_dir).resolve()),model_seeds=models,model_budget=args.model_budget,
                  model_hashes=hashes,source_quality=source_quality,study=args.study,resume=not args.fresh,
                  learner_state=args.learner_state,require_nominal_quality=require_quality,
                  network=asdict(NetworkConfig(num_users=8,num_edge_servers=3)), cost=asdict(CostConfig()),
                  optimizer_variants={name:asdict(value) for name,value in optimizer_settings.items()},
                  allocator='Uniform CPU per compute target; shared uniform radio per access edge. Not an analytical optimum.',
                  channel_guidance='Variance of log1p current observed SINR; no access to simulator-private channel history or future shock labels.',
                  stopping_rule='Shared FPA/GA heuristic: rolling improvement plus optimism bonus versus exact incumbent waiting cost. Eight retained incumbents; no optimal-stopping guarantee.',
                  timing='Both decision-only and serialized decision+learning/memory snapshot sensitivities; no asynchronous queue model. Soft budgets, overruns reported.',
                  learning='Full training state restored by default. Exploring and greedy-deployment online DQN reported separately. Weights-only reset explicitly labelled.',
                  learning_control='DQN-Frozen-AnchorSafe explicitly combines the same common anchors with frozen predictions; the pure policy is also reported without overrides.',
                  novelty='Categorical adaptation is an experimental variant, not a claim of novel or globally optimal operators.')
    if config['scale_slots']<1:
        parser.error('--scale-slots must be positive')
    protocol_config = {k:v for k,v in config.items() if k!='resume'}
    expected_protocol = dict(config=protocol_config,implementation=implementation_signature())
    if args.freeze_protocol:
        if args.evaluation_role!='final':
            parser.error('--freeze-protocol is for a declared final evaluation')
        destination = Path(args.protocol) if args.protocol else output/'frozen_protocol.json'
        destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists():
            parser.error('Frozen protocol already exists; preserve it and choose a new path for a new protocol')
        with RunLock(output/'run.lock'):
            destination.write_text(json.dumps(expected_protocol,indent=2))
        print(f'Protocol frozen at {destination.resolve()}; no evaluation was run.')
        return
    if args.evaluation_role=='final':
        if not args.protocol or json.loads(Path(args.protocol).read_text())!=expected_protocol:
            parser.error('Final configuration/code must match --protocol exactly. Freeze the protocol before accessing final trajectories.')
    if args.workers>1:
        print('Parallel workers share CPU: timings and budgeted quality reflect contention. Use --workers 1 for paper timing.',flush=True)
    cases = []
    for seed in seeds:
        if args.study in ('all','shock'):
            cases.extend(('shock',seed,8,fade) for fade in config['fades'])
        if args.study in ('all','scalability'):
            cases.extend(('scalability',seed,n,0.) for n in users)
    started = time.perf_counter()
    with RunLock(output/'run.lock'):
        checkpoint_root = namespace(output/'checkpoints',{k:v for k,v in config.items() if k!='resume'})
        (output/'config.json').write_text(json.dumps(config,indent=2))
        rows = []
        if args.workers==1:
            for index, case in enumerate(cases,1):
                rows.extend(run_case(config,checkpoint_root,*case))
                temporary = output/'slots.pending.csv'
                pd.DataFrame(rows).to_csv(temporary,index=False)
                temporary.replace(output/'slots.csv')
                elapsed = time.perf_counter()-started
                print(f'Completed {index}/{len(cases)}: {case}; elapsed {elapsed/60:.1f} min',flush=True)
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(run_case,config,checkpoint_root,*case) for case in cases]
                for index, future in enumerate(as_completed(futures),1):
                    rows.extend(future.result())
                    temporary = output/'slots.pending.csv'
                    pd.DataFrame(rows).to_csv(temporary,index=False)
                    temporary.replace(output/'slots.csv')
                    print(f'Completed {index}/{len(cases)} cases',flush=True)
        summarize(pd.DataFrame(rows),output)
        (output/'timing.json').write_text(json.dumps(dict(seconds=time.perf_counter()-started, cases=len(cases),
                                                        records=len(rows), resumed=not args.fresh),indent=2))
    print(f'Results and figures: {output.resolve()}. Repeat the same command to resume.',flush=True)


if __name__=='__main__':
    main()
