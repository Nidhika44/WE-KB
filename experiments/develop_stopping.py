"""Equal-grid development comparison of shared stopping for FPA and GA.

Candidate selection uses calibration trajectories only. The check split is
development evidence, not the reserved final evaluation.
"""
import argparse
from dataclasses import asdict,replace
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy.stats import t
from we_kb_fpa.anytime import AnytimeConfig,AnytimeWEKBFPA
from we_kb_fpa.checkpoint import Checkpoint,RunLock,namespace
from we_kb_fpa.config import CostConfig
from we_kb_fpa.evaluation import snapshots,fingerprint,delay_adjusted_metrics
from we_kb_fpa.knowledge_base import WirelessEdgeKnowledgeBase


def candidate_grid(budget):
    result={}
    for strategy in ['fpa','ga']:
        for population in [8,12,20]:
            base=AnytimeConfig(strategy=strategy,budget_ms=budget,population=population,event_trigger=False)
            for patience in [20,40,80]:
                result[f'fixed-{strategy}-p{population}-q{patience}']=replace(base,patience=patience)
                result[f'ranking-{strategy}-p{population}-q{patience}']=replace(base,patience=patience,deadline_aware=True,stopping_enabled=False)
            for optimism in [0.,1.,2.]:
                result[f'deadline-{strategy}-p{population}-o{optimism:g}']=replace(base,deadline_aware=True,stop_optimism=optimism)
    return result


def run_split(configs,seeds,slots,fades,root,role):
    records=[]
    for seed in seeds:
        for fade in fades:
            checkpoint=Checkpoint(root,f'{role}-{seed}-{fade:g}',dict(role=role,seed=seed,fade=fade,
                                  candidates={name:asdict(cfg) for name,cfg in configs.items()}))
            saved=checkpoint.load()
            if saved is not None:
                records.extend(saved)
                continue
            stream=snapshots(seed,slots,fading_db=fade,midpoint=slots//2)
            optimizers={name:AnytimeWEKBFPA(cfg,net_config=stream.sim.net_cfg,seed=seed) for name,cfg in configs.items()}
            knowledge={name:WirelessEdgeKnowledgeBase(net_config=stream.sim.net_cfg) for name in configs}
            rows=[]
            for slot,args,_,_ in stream:
                expected=fingerprint(args)
                variance=float(np.var(np.log1p(args[4])))
                names=list(configs)
                offset=slot%len(names)
                for name in names[offset:]+names[:offset]:
                    optimizer=optimizers[name]
                    started=time.perf_counter()
                    out=optimizer.optimize(*args,channel_variance=variance,knowledge_base=knowledge[name])
                    knowledge[name].add_experience(args[4],args[0],args[5],out.best_targets,out.best_cpu,out.best_bw,out.best_cost)
                    out.wall_clock_time_sec=time.perf_counter()-started
                    if fingerprint(args)!=expected:
                        raise RuntimeError('Candidate mutated the shared snapshot')
                    cfg=configs[name]
                    rows.append(dict(role=role,seed=seed,fade=fade,slot=slot,phase='pre' if slot<slots//2 else 'post',
                        candidate=name,arm=('deadline-' if cfg.deadline_aware and cfg.stopping_enabled else 'ranking-' if cfg.deadline_aware else 'fixed-')+cfg.strategy,
                        cost=out.best_cost,runtime_ms=out.wall_clock_time_sec*1000,snapshot_hash=expected,
                        **delay_adjusted_metrics(out,CostConfig()),**optimizer.last_diagnostics))
            checkpoint.save(rows,compact=True)
            records.extend(rows)
            print(f'{role}: completed seed {seed}, fade {fade:g}',flush=True)
    return pd.DataFrame(records)


def summarize(frame):
    post=frame[frame.phase=='post']
    metrics=['cost','delay_adjusted_cost','delay_adjusted_violation_pct','runtime_ms','evaluations']
    per_seed=post.groupby(['seed','candidate','arm'])[metrics].mean().reset_index()
    summary=per_seed.groupby(['candidate','arm'])[metrics].mean().reset_index()
    stop_rates=post.assign(value_stop=post.stop_reason.eq('estimated_gain_below_waiting_cost')).groupby('candidate').value_stop.mean()
    summary['value_stop_rate']=summary.candidate.map(stop_rates)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--budget-ms',type=float,default=10.)
    parser.add_argument('--slots',type=int,default=12)
    parser.add_argument('--output-dir',default='output/stopping_development')
    args=parser.parse_args()
    if args.slots<6 or args.slots%2:
        parser.error('Use an even slot count >=6')
    grid=candidate_grid(args.budget_ms)
    output=Path(args.output_dir)
    output.mkdir(parents=True,exist_ok=True)
    settings=dict(role='development only',fit_seeds=[10001,10002],check_seeds=[10003,10004],
                  slots=args.slots,fades=[0.,15.,35.],grid={name:asdict(cfg) for name,cfg in grid.items()},
                  selection='Mean post-phase delay-adjusted cost; equal seed/scenario weights; 9 candidates per arm',
                  rule='Exact fixed-incumbent waiting cost versus rolling empirical improvement plus heuristic optimism bonus; no optimality guarantee')
    settings['incumbents']='At most eight delay-ranked observed allocations with raw cost <= current greedy; shared by both strategies'
    started=time.perf_counter()
    with RunLock(output/'run.lock'):
        root=namespace(output/'checkpoints',settings)
        (output/'config.json').write_text(json.dumps(settings,indent=2))
        fit=run_split(grid,settings['fit_seeds'],args.slots,settings['fades'],root,'calibration')
        fit.to_csv(output/'calibration_slots.csv',index=False)
        fit_summary=summarize(fit)
        fit_summary.to_csv(output/'calibration_summary.csv',index=False)
        selected=fit_summary.loc[fit_summary.groupby('arm').delay_adjusted_cost.idxmin()]
        selected.to_csv(output/'selected_on_calibration.csv',index=False)
        check_configs={name:grid[name] for name in selected.candidate}
        check_configs['anchor-only']=AnytimeConfig(strategy='anchor',budget_ms=args.budget_ms,event_trigger=False)
        for strategy in ['fpa','ga']:
            chosen=grid[selected[selected.arm==f'deadline-{strategy}'].candidate.iloc[0]]
            check_configs[f'matched-ranking-{strategy}']=replace(chosen,stopping_enabled=False)
            check_configs[f'matched-fixed-{strategy}']=replace(chosen,deadline_aware=False)
        check=run_split(check_configs,settings['check_seeds'],args.slots,settings['fades'],root,'development_check')
        check.to_csv(output/'check_slots.csv',index=False)
        report=summarize(check)
        report.to_csv(output/'check_summary.csv',index=False)
        means=check[check.phase=='post'].groupby(['seed','fade','candidate'])[
            ['delay_adjusted_cost','delay_adjusted_violation_pct','runtime_ms']].mean().reset_index()
        effects=[]
        for strategy in ['fpa','ga']:
            chosen=selected[selected.arm==f'deadline-{strategy}'].candidate.iloc[0]
            baseline_names=[selected[selected.arm==f'fixed-{strategy}'].candidate.iloc[0],
                            selected[selected.arm==f'ranking-{strategy}'].candidate.iloc[0],
                            f'matched-ranking-{strategy}',f'matched-fixed-{strategy}','anchor-only']
            for baseline in baseline_names:
                paired=means[means.candidate==chosen].merge(means[means.candidate==baseline],on=['seed','fade'],suffixes=('_aware','_baseline'))
                for metric in ['delay_adjusted_cost','delay_adjusted_violation_pct','runtime_ms']:
                    differences=(paired[metric+'_aware']-paired[metric+'_baseline']).groupby(paired.seed).mean()
                    n=len(differences)
                    effects.append(dict(strategy=strategy,baseline=baseline,metric=metric,
                                        mean_difference=float(differences.mean()),independent_seeds=n,
                                        ci95_margin=float(t.ppf(.975,n-1)*differences.std()/np.sqrt(n)) if n>1 else None))
        pd.DataFrame(effects).to_csv(output/'check_effects.csv',index=False)
        canonical={}
        for name,cfg in check_configs.items():
            if name.startswith('matched-'):
                continue
            if cfg.strategy=='anchor':
                label=f'Anchor-Only-{args.budget_ms:g}ms'
            elif cfg.deadline_aware and cfg.stopping_enabled:
                label=f'Deadline-{cfg.strategy.upper()}-{args.budget_ms:g}ms'
            elif cfg.deadline_aware:
                label=f'DelayRank-{cfg.strategy.upper()}-{args.budget_ms:g}ms'
            elif cfg.strategy=='fpa':
                label=f'No-Event-{args.budget_ms:g}ms'
            else:
                label=f'KB-Categorical-GA-{args.budget_ms:g}ms'
            canonical[label]=asdict(cfg)
            if cfg.strategy!='anchor':
                binary_label=(f'Binary-Deadline-{cfg.strategy.upper()}-{args.budget_ms:g}ms' if cfg.deadline_aware and cfg.stopping_enabled else
                              f'Binary-DelayRank-{cfg.strategy.upper()}-{args.budget_ms:g}ms' if cfg.deadline_aware else
                              f'Binary-Anytime-{args.budget_ms:g}ms' if cfg.strategy=='fpa' else f'Binary-GA-{args.budget_ms:g}ms')
                canonical[binary_label]=asdict(replace(cfg,action_scope='binary_nearest'))
        (output/'selected_configs.json').write_text(json.dumps(canonical,indent=2))
        (output/'timing.json').write_text(json.dumps(dict(seconds=time.perf_counter()-started),indent=2))
        print('Development check; parameters selected on calibration only:')
        print(report.round(4).to_string(index=False))
        print(f'Results: {output.resolve()}. No final evaluation was run.')


if __name__=='__main__':
    main()
