"""Development-only operator probe with equal candidate grids and common anchors."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import pandas as pd
import numpy as np
from we_kb_fpa.anytime import AnytimeConfig, AnytimeWEKBFPA
from we_kb_fpa.evaluation import snapshots, delay_adjusted_metrics
from we_kb_fpa.config import CostConfig
from we_kb_fpa.checkpoint import RunLock, namespace, Checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick',action='store_true')
    parser.add_argument('--budget-ms',type=float,default=10.)
    parser.add_argument('--output-dir',default='output/operator_development')
    args = parser.parse_args()
    configs = {f'{strategy}-pop{population}':AnytimeConfig(budget_ms=args.budget_ms,population=population,
               use_kb=False,warm_start=False,event_trigger=False,strategy=strategy)
               for strategy in ['fpa','ga','pso'] for population in [8,12,20]}
    output = Path(args.output_dir)
    output.mkdir(parents=True,exist_ok=True)
    settings = dict(role='development only; common greedy/all-local anchors; no historical seeding',
                    seeds=[10001,10002] if args.quick else [10001,10002,10003,10004],
                    slots=8 if args.quick else 20,fades=[0.,35.] if args.quick else [0.,15.,35.],
                    candidates={name:asdict(cfg) for name,cfg in configs.items()},
                    selection_metric='mean delay-adjusted cost across development cases',
                    channel_guidance='Variance of log1p current observed SINR; no simulator-private history')
    rows = []
    with RunLock(output/'run.lock'):
        root = namespace(output/'checkpoints',settings)
        (output/'config.json').write_text(json.dumps(settings,indent=2))
        for seed in settings['seeds']:
            for fade in settings['fades']:
                checkpoint = Checkpoint(root,f'{seed}-{fade:g}',dict(seed=seed,fade=fade))
                saved = checkpoint.load()
                if saved is not None:
                    rows.extend(saved)
                    continue
                records = []
                optimizers = {name:AnytimeWEKBFPA(cfg,seed=seed) for name,cfg in configs.items()}
                for slot,arguments,variance,_ in snapshots(seed,settings['slots'],fading_db=fade,midpoint=settings['slots']//2):
                    variance=float(np.var(np.log1p(arguments[4])))
                    names = list(optimizers)
                    offset = slot%len(names)
                    for name in names[offset:]+names[:offset]:
                        optimizer = optimizers[name]
                        result = optimizer.optimize(*arguments,channel_variance=variance)
                        records.append(dict(seed=seed,fade=fade,slot=slot,candidate=name,strategy=optimizer.config.strategy,
                                            population=optimizer.config.population,cost=result.best_cost,runtime_ms=result.wall_clock_time_sec*1000,
                                            **delay_adjusted_metrics(result,CostConfig()),**optimizer.last_diagnostics))
                checkpoint.save(records,compact=True)
                rows.extend(records)
        frame = pd.DataFrame(rows)
        frame.to_csv(output/'slots.csv',index=False)
        metrics = ['cost','delay_adjusted_cost','runtime_ms','search_gain','search_improvements','cache_hits','mean_changed_users','evaluations']
        per_seed = frame.groupby(['seed','candidate','strategy','population'])[metrics].mean().reset_index()
        summary = per_seed.groupby(['candidate','strategy','population'])[metrics].mean().reset_index()
        summary.to_csv(output/'summary.csv',index=False)
        selected = summary.loc[summary.groupby('strategy').delay_adjusted_cost.idxmin()]
        selected.to_csv(output/'development_selected.csv',index=False)
        print(summary.round(4).to_string(index=False))
        print('Development selections only; no final-test claim:')
        print(selected[['strategy','population','delay_adjusted_cost']].round(4).to_string(index=False))


if __name__=='__main__':
    main()
