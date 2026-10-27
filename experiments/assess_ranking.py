"""Bounded development screen of delay-aware ranking, without retuning or DRL claims."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy.stats import t
from experiments.run_revision import run_case
from we_kb_fpa.anytime import AnytimeConfig
from we_kb_fpa.checkpoint import RunLock,namespace


def matched_configs(path,budget):
    source=json.loads(Path(path).read_text())
    names=[f'No-Event-{budget:g}ms',f'KB-Categorical-GA-{budget:g}ms',
           f'DelayRank-FPA-{budget:g}ms',f'DelayRank-GA-{budget:g}ms',f'Anchor-Only-{budget:g}ms']
    configs={name:AnytimeConfig(**source[name]) for name in names}
    for raw,ranked in [(names[0],names[2]),(names[1],names[3])]:
        left,right=asdict(configs[raw]),asdict(configs[ranked])
        for key in ['deadline_aware','stopping_enabled']:
            left.pop(key);right.pop(key)
        if left!=right or configs[raw].deadline_aware or not configs[ranked].deadline_aware or configs[ranked].stopping_enabled:
            raise ValueError('Ranking screen requires identical search settings with only deployment ranking changed')
    return {name:asdict(cfg) for name,cfg in configs.items()}


def report(frame,output,threshold):
    metrics=['cost','delay_adjusted_cost','delay_adjusted_violation_pct','runtime_ms']
    means=frame[frame.phase=='post'].groupby(['seed','fade','algorithm'])[metrics].mean().reset_index()
    means.groupby(['fade','algorithm'])[metrics].mean().to_csv(output/'scenario_summary.csv')
    seed_means=means.groupby(['seed','algorithm'])[metrics].mean().reset_index()
    summary=seed_means.groupby('algorithm')[metrics].mean()
    summary.to_csv(output/'summary.csv')
    effects=[]
    flags={}
    for strategy,baseline in [('FPA','No-Event-10ms'),('GA','KB-Categorical-GA-10ms')]:
        ranked=f'DelayRank-{strategy}-10ms'
        for reference in [baseline,'Greedy-Uniform','Anchor-Only-10ms']:
            paired=seed_means[seed_means.algorithm==ranked].merge(seed_means[seed_means.algorithm==reference],on='seed',suffixes=('_ranked','_baseline'))
            for metric in metrics:
                delta=paired[metric+'_ranked']-paired[metric+'_baseline']
                n=len(delta)
                margin=float(t.ppf(.975,n-1)*delta.std()/np.sqrt(n)) if n>1 else None
                gain=100*(paired[metric+'_baseline'].mean()-paired[metric+'_ranked'].mean())/max(1e-12,paired[metric+'_baseline'].mean())
                effects.append(dict(strategy=strategy,baseline=reference,metric=metric,
                    mean_difference=float(delta.mean()),ci95_margin=margin,relative_reduction_pct=float(gain),independent_seeds=n))
                if reference==baseline and metric=='delay_adjusted_cost':
                    flags[strategy]=dict(relative_reduction_pct=float(gain),ci95_excludes_zero_in_favor=bool(margin is not None and delta.mean()+margin<0),
                                         passes_development_screen=bool(margin is not None and delta.mean()+margin<0 and gain>=threshold))
    pd.DataFrame(effects).to_csv(output/'paired_effects.csv',index=False)
    passed=all(item['passes_development_screen'] for item in flags.values())
    assessment=dict(role='development screen, not a publication or novelty assessment',
                    independent_seeds=int(frame.seed.nunique()),selection_on_these_seeds=False,
                    minimum_mean_reduction_pct=threshold,strategies=flags,
                    recommendation='Further targeted validation is justified; DRL convergence and novelty remain separate requirements.' if passed else
                    'The predeclared ranking screen did not pass for both strategies. Do not launch a full final suite or claim a general framework advantage.',
                    final_evaluation_run=False)
    (output/'assessment.json').write_text(json.dumps(assessment,indent=2))
    print(summary.round(4).to_string())
    print(json.dumps(assessment,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selected-configs',default='output/stopping_development/selected_configs.json')
    parser.add_argument('--seeds',type=int,nargs='+',default=list(range(10011,10021)))
    parser.add_argument('--slots',type=int,default=20)
    parser.add_argument('--min-reduction-pct',type=float,default=1.)
    parser.add_argument('--output-dir',default='output/ranking_assessment')
    args=parser.parse_args()
    if (args.slots<6 or args.slots%2 or len(set(args.seeds))!=len(args.seeds) or
            any(seed<10000 or seed>=30000 for seed in args.seeds) or
            not np.isfinite(args.min_reduction_pct) or args.min_reduction_pct<0):
        parser.error('Use even slots >=6, distinct development seeds in [10000,30000), and a finite nonnegative screening threshold')
    calibration=Path(args.selected_configs).parent/'config.json'
    if calibration.exists():
        declared=json.loads(calibration.read_text())
        used=set(declared.get('fit_seeds',[])+declared.get('check_seeds',[]))
        if used.intersection(args.seeds):
            parser.error('Screen seeds must differ from the stopping study calibration and check seeds')
    configs=matched_configs(args.selected_configs,10.)
    settings=dict(evaluation_role='development',seeds=args.seeds,slots=args.slots,scale_slots=1,
                  workers=1,checkpoint_every=10,resume=True,include_current=False,model_seeds=[],
                  budgets_ms=[10.],optimizer_variants=configs,fades=[0.,15.,35.],
                  parent_configs_sha256=hashlib.sha256(Path(args.selected_configs).read_bytes()).hexdigest(),
                  primary_metric='Post-phase delay-adjusted cost; equal weights across seed/scenario means',
                  screen='For each strategy require paired 95% CI below zero and mean cost reduction >= declared threshold. Exploratory gate, not a final confirmatory test.',
                  minimum_reduction_pct=args.min_reduction_pct,
                  information='Current shared snapshots and method-specific causal history only',
                  purpose='One bounded ranking assessment, no retuning, no final-test seeds, no DRL conclusion')
    output=Path(args.output_dir)
    output.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    with RunLock(output/'run.lock'):
        root=namespace(output/'checkpoints',settings)
        (output/'config.json').write_text(json.dumps(settings,indent=2))
        records=[]
        cases=[(seed,fade) for seed in args.seeds for fade in settings['fades']]
        for index,(seed,fade) in enumerate(cases,1):
            records.extend(run_case(settings,root,'shock',seed,8,fade))
            temporary=output/'slots.pending.csv'
            pd.DataFrame(records).to_csv(temporary,index=False)
            temporary.replace(output/'slots.csv')
            print(f'Completed {index}/{len(cases)} development cases; {(time.perf_counter()-started):.1f} s',flush=True)
        report(pd.DataFrame(records),output,args.min_reduction_pct)
        (output/'timing.json').write_text(json.dumps(dict(seconds=time.perf_counter()-started),indent=2))


if __name__=='__main__':
    main()
