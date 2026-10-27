"""Quality-only summaries: no pooled runtime or delay-adjusted inference."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from experiments.research_report import METRICS,PRIMARY,interval,exact_report,quality_report,plot_sweep


def timing_column(name):
    return (name in {'retrieval_ms','kb_build_seconds','delay_feasible_cost'} or
            'runtime' in name or 'wall_clock' in name or 'delay_adjusted' in name or
            'delay_physical' in name or name.endswith('_seconds'))


def create_quality_reports(frame,output,settings):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);(output/'figures').mkdir(exist_ok=True)
    frame=frame.drop(columns=[name for name in frame.columns if timing_column(name)]).copy()
    metrics=[name for name in METRICS if name in frame and not timing_column(name)]
    pending=output/'quality_slots.pending.csv';frame.to_csv(pending,index=False);pending.replace(output/'quality_slots.csv')
    seeds=frame.groupby(['study','scenario','seed','phase','algorithm'])[metrics].mean().reset_index()
    seeds.to_csv(output/'seed_summary.csv',index=False)
    rows=[]
    for keys,group in seeds.groupby(['study','scenario','phase','algorithm']):
        for metric in metrics:
            mean,margin,n=interval(group[metric])
            rows.append(dict(zip(['study','scenario','phase','algorithm'],keys),metric=metric,mean=mean,ci95_margin=margin,independent_seeds=n))
    summary=pd.DataFrame(rows);summary.to_csv(output/'summary_ci95.csv',index=False)
    wide=summary.pivot(index=['study','scenario','phase','algorithm'],columns='metric',values=['mean','ci95_margin'])
    wide.columns=['_'.join(names) for names in wide.columns];wide.to_csv(output/'method_family_comparison.csv')
    definitions=pd.DataFrame(settings['scenarios']).rename(columns={'name':'scenario'})
    definitions.to_csv(output/'scenario_table.csv',index=False)
    summary[summary.metric.str.contains('violation|admission|feasible_pct')].merge(definitions,on=['study','scenario']).to_csv(output/'qos_scenario_table.csv',index=False)
    pd.DataFrame([dict(parameter=key,value=json.dumps(value)) for key,value in settings.items() if key not in ('scenarios','learner_checksums')]).to_csv(output/'parameter_table.csv',index=False)
    pd.DataFrame([dict(algorithm=name,objective_evaluations=settings['evaluations'] if name not in ('Greedy-Latency','Greedy-Uniform','DQN-Frozen-Legacy','DROO-Adapted-Frozen-Legacy') else None,
                      note='Allocation alias of Greedy-Latency' if name=='Greedy-Uniform' else '') for name in sorted(frame.algorithm.unique())]).to_csv(output/'method_family_table.csv',index=False)
    pairs=[];left=frame[frame.algorithm=='WE-KB-FPA-Greedy']
    for name in sorted(set(frame.algorithm)-{'WE-KB-FPA-Greedy'}):
        paired=left.merge(frame[frame.algorithm==name],on=['study','scenario','seed','slot','phase'],suffixes=('_fpa','_baseline'))
        matched=paired.feasible_task_ids_fpa.fillna('')==paired.feasible_task_ids_baseline.fillna('')
        paired['same_feasible_tasks']=matched
        paired['conditional_difference']=np.where(matched,paired.feasible_cost_fpa-paired.feasible_cost_baseline,np.nan)
        for keys,group in paired.groupby(['study','scenario','seed','phase']):
            pairs.append(dict(zip(['study','scenario','seed','phase'],keys),baseline=name,
                matched_task_set_slots=int(group.same_feasible_tasks.sum()),total_slots=len(group),
                mean_conditional_cost_difference=group.conditional_difference.mean()))
    pd.DataFrame(pairs).to_csv(output/'matched_cost_comparisons.csv',index=False)
    frame.groupby(['study','scenario','N','algorithm'])[['pruning_rate','mean_sinr_db']].mean().to_csv(output/'pruning_by_sinr_and_N.csv')
    exact_report(frame,output);quality_report(frame,output,settings)
    if 'best_known_ratio' in frame:
        plot_sweep(frame,'exact','N','best_known_ratio','Cost / best-known incumbent cost',output/'figures'/'best_known_reference_ratio')
        plot_sweep(frame,'exact','N','certified_gap_upper_pct','Upper bound on gap to relaxed joint optimum (%)',output/'figures'/'reference_gap_upper_bound')
    # Crossovers require isolated serial timings, not values from the quality run.
    pd.DataFrame([dict(status='Unavailable in quality-only dataset; use the separate serial snapshot benchmark')]).to_csv(output/'deadline_crossover_brackets.csv',index=False)
    for study,x in [('channel','target_sinr_db'),('scalability','N'),('workload','task_mbit')]:
        for metric,label in [('violation_pct','Service deadline violations, excluding decision time (%)'),
                             ('feasible_cost','Cost over deadline-feasible tasks'),('search_objective','Penalized search objective')]:
            plot_sweep(frame,study,x,metric,label,output/'figures'/f'{study}_{metric}')
    for channel,part in frame[frame.study=='deadline'].groupby('target_sinr_db'):
        plot_sweep(part,'deadline','tau_scale','violation_pct','Service deadline violations, excluding decision time (%)',output/'figures'/f'deadline_service_sinr_{channel:g}')
    for (study,name),part in frame[frame.study.isin(['shock','drl'])].groupby(['study','scenario']):
        for metric in ['violation_pct','feasible_cost']:
            plot_sweep(part,study,'slot',metric,metric.replace('_',' '),output/'figures'/f'{study}_{name}_{metric}')
    for study,x in [('channel','target_sinr_db'),('scalability','N')]:
        plot_sweep(frame,study,x,'pruning_rate','Fraction of target choices excluded',output/'figures'/f'pruning_{x}')
    q=frame[(frame.study=='kb')&frame.scenario.str.startswith('Q')]
    for metric in ['violation_pct','kb_bytes_per_entry','compatible_anchors']:
        plot_sweep(q,'kb','q',metric,metric.replace('_',' '),output/'figures'/f'kb_{metric}')
    for column in ['csi_error_db','csi_age']:
        part=frame[(frame.study=='csi')&(frame.csi_age==0 if column=='csi_error_db' else frame.csi_error_db==0)]
        plot_sweep(part,'csi',column,'violation_pct','Service deadline violations (%)',output/'figures'/f'{column}_service_violations')
    ablation=frame[(frame.study=='ablation')&frame.algorithm.str.match(r'FPA-P[01]-Phi[01]-Pt[01]-KB[01]$')].copy()
    effects=[]
    if not ablation.empty:
        flags=ablation.algorithm.str.extract(r'FPA-P([01])-Phi([01])-Pt([01])-KB([01])').astype(int)
        flags.columns=['pruner','phi','switch','kb']
        for column in flags:
            ablation[column]=flags[column]
        for component in flags:
            for metric in ['search_objective','violation_pct','feasible_cost']:
                trial=ablation.groupby(['seed',component])[metric].mean().unstack()
                mean,margin,n=interval(trial[1]-trial[0])
                effects.append(dict(component=component,metric=metric,enabled_minus_disabled=mean,ci95_margin=margin,independent_seeds=n))
        pd.DataFrame(effects).to_csv(output/'factorial_main_effects.csv',index=False)
        for metric in ['search_objective','violation_pct']:
            trial=ablation.groupby(['seed','algorithm'])[metric].mean().reset_index()
            values=[]
            for name,group in trial.groupby('algorithm'):
                mean,margin,_=interval(group[metric]);values.append((name,mean,margin or 0.))
            fig,axis=plt.subplots(figsize=(10,5))
            axis.bar(range(len(values)),[value[1] for value in values],yerr=[value[2] for value in values],capsize=2)
            axis.set_xticks(range(len(values)),[value[0] for value in values],rotation=90,fontsize=7)
            axis.set_ylabel(metric.replace('_',' '));fig.tight_layout()
            fig.savefig(output/'figures'/f'factorial_{metric}.pdf');plt.close(fig)
    (output/'scope.json').write_text(json.dumps(dict(scope='Time-independent quality metrics only; no runtime substitution',
        deadline_crossovers='Serial snapshot benchmark only',reporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
