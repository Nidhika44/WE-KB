"""Seed-level uncertainty, scenario provenance, and research figures."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


METRICS=['search_objective','feasible_cost','feasible_cost_per_task','delay_feasible_cost','admission_pct',
    'deadline_feasible_pct','violation_pct','delay_adjusted_violation_pct','physical_violation_lower_bound_pct',
    'delay_physical_violation_lower_bound_pct','runtime_ms','cost_evaluations','retrieval_ms','kb_entries',
    'kb_bytes_per_entry','compatible_anchors','pruning_rate']
METRICS+=['observed_pruned_but_realized_possible_pct','optimistically_impossible_target_fraction']
PRIMARY=['WE-KB-FPA-Greedy','PSO+WE-KB','GA+WE-KB','FPA+Pruner','FPA-Greedy','Greedy-Latency','Greedy-Uniform',
         'DQN-Frozen-Legacy','DROO-Adapted-Frozen-Legacy']


def interval(values):
    values=pd.Series(values).dropna();n=len(values)
    mean=float(values.mean()) if n else None
    margin=float(t.ppf(.975,n-1)*values.std()/np.sqrt(n)) if n>1 else None
    return mean,margin,n


def bootstrap_median(values):
    values=np.asarray(values,dtype=float);values=values[np.isfinite(values)]
    if len(values)<2:
        return (float(np.median(values)) if len(values) else None,None,None)
    rng=np.random.default_rng(20261026)
    medians=np.median(rng.choice(values,(2000,len(values)),replace=True),axis=1)
    return float(np.median(values)),*map(float,np.quantile(medians,[.025,.975]))


def plot_sweep(frame,study,x,metric,label,path,primary=PRIMARY):
    part=frame[(frame.study==study)&frame.algorithm.isin(primary)].copy()
    part=part.dropna(subset=[x,metric])
    if part.empty:
        return
    values=part.groupby(['seed','algorithm',x])[metric].mean().reset_index()
    fig,axis=plt.subplots(figsize=(7,4.5))
    for name,group in values.groupby('algorithm'):
        xs=[];means=[];margins=[]
        for value,points in group.groupby(x):
            mean,margin,_=interval(points[metric]);xs.append(value);means.append(mean);margins.append(margin or 0.)
        axis.errorbar(xs,means,yerr=margins,marker='o',capsize=2,label=name,linewidth=1)
    axis.set(xlabel=x.replace('_',' '),ylabel=label);axis.grid(alpha=.2)
    axis.legend(fontsize=7,loc='best');fig.tight_layout();fig.savefig(path.with_suffix('.pdf'));fig.savefig(path.with_suffix('.png'),dpi=200);plt.close(fig)


def deadline_crossovers(frame,output):
    part=frame[frame.study=='deadline']
    rows=[]
    for channel,channel_group in part.groupby('target_sinr_db'):
        means=channel_group.groupby(['seed','algorithm','tau_scale']).delay_adjusted_violation_pct.mean().reset_index()
        fpa=means[means.algorithm=='WE-KB-FPA-Greedy']
        for baseline in [name for name in PRIMARY if name!='WE-KB-FPA-Greedy']:
            other=means[means.algorithm==baseline]
            paired=fpa.merge(other,on=['seed','tau_scale'],suffixes=('_fpa','_baseline'))
            points=[]
            for scale,group in paired.groupby('tau_scale'):
                difference=group.delay_adjusted_violation_pct_fpa-group.delay_adjusted_violation_pct_baseline
                mean,margin,n=interval(difference);points.append((scale,mean,margin,n))
            found=False
            for left,right in zip(points,points[1:]):
                if left[1]*right[1]<0:
                    found=True
                    rows.append(dict(target_sinr_db=channel,baseline=baseline,tau_scale_lower=left[0],tau_scale_upper=right[0],
                        delta_lower=left[1],delta_upper=right[1],status='Observed sign-change bracket; hardware- and grid-dependent',
                        independent_seeds=min(left[3],right[3])))
            if points and not found:
                rows.append(dict(target_sinr_db=channel,baseline=baseline,tau_scale_lower=points[0][0],tau_scale_upper=points[-1][0],
                    status='No strict sign change observed on sampled grid',independent_seeds=points[0][3]))
        for metric,label in [('delay_adjusted_violation_pct','Delay-adjusted deadline violations (%)'),('runtime_ms','Decision time (ms)')]:
            plot_sweep(channel_group,'deadline','tau_scale',metric,label,output/'figures'/f'deadline_sinr_{channel:g}_{metric}')
    pd.DataFrame(rows).to_csv(output/'deadline_crossover_brackets.csv',index=False)


def exact_report(frame,output):
    part=frame[frame.study=='exact'];rows=[]
    if part.empty:
        return
    for (n,name),group in part.groupby(['N','algorithm']):
        median,lo,hi=bootstrap_median(100*(group.best_known_ratio-1))
        bound_median,bound_lo,bound_hi=bootstrap_median(group.certified_gap_upper_pct)
        exact_median,exact_lo,exact_hi=bootstrap_median(group.gap_to_proven_optimum_pct)
        rows.append(dict(N=n,algorithm=name,independent_seeds=group.seed.nunique(),
            mean_of_per_trial_ratios=group.best_known_ratio.mean(),ratio_of_means=group.search_objective.mean()/group.best_known_cost.mean(),
            median_difference_from_best_known_pct=median,median_ci95_lower=lo,median_ci95_upper=hi,
            median_certified_gap_upper_pct=bound_median,certified_gap_median_ci95_lower=bound_lo,certified_gap_median_ci95_upper=bound_hi,
            median_proven_optimality_gap_pct=exact_median,proven_gap_median_ci95_lower=exact_lo,proven_gap_median_ci95_upper=exact_hi,
            mean_certified_gap_upper_pct=group.certified_gap_upper_pct.mean(),
            proven_optimum_trials=int(group.reference_is_optimal.sum()),
            mean_solver_mip_gap=group.reference_reported_mip_gap.mean(),
            reference_scope=group.reference_scope.iloc[0]))
    pd.DataFrame(rows).to_csv(output/'reference_gap_statistics.csv',index=False)
    part[['seed','N','reference_status','reference_lower_bound','reference_incumbent_cost','reference_reported_mip_gap',
          'reference_simulator_match','reference_scope','reference_message']].drop_duplicates().to_csv(output/'reference_status.csv',index=False)


def quality_report(frame,output,settings):
    if not settings.get('include_drl') or not frame.algorithm.str.contains('Legacy').any():
        return
    records=json.loads((Path(settings['models_dir'])/'quality_gate.json').read_text())
    gates=[]
    for record in records:
        if record['training_budget']==settings['model_budget']:
            for kind in ['dqn','droo']:
                gates.append(dict(model_seed=record['seed'],kind=kind,**record[kind]))
    gates=pd.DataFrame(gates);gates.to_csv(output/'legacy_quality_per_seed.csv',index=False)
    gates.groupby('kind').agg(seeds=('model_seed','size'),passing=('nominal_quality_pass','sum'),pass_rate=('nominal_quality_pass','mean'),
        mean_nominal_cost_ratio=('nominal_cost_ratio','mean')).to_csv(output/'legacy_quality_gate_pass_rates.csv')
    selected=[]
    for kind,label in [('dqn','DQN-Frozen-Legacy'),('droo','DROO-Adapted-Frozen-Legacy')]:
        passing=set(gates[(gates.kind==kind)&gates.nominal_quality_pass].model_seed)
        rows=frame[frame.study.isin(['deadline','drl'])&frame.model_seed.isin(passing)].copy()
        rows['passing_reference']=label
        selected.append(rows)
    if selected:
        selected=pd.concat(selected)
        selected.to_csv(output/'passing_seed_comparisons.csv',index=False)
        metrics=[name for name in METRICS if name in selected]
        trials=selected.groupby(['passing_reference','study','scenario','seed','phase','algorithm'])[metrics].mean().reset_index()
        summaries=[]
        for keys,group in trials.groupby(['passing_reference','study','scenario','phase','algorithm']):
            for metric in metrics:
                mean,margin,n=interval(group[metric])
                summaries.append(dict(zip(['passing_reference','study','scenario','phase','algorithm'],keys),metric=metric,
                    mean=mean,ci95_margin=margin,passing_independent_seeds=n))
        pd.DataFrame(summaries).to_csv(output/'passing_seed_summary_ci95.csv',index=False)


def report(frame,output,settings):
    output=Path(output);(output/'figures').mkdir(exist_ok=True)
    metrics=[name for name in METRICS if name in frame]
    seeds=frame.groupby(['study','scenario','seed','phase','algorithm'],dropna=False)[metrics].mean().reset_index()
    seeds.to_csv(output/'seed_summary.csv',index=False)
    rows=[]
    for keys,group in seeds.groupby(['study','scenario','phase','algorithm']):
        for metric in metrics:
            mean,margin,n=interval(group[metric])
            rows.append(dict(zip(['study','scenario','phase','algorithm'],keys),metric=metric,mean=mean,ci95_margin=margin,independent_seeds=n))
    summary=pd.DataFrame(rows)
    summary.to_csv(output/'summary_ci95.csv',index=False)
    family=summary.pivot(index=['study','scenario','phase','algorithm'],columns='metric',values=['mean','ci95_margin'])
    family.columns=['_'.join(names) for names in family.columns]
    family.to_csv(output/'method_family_comparison.csv')
    pairs=[]
    left=frame[frame.algorithm=='WE-KB-FPA-Greedy']
    for baseline in sorted(set(frame.algorithm)-{'WE-KB-FPA-Greedy'}):
        paired=left.merge(frame[frame.algorithm==baseline],on=['study','scenario','seed','slot','phase'],suffixes=('_fpa','_baseline'))
        paired['same_feasible_tasks']=paired.feasible_task_ids_fpa.fillna('')==paired.feasible_task_ids_baseline.fillna('')
        paired['conditional_cost_difference']=np.where(paired.same_feasible_tasks,paired.feasible_cost_fpa-paired.feasible_cost_baseline,np.nan)
        for keys,group in paired.groupby(['study','scenario','seed','phase']):
            pairs.append(dict(zip(['study','scenario','seed','phase'],keys),baseline=baseline,
                matched_task_set_slots=int(group.same_feasible_tasks.sum()),total_slots=len(group),
                mean_conditional_cost_difference=group.conditional_cost_difference.mean()))
    pd.DataFrame(pairs).to_csv(output/'matched_cost_comparisons.csv',index=False)
    definitions=pd.DataFrame(settings['scenarios']).rename(columns={'name':'scenario'})
    summary[summary.metric.str.contains('violation|admission|feasible_pct')].merge(definitions,on=['study','scenario']).to_csv(output/'qos_scenario_table.csv',index=False)
    seed_table=[]
    for name,value in settings.items():
        if name not in ('scenarios','learner_checksums'):
            seed_table.append(dict(parameter=name,value=json.dumps(value)))
    pd.DataFrame(seed_table).to_csv(output/'parameter_table.csv',index=False)
    pd.DataFrame(settings['scenarios']).to_csv(output/'scenario_table.csv',index=False)
    pd.DataFrame([
        dict(method=name,family='FPA' if 'FPA' in name else 'PSO' if 'PSO' in name else 'GA' if 'GA+' in name else 'DQN' if 'DQN' in name else 'DROO adaptation' if 'DROO' in name else 'Greedy',
             scope='binary nearest-edge + uniform' if 'DROO' in name else 'all targets + uniform' if 'DQN' in name else 'continuous 3N joint' if name not in ['Greedy-Latency','Greedy-Uniform'] else 'latency targets + uniform resources',
             evaluations=settings['evaluations'] if name in ['WE-KB-FPA-Greedy','PSO+WE-KB','GA+WE-KB','FPA+Pruner','FPA-Greedy'] else None,
             note='Alias of Greedy-Latency in this legacy implementation' if name=='Greedy-Uniform' else '') for name in PRIMARY
    ]).to_csv(output/'method_family_table.csv',index=False)
    frame.groupby(['study','scenario','N','algorithm'])[['pruning_rate','mean_sinr_db']].mean().to_csv(output/'pruning_by_sinr_and_N.csv')
    exact_report(frame,output);quality_report(frame,output,settings);deadline_crossovers(frame,output)
    if 'best_known_ratio' in frame:
        plot_sweep(frame,'exact','N','certified_gap_upper_pct','Upper bound on gap to relaxed joint optimum (%)',output/'figures'/'reference_gap_upper_bound')
        plot_sweep(frame,'exact','N','best_known_ratio','Cost / best-known incumbent cost',output/'figures'/'best_known_reference_ratio')
        plot_sweep(frame,'exact','N','runtime_ms','Decision time (ms)',output/'figures'/'reference_study_runtime')
    for study,x,prefix in [('channel','target_sinr_db','sinr'),('scalability','N','scalability'),('workload','task_mbit','workload')]:
        for metric,label in [('delay_adjusted_violation_pct','Delay-adjusted violations (%)'),('feasible_cost','Cost over deadline-feasible tasks'),('runtime_ms','Decision time (ms)')]:
            plot_sweep(frame,study,x,metric,label,output/'figures'/f'{prefix}_{metric}')
    plot_sweep(frame,'channel','target_sinr_db','pruning_rate','Fraction of target choices excluded',output/'figures'/'pruning_vs_sinr')
    plot_sweep(frame,'scalability','N','pruning_rate','Fraction of target choices excluded',output/'figures'/'pruning_vs_N')
    qpart=frame[(frame.study=='kb')&frame.scenario.str.startswith('Q')]
    for metric in ['delay_adjusted_violation_pct','retrieval_ms','kb_bytes_per_entry']:
        plot_sweep(qpart,'kb','q',metric,metric.replace('_',' '),output/'figures'/f'kb_{metric}')
    for x in ['csi_error_db','csi_age']:
        part=frame[(frame.study=='csi')&(frame.csi_age==0 if x=='csi_error_db' else frame.csi_error_db==0)]
        plot_sweep(part,'csi',x,'delay_adjusted_violation_pct','Delay-adjusted violations (%)',output/'figures'/f'{x}_violations')
    for (study,scenario),part in frame[frame.study.isin(['shock','drl'])].groupby(['study','scenario']):
        for metric in ['delay_adjusted_violation_pct','feasible_cost']:
            plot_sweep(part,study,'slot',metric,metric.replace('_',' '),output/'figures'/f'{study}_{scenario}_{metric}')
    ablation=frame[(frame.study=='ablation')&frame.algorithm.str.match(r'FPA-P[01]-Phi[01]-Pt[01]-KB[01]$')].copy()
    effects=[]
    if not ablation.empty:
        flags=ablation.algorithm.str.extract(r'FPA-P([01])-Phi([01])-Pt([01])-KB([01])').astype(int)
        flags.columns=['pruner','phi','switch','kb']
        for column in flags:
            ablation[column]=flags[column]
        for component in flags:
            for metric in ['search_objective','delay_adjusted_violation_pct','runtime_ms']:
                trial=ablation.groupby(['seed',component])[metric].mean().unstack()
                mean,margin,n=interval(trial[1]-trial[0])
                effects.append(dict(component=component,metric=metric,enabled_minus_disabled=mean,ci95_margin=margin,independent_seeds=n))
        pd.DataFrame(effects).to_csv(output/'factorial_main_effects.csv',index=False)
        means=ablation.groupby('algorithm')[['search_objective','delay_adjusted_violation_pct']].mean()
        fig,axes=plt.subplots(2,1,figsize=(10,7))
        for axis,metric in zip(axes,means.columns):
            means[metric].plot.bar(ax=axis);axis.set_ylabel(metric.replace('_',' '));axis.tick_params(axis='x',labelsize=7)
        fig.tight_layout();fig.savefig(output/'figures'/'factorial_ablation.pdf');plt.close(fig)
