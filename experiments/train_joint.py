"""Checkpointed development-only tuning and training for the joint protocol."""
import argparse
import copy
import hashlib
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from we_kb_fpa.checkpoint import Checkpoint,RunLock,namespace,implementation_signature
from we_kb_fpa.config import NetworkConfig
from we_kb_fpa.evaluation import snapshots
from we_kb_fpa.joint_allocation import JointAllocationConfig
from we_kb_fpa.joint_learning import JointLearner,JointLearningConfig,utility
from we_kb_fpa.joint_search import JointOffloadingOptimizer,JointSearchConfig


def validate(learner,seed,count):
    trial=copy.deepcopy(learner)
    scores=[];admission=[];references=[]
    baseline=JointOffloadingOptimizer(JointSearchConfig(strategy='greedy',budget_ms=learner.config.budget_ms),
        learner.allocator.config,learner.allocator.net,seed=seed)
    for _,args,_,_ in snapshots(seed,count,fading_db=15.,midpoint=count//2):
        result=trial.run(args,training=False)
        reference=baseline.optimize(*args)
        scores.append(utility(result.solution));admission.append(result.solution.admitted)
        references.append(reference.solution.admitted)
    return dict(mean_utility=float(np.mean(scores)),mean_admitted=float(np.mean(admission)),
                greedy_mean_admitted=float(np.mean(references)),
                admission_gap_to_greedy=float(np.mean(admission)-np.mean(references)))


def fit(root,name,config,allocation,net,seed,slots,validation_slots,checkpoint_every):
    metadata=dict(config=asdict(config),allocation=asdict(allocation),network=asdict(net),seed=seed,
                  slots=slots,validation_slots=validation_slots,checkpoint_every=checkpoint_every)
    checkpoint=Checkpoint(root,name,metadata)
    state=checkpoint.load()
    if state and state.get('complete'):
        return state
    if state is None:
        state=dict(agent=JointLearner(config,allocation,net,seed),completed=0,stream=None,reports=[],best=None,
                   best_score=-float('inf'),seconds=0.)
    tick=time.perf_counter()
    interval=max(1,slots//4)
    while state['completed']<slots:
        if state['stream'] is None:
            block=state['completed']//50
            fade=[0.,5.,15.,25.][block%4]
            state['stream']=snapshots(seed*1000000+block, min(50,slots-state['completed']),fading_db=fade,midpoint=0)
        try:
            _,args,_,_=next(state['stream'])
        except StopIteration:
            state['stream']=None
            continue
        state['agent'].run(args,training=True)
        state['completed']+=1
        if state['completed']%interval==0 or state['completed']==slots:
            score=validate(state['agent'],210000+seed,validation_slots)
            state['reports'].append(dict(training_slots=state['completed'],**score))
            if score['mean_utility']>state['best_score']:
                state['best']=copy.deepcopy(state['agent']);state['best_score']=score['mean_utility']
            print(f"{name}: {state['completed']}/{slots}, validation admission gap={score['admission_gap_to_greedy']:.3f}",flush=True)
        if state['completed']%checkpoint_every==0:
            state['seconds']+=time.perf_counter()-tick;checkpoint.save(state);tick=time.perf_counter()
    state['seconds']+=time.perf_counter()-tick
    state['complete']=True
    checkpoint.save(state,compact=True)
    return state


def load_models(directory,allocation,net):
    directory=Path(directory)
    manifest=json.loads((directory/'manifest.json').read_text())
    if manifest['implementation']!=implementation_signature():
        raise ValueError('Learner bank source/environment changed; retrain or restore the exact implementation')
    if manifest['allocation']!=asdict(allocation) or manifest['network']!=asdict(net):
        raise ValueError('Learner and evaluation allocation/network configurations differ')
    return manifest


def load_model(manifest,kind,seed):
    item=manifest['models'][f'{kind}-{seed}']
    checkpoint=Checkpoint(item['root'],item['name'],item['metadata'])
    if hashlib.sha256(checkpoint.path.read_bytes()).hexdigest()!=item['sha256']:
        raise ValueError('Trained artifact changed after bank publication')
    state=checkpoint.load()
    if not state or not state.get('complete'):
        raise ValueError('Incomplete learner artifact')
    return copy.deepcopy(state['best'])


def training_reports(manifest,output):
    rows=[]
    for name,item in manifest['models'].items():
        kind,seed=name.rsplit('-',1)
        rows.extend(dict(kind=kind,model_seed=int(seed),training_seconds=item['training_seconds'],**row)
                    for row in item['validation_history'])
    frame=pd.DataFrame(rows)
    frame.to_csv(output/'training_validation.csv',index=False)
    selected=frame.loc[frame.groupby(['kind','model_seed']).mean_utility.idxmax()].copy()
    selected['selected_epoch_below_greedy_admission']=selected.admission_gap_to_greedy<-.000001
    selected.to_csv(output/'selected_model_quality.csv',index=False)
    for metric in ['mean_admitted','admission_gap_to_greedy','mean_utility']:
        fig,axis=plt.subplots(figsize=(6,4))
        for kind,group in frame.groupby('kind'):
            values=group.groupby('training_slots')[metric].agg(['mean','std'])
            axis.plot(values.index,values['mean'],marker='o',label=kind)
            axis.fill_between(values.index,values['mean']-values['std'].fillna(0),values['mean']+values['std'].fillna(0),alpha=.15)
        axis.set(xlabel='Development training slots',ylabel=metric.replace('_',' '))
        axis.legend();axis.grid(alpha=.2);fig.tight_layout()
        fig.savefig(output/f'training_{metric}.pdf');plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick',action='store_true')
    parser.add_argument('--output-dir',default='output/joint_models')
    parser.add_argument('--training-slots',type=int)
    parser.add_argument('--model-seeds',type=int,nargs='+')
    parser.add_argument('--validation-slots',type=int)
    parser.add_argument('--checkpoint-every',type=int,default=100)
    parser.add_argument('--budget-ms',type=float,default=50.)
    args=parser.parse_args()
    count=args.training_slots or (40 if args.quick else 10000)
    validation=args.validation_slots or (6 if args.quick else 40)
    seeds=args.model_seeds or list(range(42,44 if args.quick else 72))
    print(f'Development training: {len(seeds)} independent replicates per learner, {count} slots each; final evaluation data are excluded.',flush=True)
    if count<4 or validation<2 or args.checkpoint_every<1 or len(set(seeds))!=len(seeds) or any(s<0 or s>=10000 for s in seeds):
        parser.error('Invalid development training settings')
    output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=True)
    allocation=JointAllocationConfig(decision_reserve_sec=args.budget_ms/1000)
    net=NetworkConfig(num_users=8,num_edge_servers=3)
    settings=dict(protocol='joint-learning-v1',training_slots=count,validation_slots=validation,seeds=seeds,
        allocation=asdict(allocation),network=asdict(net),quick=args.quick,checkpoint_every=args.checkpoint_every)
    with RunLock(output/'train.lock'):
        root=namespace(output/'checkpoints',settings)
        manifest=dict(implementation=implementation_signature(),allocation=asdict(allocation),network=asdict(net),
            training=settings,models={},selection={},
            reward='Terminal admitted_count - service_cost/(1+service_cost), gamma=1. Same per-snapshot ordering as lexicographic rank; expected scalar utility is not strict lexicographic expected admission.',
            distribution='Separate simulator seeds; nominal and 5/15/25 dB development fades. No final 35 dB trajectories used.',
            tuning='Two learning rates, two independent calibration training seeds each; same slots and validation effort for both learning methods.',
            adaptation='Categorical candidate replay with cross-entropy and common local/latency-greedy candidates to prevent self-training collapse; DROO-inspired, not original binary DROO.',
            status='smoke_only' if args.quick else 'development_trained_no_convergence_guarantee')
        for kind in ['dqn','categorical']:
            candidates=[]
            for index,rate in enumerate([.0003,.001]):
                scores=[]
                config=JointLearningConfig(kind=kind,budget_ms=args.budget_ms,learning_rate=rate)
                for calibration_seed in [9001,9002]:
                    state=fit(root,f'tune-{kind}-{index}-{calibration_seed}',config,allocation,net,calibration_seed,
                              min(count,2000),validation,args.checkpoint_every)
                    scores.append(validate(state['best'],230000+calibration_seed,validation)['mean_utility'])
                candidates.append(dict(rate=rate,mean_validation_utility=float(np.mean(scores))))
            chosen=max(candidates,key=lambda item:item['mean_validation_utility'])['rate']
            manifest['selection'][kind]=dict(candidates=candidates,selected_learning_rate=chosen)
            config=JointLearningConfig(kind=kind,budget_ms=args.budget_ms,learning_rate=chosen)
            for seed in seeds:
                name=f'model-{kind}-{seed}'
                state=fit(root,name,config,allocation,net,seed,count,validation,args.checkpoint_every)
                metadata=dict(config=asdict(config),allocation=asdict(allocation),network=asdict(net),seed=seed,
                    slots=count,validation_slots=validation,checkpoint_every=args.checkpoint_every)
                manifest['models'][f'{kind}-{seed}']=dict(root=str(root.resolve()),name=name,metadata=metadata,
                    validation_history=state['reports'],training_seconds=state['seconds'],
                    sha256=hashlib.sha256(Checkpoint(root,name,metadata).path.read_bytes()).hexdigest())
        pending=output/'manifest.pending.json';pending.write_text(json.dumps(manifest,indent=2));pending.replace(output/'manifest.json')
        training_reports(manifest,output)
    print(f'Matched learner bank complete: {output.resolve()}',flush=True)


if __name__=='__main__':
    main()
