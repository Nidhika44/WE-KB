"""Bounded reward-scale study using development data and unchanged linear cost rewards."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from experiments.run_topology_shock import training_config
from we_kb_fpa.checkpoint import Checkpoint,RunLock,namespace
from we_kb_fpa.dqn import DQNBaseline,DQNConfig
from we_kb_fpa.evaluation import snapshots,validation_bank,validate


def configurations():
    return {f'scale{scale:g}-lr{rate:g}':DQNConfig(reward_transform='linear',reward_scale=scale,
            learning_rate=rate,gamma=1.) for scale in [100.,1000.,10000.] for rate in [.001,.0003]}


def select_on_calibration(rows):
    frame=pd.DataFrame(rows)
    if set(frame.role)!= {'calibration'}:
        raise ValueError('Candidate selection can only use calibration diagnostics')
    return str(frame.groupby('candidate').validation_cost.mean().idxmin())


def critic_diagnostics(agent):
    count=min(1024,len(agent.memory))
    indices=np.random.default_rng(9271).choice(len(agent.memory),size=count,replace=False)
    states,actions,rewards,next_states,done=map(np.asarray,zip(*(agent.memory[int(i)] for i in indices)))
    predicted=agent._forward(states,agent.weights)[0][np.arange(count),actions]
    targets=rewards+agent.config.gamma*(1-done)*agent._forward(next_states,agent.target)[0].max(axis=1)
    error=predicted-targets
    return dict(huber_linear_region_pct=float(100*np.mean(abs(error)>1)),
                reward_min=float(rewards.min()),reward_max=float(rewards.max()),
                td_abs_median=float(np.median(abs(error))),td_abs_p99=float(np.quantile(abs(error),.99)),
                all_finite=bool(np.isfinite(states).all() and np.isfinite(error).all()))


def fit(candidate,config,split,role,slots,bank,root):
    settings=dict(candidate=candidate,config=asdict(config),split=split,role=role,slots=slots)
    checkpoint=Checkpoint(root,f'{role}-{split}-{candidate}',settings)
    state=checkpoint.load()
    if state and state.get('complete'):
        return state['diagnostic']
    if state is None:
        state=dict(agent=DQNBaseline(seed=1000000+split,config=training_config(config,slots)),
                   stream=snapshots(1000000+split,slots))
    last_saved=time.perf_counter()
    for slot,args,_,_ in state['stream']:
        state['agent'].run(*args,training=True)
        if (slot+1)%250==0 or time.perf_counter()-last_saved>=60:
            checkpoint.save(state)
            last_saved=time.perf_counter()
    diagnostic=dict(candidate=candidate,split=split,role=role,training_slots=slots,
                    training_seed=1000000+split,validation_seed=2000000+split,
                    scheduled_epsilon_decay=state['agent'].config.epsilon_decay,
                    **asdict(config),**validate(state['agent'],bank),**critic_diagnostics(state['agent']))
    checkpoint.save(dict(complete=True,diagnostic=diagnostic),compact=True)
    print(f'{role}: {candidate}, split {split}, nominal ratio {diagnostic["nominal_cost_ratio"]:.3f}, pass={diagnostic["nominal_quality_pass"]}',flush=True)
    return diagnostic


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-slots',type=int,default=2000)
    parser.add_argument('--validation-slots',type=int,default=100)
    parser.add_argument('--output-dir',default='output/dqn_scale_development')
    args=parser.parse_args()
    if args.training_slots<1 or args.validation_slots<1:
        parser.error('Training and validation slots must be positive')
    output=Path(args.output_dir)
    output.mkdir(parents=True,exist_ok=True)
    grid=configurations()
    settings=dict(role='development diagnosis, not final validation or proof of convergence',
                  training_slots=args.training_slots,validation_slots=args.validation_slots,
                  calibration_splits=[0,1],check_splits=[2,3],grid={name:asdict(cfg) for name,cfg in grid.items()},
                  selection='Minimum mean raw validation cost across both calibration splits; all failures retained',
                  epsilon_schedule='epsilon_decay is the base config; scheduled_epsilon_decay is exp(log(epsilon_min)/max(1,0.8*training_slots))',
                  objective='Identical linear cost objective, positive constant reward rescaling, gamma=1; no clipping of rewards')
    started=time.perf_counter()
    with RunLock(output/'run.lock'):
        root=namespace(output/'checkpoints',settings)
        (output/'config.json').write_text(json.dumps(settings,indent=2))
        calibration=[]
        for split in settings['calibration_splits']:
            bank=validation_bank(2000000+split,args.validation_slots)
            for name,config in grid.items():
                calibration.append(fit(name,config,split,'calibration',args.training_slots,bank,root))
        pd.DataFrame(calibration).to_csv(output/'calibration.csv',index=False)
        selected=select_on_calibration(calibration)
        (output/'selected_dqn_config.json').write_text(json.dumps(asdict(grid[selected]),indent=2))
        reference='scale100-lr0.001'
        checked=[]
        for split in settings['check_splits']:
            bank=validation_bank(2000000+split,args.validation_slots)
            for name in dict.fromkeys([selected,reference]):
                checked.append(fit(name,grid[name],split,'development_check',args.training_slots,bank,root))
        check=pd.DataFrame(checked)
        check.to_csv(output/'check.csv',index=False)
        chosen=check[check.candidate==selected]
        assessment=dict(role='development-only diagnosis',selected_candidate=selected,
                        quality_passes=int(chosen.nominal_quality_pass.sum()),check_splits=len(chosen),
                        mean_nominal_cost_ratio=float(chosen.nominal_cost_ratio.mean()),
                        adequate_training_demonstrated=False,
                        explanation='A short scale study can diagnose optimization behavior; it cannot establish convergence or final robustness.',
                        final_evaluation_run=False)
        (output/'assessment.json').write_text(json.dumps(assessment,indent=2))
        (output/'timing.json').write_text(json.dumps(dict(seconds=time.perf_counter()-started),indent=2))
        print(json.dumps(assessment,indent=2))


if __name__=='__main__':
    main()
