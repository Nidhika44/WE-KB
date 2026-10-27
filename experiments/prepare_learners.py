"""Train and validate full-state learning baselines without running test trajectories."""
import argparse
from dataclasses import asdict
import json
import hashlib
from pathlib import Path
from experiments.run_topology_shock import tune_dqn, training_config
from we_kb_fpa.checkpoint import RunLock, namespace
from we_kb_fpa.training import train_seed
from we_kb_fpa.config import NetworkConfig, CostConfig
from we_kb_fpa.dqn import DQNConfig


def no_test_evaluation(*args, **kwargs):
    return []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-slots',type=int,default=80000)
    parser.add_argument('--training-budgets',type=int,nargs='+',help='Intermediate full-state exports; final budget must equal training-slots')
    parser.add_argument('--tuning-slots',type=int,default=5000)
    parser.add_argument('--validation-slots',type=int,default=200)
    parser.add_argument('--validation-interval',type=int)
    parser.add_argument('--seeds',type=int,nargs='+',default=list(range(42,72)))
    parser.add_argument('--checkpoint-every',type=int,default=1000)
    parser.add_argument('--output-dir',default='output/learners_corrected')
    parser.add_argument('--fresh',action='store_true')
    parser.add_argument('--dqn-config',help='Development-selected DQNConfig JSON; bypasses the default learning-rate tuning grid')
    args = parser.parse_args()
    interval = args.validation_interval or max(1,args.training_slots//10)
    if (min(args.training_slots,args.tuning_slots,args.validation_slots,args.checkpoint_every,interval)<1
            or len(set(args.seeds))!=len(args.seeds) or any(s<0 or s>=100000 for s in args.seeds)):
        parser.error('Use positive budgets and unique model seeds in [0,100000)')
    output = Path(args.output_dir)
    output.mkdir(parents=True,exist_ok=True)
    budgets=sorted(set(args.training_budgets or [args.training_slots]))
    if budgets[0]<1 or budgets[-1]!=args.training_slots:
        parser.error('Training budgets must be positive and end at training-slots')
    config = dict(training_slots=args.training_slots,training_budgets=budgets,
                  tuning_slots=args.tuning_slots,validation_slots=args.validation_slots,
                  validation_interval=interval,require_nominal_quality=False,
                  slots=6,fading_values=[35.],population=20,iterations=30,
                  seeds=args.seeds,selection_metric='validation_cost',
                  network=asdict(NetworkConfig(num_users=8,num_edge_servers=3)),cost=asdict(CostConfig()),
                  purpose='Training and validation only; no test trajectory selection')
    config['reward_objective']='Linear scaled raw cost; gamma=1 finite assignment episode; checkpoint selection uses mean raw validation cost'
    supplied=None
    if args.dqn_config:
        supplied=DQNConfig(**json.loads(Path(args.dqn_config).read_text()))
        if supplied.reward_transform!='linear' or supplied.gamma!=1.:
            parser.error('Aligned baseline preparation requires linear rewards and gamma=1')
        config['supplied_dqn_config']=asdict(supplied)
        config['supplied_config_sha256']=hashlib.sha256(Path(args.dqn_config).read_bytes()).hexdigest()
    with RunLock(output/'run.lock'):
        root = namespace(output/'checkpoints',config)
        selected = training_config(supplied if supplied is not None else
                                   tune_dqn(args.tuning_slots,args.validation_slots,output,root,
                                            args.checkpoint_every,not args.fresh,'validation_cost'),args.training_slots)
        (output/'config.json').write_text(json.dumps(dict(config,selected_dqn=asdict(selected)),indent=2))
        qualities = []
        for seed in args.seeds:
            state,_ = train_seed(seed,config,selected,root,output,no_test_evaluation,args.checkpoint_every,not args.fresh)
            qualities.extend(state['qualities'])
            (output/'quality_gate.json').write_text(json.dumps(qualities,indent=2))
        final=[row for row in qualities if row['training_budget']==args.training_slots]
        passed = sum(row['dqn']['nominal_quality_pass'] for row in final)
        print(f'DQN final nominal quality: {passed}/{len(final)} passed. All seeds retained; training length is not proof of convergence.')
        print(f'Full-state exports: {output.resolve()}')


if __name__=='__main__':
    main()
