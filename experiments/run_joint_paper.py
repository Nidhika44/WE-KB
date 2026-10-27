"""Train on development data, freeze the protocol, then run held-out evaluation."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick',action='store_true')
    parser.add_argument('--output-dir',default='output/joint_paper')
    parser.add_argument('--training-slots',type=int)
    args=parser.parse_args()
    for variable in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:
        os.environ[variable]='1'
    root=Path(args.output_dir)
    models=root/'models';evaluation=root/'evaluation';protocol=root/'frozen_protocol.json'
    train=[sys.executable,'-m','experiments.train_joint','--output-dir',str(models)]
    if args.quick:
        train+=['--quick']
    if args.training_slots:
        train+=['--training-slots',str(args.training_slots)]
    subprocess.run(train,check=True)
    command=[sys.executable,'-m','experiments.run_joint','--models-dir',str(models),
             '--output-dir',str(evaluation),'--max-evaluations','10000']
    if args.quick:
        subprocess.run(command+['--quick'],check=True)
    else:
        command+=['--rigorous','--role','final','--protocol',str(protocol)]
        if not protocol.exists():
            subprocess.run(command+['--freeze-protocol'],check=True)
        subprocess.run(command,check=True)


if __name__=='__main__':
    main()
