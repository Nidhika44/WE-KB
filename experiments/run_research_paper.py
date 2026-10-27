"""Freeze and run the original-family research extension on reserved seeds."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick',action='store_true')
    parser.add_argument('--output-dir')
    parser.add_argument('--require-gurobi',action='store_true')
    args=parser.parse_args()
    for name in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:
        os.environ[name]='1'
    output=Path(args.output_dir or ('output/research_paper_smoke' if args.quick else 'output/research_paper'))
    command=[sys.executable,'-m','experiments.run_research','--output-dir',str(output)]
    if args.require_gurobi:
        command+=['--require-gurobi']
    if args.quick:
        subprocess.run(command+['--quick'],check=True)
    else:
        command+=['--rigorous','--role','final']
        if not (output/'frozen_protocol.json').exists():
            subprocess.run(command+['--freeze-protocol'],check=True)
        subprocess.run(command,check=True)


if __name__=='__main__':
    main()
